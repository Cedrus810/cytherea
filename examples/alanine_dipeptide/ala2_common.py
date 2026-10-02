"""Shared helpers for the A1 alanine-dipeptide *reference* scripts.

Deliberately plain OpenMM + numpy: nothing here imports ``cytherea``, because
the long reference trajectory is the ground truth the engine is checked
against and must not share code with it.

Contents
--------
* platform construction (CUDA mixed + DeterministicForces, or CPU for smoke runs)
* phi/psi atom lookup and a vectorised dihedral
* the ``phipsi.bin`` record format (fixed 16-byte records) + loader/truncation
* DCD truncation (so ``--resume`` can drop frames written after the last checkpoint)
* THE conformational-state definition (core boxes, core labels, transition-based
  assignment, the 1 ps observation interval, ``label_shot``) shared by the 14a
  reference analysis and the 14b shooting labels
* ``FrameIndex``: DCD frame <-> MD step <-> phi/psi record index
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import openmm

# ---------------------------------------------------------------------------
# Physics constants shared by build_system.py and ref_long.py
# ---------------------------------------------------------------------------
TEMPERATURE_K = 300.0
TIMESTEP_PS = 0.002          # 2 fs
PRESSURE_BAR = 1.0
NONBONDED_CUTOFF_NM = 0.9
#: Minimum distance from the solute's bounding sphere to every face of the cubic
#: box (GROMACS ``editconf -d`` / AMBER ``solvatebox`` meaning): edge = 2 r + 2 pad.
#: NOTE OpenMM Modeller's own ``padding=p`` means edge = max(2 r + p, 2 p), i.e.
#: p is the solute-to-periodic-image distance; p = 1.0 nm would give a 2.0 nm box
#: (barely above 2 x cutoff), so build_system passes an explicit boxSize instead.
PADDING_NM = 1.0
FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")

# ---------------------------------------------------------------------------
# phi/psi record file
# ---------------------------------------------------------------------------
#: One record per saved frame: MD step (int64), phi and psi in DEGREES (float32),
#: range (-180, 180]. Little-endian, no header -> file size is a multiple of 16.
PHIPSI_DTYPE = np.dtype([("step", "<i8"), ("phi", "<f4"), ("psi", "<f4")])


def make_platform(name: str):
    """Return (platform, properties). No silent fallback between platforms."""
    platform = openmm.Platform.getPlatformByName(name)
    if name == "CUDA":
        props = {"Precision": "mixed", "DeterministicForces": "true"}
    elif name == "OpenCL":
        props = {"Precision": "mixed"}
    elif name == "CPU":
        props = {}
    elif name == "Reference":
        props = {}
    else:
        raise ValueError(f"unsupported platform {name!r}")
    return platform, props


def phi_psi_indices(topology) -> tuple[list[int], list[int]]:
    """phi = C(ACE)-N(ALA)-CA(ALA)-C(ALA); psi = N(ALA)-CA(ALA)-C(ALA)-N(NME)."""
    atoms = {}
    for atom in topology.atoms():
        atoms[(atom.residue.name, atom.name)] = atom.index
    try:
        phi = [atoms[("ACE", "C")], atoms[("ALA", "N")], atoms[("ALA", "CA")], atoms[("ALA", "C")]]
        psi = [atoms[("ALA", "N")], atoms[("ALA", "CA")], atoms[("ALA", "C")], atoms[("NME", "N")]]
    except KeyError as exc:  # pragma: no cover - topology is fixed
        raise RuntimeError(f"alanine-dipeptide atom not found in topology: {exc}") from exc
    return phi, psi


def dihedral_deg(xyz: np.ndarray, idx: list[int]) -> float:
    """IUPAC dihedral of four atoms (positions in any length unit), degrees in (-180, 180].

    ``xyz`` must hold the molecule *unwrapped* (OpenMM getState without
    enforcePeriodicBox keeps bonded atoms contiguous).
    """
    p0, p1, p2, p3 = (np.asarray(xyz[i], dtype=np.float64) for i in idx)
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def load_phipsi(path: str | os.PathLike) -> np.ndarray:
    """Load all *complete* records (a torn trailing partial record is ignored)."""
    path = Path(path)
    size = path.stat().st_size
    n = size // PHIPSI_DTYPE.itemsize
    if n == 0:
        return np.zeros(0, dtype=PHIPSI_DTYPE)
    return np.fromfile(path, dtype=PHIPSI_DTYPE, count=n)


def truncate_phipsi(path: str | os.PathLike, n_records: int) -> None:
    with open(path, "r+b") as fh:
        fh.truncate(n_records * PHIPSI_DTYPE.itemsize)
        fh.flush()
        os.fsync(fh.fileno())


def write_phipsi_meta(path: str | os.PathLike, phi_idx, psi_idx, interval_steps: int) -> None:
    meta = {
        "format": "raw little-endian records, no header",
        "dtype": [["step", "<i8"], ["phi", "<f4"], ["psi", "<f4"]],
        "record_bytes": PHIPSI_DTYPE.itemsize,
        "units": {"step": "MD steps (dt = %g ps)" % TIMESTEP_PS, "phi": "degrees", "psi": "degrees"},
        "angle_range": "(-180, 180]",
        "phi_atoms": {"indices": phi_idx, "names": "ACE:C ALA:N ALA:CA ALA:C"},
        "psi_atoms": {"indices": psi_idx, "names": "ALA:N ALA:CA ALA:C NME:N"},
        "interval_steps": interval_steps,
        "interval_ps": interval_steps * TIMESTEP_PS,
        "first_record_step": 0,
        "load": "np.fromfile(path, dtype=[('step','<i8'),('phi','<f4'),('psi','<f4')])",
    }
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# DCD helpers (layout as written by openmm.app.DCDFile)
# ---------------------------------------------------------------------------
def _dcd_layout(fh) -> tuple[int, int, int, bool]:
    """Return (header_bytes, frame_bytes, n_frames_in_header, has_box)."""
    fh.seek(0)
    magic = fh.read(8)
    if magic[4:8] != b"CORD" or struct.unpack("<i", magic[:4])[0] != 84:
        raise ValueError("not an OpenMM DCD file")
    n_frames = struct.unpack("<i", fh.read(4))[0]
    fh.seek(48)
    has_box = struct.unpack("<i", fh.read(4))[0] != 0
    fh.seek(92)
    comment_bytes = struct.unpack("<i", fh.read(4))[0]
    fh.seek(104 + comment_bytes)
    n_atoms = struct.unpack("<i", fh.read(4))[0]
    header = 104 + comment_bytes + 8
    frame = (56 if has_box else 0) + 3 * (8 + 4 * n_atoms)
    return header, frame, n_frames, has_box


def dcd_n_frames(path: str | os.PathLike) -> int:
    """Number of complete frames physically present in the file."""
    with open(path, "rb") as fh:
        header, frame, _, _ = _dcd_layout(fh)
        fh.seek(0, os.SEEK_END)
        return (fh.tell() - header) // frame


def truncate_dcd(path: str | os.PathLike, n_frames: int) -> None:
    """Keep the first ``n_frames`` frames and make the header consistent."""
    with open(path, "r+b") as fh:
        header, frame, _, _ = _dcd_layout(fh)
        fh.seek(12)
        first_step, interval = struct.unpack("<ii", fh.read(8))
        fh.truncate(header + n_frames * frame)
        fh.seek(8)
        fh.write(struct.pack("<i", n_frames))
        fh.seek(20)
        fh.write(struct.pack("<i", first_step + max(n_frames - 1, 0) * interval))
        fh.flush()
        os.fsync(fh.fileno())


def fsync_path(path: str | os.PathLike) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | os.PathLike) -> None:
    """Best-effort fsync of a directory (makes a preceding rename durable).

    Some filesystems (e.g. certain NFS setups) refuse fsync on a directory; that is
    ignored -- the rename itself is still atomic, only its durability after a power
    loss is not guaranteed there.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Conformational states: the ONE definition used by 14a (reference MSM) and
# 14b (labels of shooting trajectories).  Degrees, phi/psi in (-180, 180],
# closed intervals.
# ---------------------------------------------------------------------------
STATE_NAMES = ["C7eq/C5", "alphaR", "alphaL"]
N_STATES = 3
CORE_BOXES_DEG = {
    "C7eq/C5": {"phi": [[-180, -30]], "psi": [[100, 180], [-180, -160]]},
    "alphaR": {"phi": [[-180, -30]], "psi": [[-80, -10]]},
    "alphaL": {"phi": [[30, 100]], "psi": [[0, 90]]},
}
NONCORE = -1

#: Observation (labelling) interval of the state contract.  Transition-based labels
#: depend on how often phi/psi are sampled (a short visit to another core between two
#: samples is missed): on the A1 data the core-start T_01(100 ps) is 0.103 at 1 ps,
#: 0.099 at 5 ps and 0.093 at 10 ps.  The reference phipsi.bin is written every 1 ps
#: and 14b MUST label its shots from phi/psi sampled at exactly this interval.
#: 14b then compares the ROW-NORMALISED core-start T_ij = C_ij / sum_j C_ij at the
#: reference's shoot_lag_ps (analysis.json core_start.msm.transition_matrix_row_normalised);
#: the reversible MLE depends on the per-state number of shots and is for ITS only.
OBS_INTERVAL_PS = 1.0
OBS_INTERVAL_STEPS = int(round(OBS_INTERVAL_PS / TIMESTEP_PS))   # 500

#: 14b contract (ruling R40): the lag grid on which analyze_ref chooses 14b's FixedLag
#: tau (shoot_lag_ps).  The choice is always a lag of the grid, so it moves with the
#: grid's granularity (19.53 ns: 100 ps on this grid, 70 ps on a denser one); 14b fixes
#: tau on THIS grid.  Any other --lags-ps (e.g. a dense grid) is diagnostic only, and
#: analysis.json then reports contract_14b.lag_grid_matches_contract = false.  Lags
#: longer than half a bootstrap block are dropped at run time (listed in lags_dropped_ps).
SHOOT_LAG_GRID_PS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def lag_grid_is_contract(lags_ps) -> bool:
    """True iff ``lags_ps`` (iterable of ps) is exactly `SHOOT_LAG_GRID_PS` as a set."""
    got = sorted({float(x) for x in lags_ps})
    want = sorted(SHOOT_LAG_GRID_PS)
    return len(got) == len(want) and all(abs(a - b) <= 1e-9 * b for a, b in zip(got, want))


def check_obs_interval(interval_ps: float) -> None:
    """Raise ValueError unless ``interval_ps`` is the contract's 1 ps labelling interval."""
    if not abs(float(interval_ps) - OBS_INTERVAL_PS) <= 1e-9 * OBS_INTERVAL_PS:
        raise ValueError(f"state labels must be computed from phi/psi sampled every {OBS_INTERVAL_PS:g} ps "
                         f"(OBS_INTERVAL_PS, the reference's interval); got {interval_ps!r} ps")


def _in_any(x, intervals):
    m = np.zeros(x.shape, dtype=bool)
    for lo, hi in intervals:
        m |= (x >= lo) & (x <= hi)
    return m


def core_labels(phi, psi) -> np.ndarray:
    """Raw core label per frame: state index, or NONCORE (-1) outside all cores."""
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    lab = np.full(phi.shape, NONCORE, dtype=np.int64)
    for s, name in enumerate(STATE_NAMES):
        box = CORE_BOXES_DEG[name]
        m = _in_any(phi, box["phi"]) & _in_any(psi, box["psi"])
        if np.any(lab[m] != NONCORE):
            raise RuntimeError("core boxes overlap")
        lab[m] = s
    return lab


def transition_based_assignment(core, initial_label: int | None = None) -> tuple[np.ndarray, int]:
    """Label every frame by the LAST core visited (core-set / milestoning assignment).

    ``core`` is the raw core-label series (NONCORE outside cores).
    * ``initial_label=None`` (reference trajectory): leading NONCORE frames before
      the first core entry are dropped; returns (labels, n_dropped).
    * ``initial_label=s`` (a 14b shot started from a frame whose label is s --
      normally its raw core label, since shots start inside a core): leading
      NONCORE frames get label s and nothing is dropped; returns (labels, 0).
    """
    core = np.asarray(core, dtype=np.int64)
    idx = np.where(core >= 0, np.arange(core.size), -1)
    np.maximum.accumulate(idx, out=idx)
    if initial_label is not None:
        if not (0 <= int(initial_label) < N_STATES):
            raise ValueError(f"initial_label must be a state index in [0, {N_STATES}), got {initial_label}")
        if core.size and core[0] >= 0 and core[0] != int(initial_label):
            raise ValueError(f"initial_label {initial_label} disagrees with the raw core label {int(core[0])} "
                             "of the first observation (a shot's first observation is its start frame)")
        out = np.where(idx >= 0, core[np.maximum(idx, 0)], int(initial_label))
        return out.astype(np.int64), 0
    first = int(np.argmax(idx >= 0)) if np.any(idx >= 0) else core.size
    return core[idx[first:]], first


def label_shot(phi_deg, psi_deg, start_label: int, interval_ps: float) -> np.ndarray:
    """14b shot labels: TBA seeded with the start core, from phi/psi sampled every
    ``interval_ps`` (must be OBS_INTERVAL_PS).  Element 0 is the start frame."""
    check_obs_interval(interval_ps)
    lab, _ = transition_based_assignment(core_labels(phi_deg, psi_deg), initial_label=start_label)
    return lab


# ---------------------------------------------------------------------------
# Frame bookkeeping: DCD frame k <-> MD step <-> phi/psi record index
# ---------------------------------------------------------------------------
class FrameIndex:
    """Index arithmetic for a ref_long.py run directory.

    phi/psi record r is at step r * phipsi_interval_steps (record 0 = step 0).
    DCD frame k (0-based) is at step (k + 1) * dcd_interval_steps (no frame at step 0).
    """

    def __init__(self, phipsi_interval_steps: int, dcd_interval_steps: int, timestep_ps: float = TIMESTEP_PS):
        if dcd_interval_steps % phipsi_interval_steps:
            raise ValueError("dcd interval must be a multiple of the phi/psi interval")
        self.phipsi_interval_steps = int(phipsi_interval_steps)
        self.dcd_interval_steps = int(dcd_interval_steps)
        self.timestep_ps = float(timestep_ps)

    @classmethod
    def from_run(cls, run_dir: str | os.PathLike) -> "FrameIndex":
        meta = json.loads((Path(run_dir) / "run.json").read_text())
        return cls(meta["phipsi_interval_steps"], meta["dcd_interval_steps"], meta.get("timestep_ps", TIMESTEP_PS))

    def dcd_frame_to_step(self, k):
        k = np.asarray(k)
        if np.any(k < 0):
            raise ValueError("DCD frame index must be >= 0")
        return (k + 1) * self.dcd_interval_steps

    def step_to_dcd_frame(self, step):
        step = np.asarray(step)
        if np.any(step % self.dcd_interval_steps) or np.any(step < self.dcd_interval_steps):
            raise ValueError("step is not on a DCD frame")
        return step // self.dcd_interval_steps - 1

    def step_to_record(self, step):
        step = np.asarray(step)
        if np.any(step % self.phipsi_interval_steps) or np.any(step < 0):
            raise ValueError("step is not on a phi/psi record")
        return step // self.phipsi_interval_steps

    def record_to_step(self, r):
        return np.asarray(r) * self.phipsi_interval_steps

    def dcd_frame_to_record(self, k):
        return self.step_to_record(self.dcd_frame_to_step(k))

    def record_to_dcd_frame(self, r):
        return self.step_to_dcd_frame(self.record_to_step(r))

    def step_to_time_ps(self, step):
        return np.asarray(step) * self.timestep_ps
