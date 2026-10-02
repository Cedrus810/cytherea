"""Ensemble frames and the weighted frame pool initial conditions are drawn
from (Task 5).

An `EnsembleFrame` is one snapshot from wherever prior sampling produced it
(equilibration, a previous WE iteration, an experimental ensemble, ...):
just coordinates plus enough bookkeeping (`weight`, `source_id`, `frame_id`,
`time`, the `topology_ref` it goes with) to reconstruct provenance later.

`EnsembleFrame.time` is provenance only: where the frame sits in its source
trajectory. It is never a shot's clock origin. `EnsembleFrameSampler` always
starts the shot at `t = 0.0` and carries the frame time in
`InitialState.meta["frame_time"]` (contract K1).

`EnsembleFramePool` looks a frame up by its `frame_id` (`get`) and picks one,
weighted, optionally restricted to a caller-supplied "state" partition
(`choose`). It does not turn a frame into an `MDState`; that is
`EnsembleFrameSampler`'s job (`sampler.py`), since only the sampler knows
about masses, kT and velocities.

Pool invariants, checked at construction:
- at least one frame;
- `frame_id`s are distinct non-negative integers. `-1` is reserved by
  `ShotKey.frame_id` for "draw a frame by weight" (contract K3), and a
  duplicate id would merge distinct frames in a cluster bootstrap over frame
  ids;
- weights are finite, non-negative and have a positive sum.

The frame sequence and weight array are immutable after construction (a
tuple and a read-only array), so the weights cannot drift out of sync with
the frames. `state_of` is evaluated once per frame, on first use, and cached
together with each state's normalised probabilities. `choose(state=...)`
therefore costs O(1) Python calls per draw instead of one `state_of` call per
frame per draw. Probabilities are normalised by the largest weight first, so
tiny (even subnormal) weights keep their relative precision.

Scale note: frames hold materialised coordinates, so a protein-scale pool
(1e5 frames x 5e4 atoms) does not fit in memory. Lazy frames are Phase B work.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Hashable, Iterable, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class EnsembleFrame:
    coordinates: np.ndarray
    box: np.ndarray | None
    topology_ref: str
    temperature: float
    weight: float
    source_id: str
    frame_id: int
    time: float


def frames_sha256(frames: Iterable[EnsembleFrame]) -> str:
    """sha256 of the frame content that defines a shot protocol (ruling R39):
    per frame its id, weight and topology_ref, its coordinates and its box
    (float64 bytes with shapes). `EnsembleFramePool.content_sha256` caches
    it per pool."""
    h = hashlib.sha256()
    for f in frames:
        h.update(f"{int(f.frame_id)}\x1f{float(f.weight)!r}\x1f{f.topology_ref}\x1f".encode())
        coords = np.ascontiguousarray(np.asarray(f.coordinates, dtype=np.float64))
        h.update(repr(coords.shape).encode() + memoryview(coords).cast("B"))
        if f.box is None:
            h.update(b"nobox")
        else:
            box = np.ascontiguousarray(np.asarray(f.box, dtype=np.float64))
            h.update(repr(box.shape).encode() + memoryview(box).cast("B"))
    return h.hexdigest()


def _frozen_frame(f: EnsembleFrame) -> EnsembleFrame:
    """`f` with its coordinate and box arrays made read-only (in place when
    they already are numpy arrays), so the pool's cached content digest can
    never go stale (int2-m3)."""
    coords = np.asarray(f.coordinates)
    box = None if f.box is None else np.asarray(f.box)
    for a in (coords, box):
        if a is not None:
            a.setflags(write=False)
    if coords is f.coordinates and box is f.box:
        return f
    return dataclasses.replace(f, coordinates=coords, box=box)


def _normalised(weights: np.ndarray) -> np.ndarray:
    """w / sum(w), computed as (w / max w) / sum(w / max w)."""
    scaled = weights / float(np.max(weights))
    out = scaled / scaled.sum()
    out.setflags(write=False)
    return out


class EnsembleFramePool:
    """A weighted collection of `EnsembleFrame`s, optionally partitioned into
    "states" by `state_of`.

    `choose` draws a single frame with probability proportional to
    `frame.weight`, restricted to `state_of(f) == state` when `state` is
    given. All randomness is consumed from the caller-supplied `rng` (a
    `numpy.random.Generator`). This class never seeds its own RNG, so
    reproducibility is entirely the caller's responsibility (see
    `EnsembleFrameSampler`, which supplies `derive_rng`-derived generators).
    """

    def __init__(
        self,
        frames: Sequence[EnsembleFrame],
        state_of: Callable[[EnsembleFrame], Hashable] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("EnsembleFramePool requires at least one frame")
        self._frames: tuple[EnsembleFrame, ...] = tuple(_frozen_frame(f) for f in frames)
        self._content_sha256: str | None = None
        self.state_of = state_of

        index: dict[int, int] = {}
        for i, f in enumerate(self._frames):
            fid = f.frame_id
            if isinstance(fid, bool) or not isinstance(fid, (int, np.integer)):
                raise ValueError(f"frame_id must be an int, got {fid!r}")
            fid = int(fid)
            if fid < 0:
                raise ValueError(
                    "frame_id must be >= 0 (-1 is reserved by ShotKey.frame_id "
                    f"for a weighted draw); got {fid}"
                )
            if fid in index:
                raise ValueError(f"duplicate frame_id {fid} in EnsembleFramePool")
            index[fid] = i
        self._index = index
        ids = np.array([int(f.frame_id) for f in self._frames], dtype=np.int64)
        ids.setflags(write=False)
        self._ids = ids

        weights = np.array([float(f.weight) for f in self._frames], dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("frame weights must be finite and non-negative")
        if not np.max(weights) > 0:  # (max, not sum: the sum may overflow)
            raise ValueError("frame weights must sum to a positive value")
        weights.setflags(write=False)
        self._weights = weights
        self._probs_all = _normalised(weights)

        # Lazily computed: one state label per frame, and per state its
        # (frame indices, probabilities or None when all weights are 0).
        self._labels: list[Hashable] | None = None
        self._by_state: dict[Hashable, tuple[np.ndarray, np.ndarray | None]] = {}

    # -- read-only views ----------------------------------------------------

    @property
    def frames(self) -> tuple[EnsembleFrame, ...]:
        return self._frames

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def frame_ids(self) -> np.ndarray:
        return self._ids

    def __len__(self) -> int:
        return len(self._frames)

    def content_sha256(self) -> str:
        """`frames_sha256` of this pool's frames, computed once (int2-m3).
        Safe to cache: frames are frozen dataclasses and the pool makes
        their arrays read-only at construction (writing through another
        view of the same memory is the caller's responsibility)."""
        if self._content_sha256 is None:
            self._content_sha256 = frames_sha256(self._frames)
        return self._content_sha256

    # -- lookup -------------------------------------------------------------

    def get(self, frame_id: int) -> EnsembleFrame:
        """The frame whose `frame_id` equals `frame_id`; `KeyError` if absent."""
        try:
            return self._frames[self._index[int(frame_id)]]
        except KeyError:
            raise KeyError(
                f"no frame with frame_id={frame_id!r} in EnsembleFramePool "
                f"({len(self._frames)} frames)"
            ) from None

    def state_label(self, frame: EnsembleFrame) -> Hashable | None:
        """Cached `state_of(frame)`, or `None` if the pool has no `state_of`.
        `frame` must belong to this pool (looked up by its `frame_id`)."""
        if self.state_of is None:
            return None
        return self._state_labels()[self._index[int(frame.frame_id)]]

    def _state_labels(self) -> list[Hashable]:
        if self._labels is None:
            assert self.state_of is not None
            self._labels = [self.state_of(f) for f in self._frames]
        return self._labels

    def _partition(self, state: Hashable | None) -> tuple[np.ndarray, np.ndarray]:
        """(frame indices, probabilities) of `state`, or of the whole pool
        when `state` is None. Raises a named `ValueError` for an empty or
        all-zero-weight state."""
        if state is None:
            return np.arange(len(self._frames)), self._probs_all
        if self.state_of is None:
            raise ValueError(
                "EnsembleFramePool.choose(state=...) requires "
                "state_of to have been provided at construction"
            )
        cached = self._by_state.get(state)
        if cached is None:
            labels = self._state_labels()
            idxs = np.array([i for i, lab in enumerate(labels) if lab == state], dtype=np.intp)
            probs = None
            if idxs.size and np.max(self._weights[idxs]) > 0:
                probs = _normalised(self._weights[idxs])
            cached = (idxs, probs)
            self._by_state[state] = cached
        idxs, probs = cached
        if idxs.size == 0:
            raise ValueError(
                f"no frames in EnsembleFramePool with state_of(f) == {state!r}"
            )
        if probs is None:
            raise ValueError(
                f"all frames in EnsembleFramePool with state_of(f) == {state!r} "
                "have zero weight; cannot draw from that state"
            )
        return idxs, probs

    # -- weighted draw ------------------------------------------------------

    def choose(
        self, rng: np.random.Generator, state: Hashable | None = None
    ) -> EnsembleFrame:
        idxs, probs = self._partition(state)
        j = int(rng.choice(len(idxs), p=probs))
        return self._frames[int(idxs[j])]

    def weight_fraction(self, frame_ids: Iterable[int], state: Hashable | None = None) -> float:
        """Fraction of the pool weight (restricted to `state` when given)
        carried by the frames in `frame_ids`. Frames outside `state`
        contribute nothing."""
        idxs, probs = self._partition(state)
        sel = np.isin(self._ids[idxs], np.fromiter((int(f) for f in frame_ids), dtype=np.int64))
        return float(np.sum(probs[sel]))


# ---------------------------------------------------------------------------
# Frame files (Task 13: written by ``prepare``, read by ``ic.kind: frames``)
# ---------------------------------------------------------------------------

_FRAMES_FORMAT = "cytherea-frames/1"


def save_frames(path, frames: Sequence[EnsembleFrame]) -> None:
    """Write `frames` to a ``.npz`` file (uncompressed, no pickles): stacked
    coordinates (internal units), boxes (all or none), weights, ids, times,
    temperatures, source ids and the shared ``topology_ref``."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to save")
    refs = {f.topology_ref for f in frames}
    if len(refs) != 1:
        raise ValueError(f"frames of one file share one topology_ref, got {sorted(refs)}")
    boxes = [f.box for f in frames]
    if any(b is None for b in boxes) and not all(b is None for b in boxes):
        raise ValueError("either every frame has a box or none has")
    arrays = {
        "format": np.array(_FRAMES_FORMAT),
        "coordinates": np.stack([np.asarray(f.coordinates, dtype=np.float64) for f in frames]),
        "weight": np.array([f.weight for f in frames], dtype=np.float64),
        "frame_id": np.array([f.frame_id for f in frames], dtype=np.int64),
        "time": np.array([f.time for f in frames], dtype=np.float64),
        "temperature": np.array([f.temperature for f in frames], dtype=np.float64),
        "source_id": np.array([f.source_id for f in frames]),
        "topology_ref": np.array(refs.pop()),
    }
    if boxes[0] is not None:
        arrays["box"] = np.stack([np.asarray(b, dtype=np.float64) for b in boxes])
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


def load_frames(path) -> list[EnsembleFrame]:
    """Read a `save_frames` file."""
    with np.load(path, allow_pickle=False) as z:
        if "format" not in z.files or str(z["format"]) != _FRAMES_FORMAT:
            raise ValueError(f"{path}: not a {_FRAMES_FORMAT} file")
        box = z["box"] if "box" in z.files else None
        ref = str(z["topology_ref"])
        return [
            EnsembleFrame(
                coordinates=z["coordinates"][i].copy(),
                box=None if box is None else box[i].copy(),
                topology_ref=ref,
                temperature=float(z["temperature"][i]),
                weight=float(z["weight"][i]),
                source_id=str(z["source_id"][i]),
                frame_id=int(z["frame_id"][i]),
                time=float(z["time"][i]),
            )
            for i in range(z["coordinates"].shape[0])
        ]
