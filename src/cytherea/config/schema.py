"""Run configuration (Task 13): pydantic models, `load_config`, `config_hash`.

A config file is YAML with the top-level keys of `RunConfig`. Every model
forbids unknown keys (a typo is an error, not a silently ignored field).
Dimensioned fields follow `cytherea.config.units` -- under ``units: openmm``
they are ``"<number> <unit>"`` strings, under ``units: reduced`` plain
numbers -- and are stored in internal units (nm, ps, kJ/mol, K, rad), so
``b: "20 angstrom"`` and ``b: "2 nm"`` load to the same config and the same
`RunConfig.config_hash`. Region thresholds take the dimension of the
observable they test (a distance: length; a dihedral: angle; a coordinate:
length, or a plain number under ``units: reduced``).

Use `load_config(path)` or `RunConfig.from_dict(data)`: both pass the unit
system to the field validators (pydantic validation context); validating a
`RunConfig` without it raises.

Mode rules (checked after the fields):

* ``prepare``: an OpenMM system, ``physics.purpose: equilibration``,
  ``ic.kind: structure`` (the topology PDB's positions), and
  ``budget.output_frames``; writes an ensemble-frame file
  (`cytherea.ic.frames.save_frames`).
* ``shoot.ensemble``: ``ic.kind`` points or frames; stop ``fixed_lag`` (T(tau))
  or ``absorbing_AB`` (committor).
* ``shoot.surface``: ``ic.kind`` points or frames on the dividing surface;
  stop ``absorbing_AB``.
* ``shoot.encounter``: ``ic.kind: encounter`` and stop ``b_surface``;
  ``budget.shots_per_frame`` is the number of shots (keys
  ``ShotKey(seed, -1, shot_id, stage)``: both frames drawn by weight).
* every shoot mode: ``physics.purpose: measurement``, ``observables``,
  ``stop`` and ``store_path``.
* analytic systems need ``units: reduced`` and ``physics.kind: analytic``;
  OpenMM systems ``units: openmm`` and ``physics.kind: openmm``.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, model_validator

from cytherea.config.units import parse_quantity
from cytherea.store import config_hash as _config_hash

ANALYTIC_POTENTIALS = (
    "FreeParticle", "DoubleWell1D", "DoubleWell2D", "MullerBrown", "ChannelDoubleWell2D", "Harmonic",
    "LJCluster",
)


def _quantity(dimension: str):
    def parse(value, info: ValidationInfo):
        units = (info.context or {}).get("units")
        if units is None:
            raise ValueError(
                "the unit system is unknown: build configs with load_config(path) or "
                "RunConfig.from_dict(data)"
            )
        return parse_quantity(value, dimension, units)

    return BeforeValidator(parse)


Length = Annotated[float, _quantity("length")]
Time = Annotated[float, _quantity("time")]
InvTime = Annotated[float, _quantity("inverse_time")]
Temperature = Annotated[float, _quantity("temperature")]
Energy = Annotated[float, _quantity("energy")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- system


class AnalyticPotentialSpec(_Model):
    name: Literal[ANALYTIC_POTENTIALS]  # type: ignore[valid-type]
    params: dict[str, int | float] = Field(default_factory=dict)  # ints stay ints (LJCluster.n_atoms)


class AnalyticSystem(_Model):
    kind: Literal["analytic"]
    potential: AnalyticPotentialSpec


class OpenMMSystem(_Model):
    kind: Literal["openmm"]
    system_xml: str
    topology_pdb: str


SystemSpec = Annotated[Union[AnalyticSystem, OpenMMSystem], Field(discriminator="kind")]


# --------------------------------------------------------------- physics


class AnalyticPhysics(_Model):
    kind: Literal["analytic"]
    integrator: Literal["overdamped", "baoab"]
    dt: Time
    kT: float = Field(gt=0)
    gamma: float = Field(ge=0)
    mass: float = Field(1.0, gt=0)
    purpose: Literal["equilibration", "measurement"] = "measurement"


class OpenMMPhysics(_Model):
    kind: Literal["openmm"]
    integrator: Literal["verlet", "langevin_middle", "nose_hoover"]
    dt: Time
    temperature: Temperature
    friction: InvTime
    constraints: Literal["none", "hbonds", "allbonds"]
    rigid_water: bool
    platform: Literal["CUDA", "CPU", "Reference"]
    precision: Literal["mixed", "double", "single"]
    deterministic_forces: bool = True
    purpose: Literal["equilibration", "measurement"]

    def to_physics_config(self):
        from cytherea.backends.base import PhysicsConfig

        return PhysicsConfig(
            integrator=self.integrator, dt_ps=self.dt, temperature_K=self.temperature,
            friction_per_ps=self.friction, constraints=self.constraints, rigid_water=self.rigid_water,
            platform=self.platform, precision=self.precision,
            deterministic_forces=self.deterministic_forces, purpose=self.purpose,
        )


PhysicsSpec = Annotated[Union[AnalyticPhysics, OpenMMPhysics], Field(discriminator="kind")]


# -------------------------------------------------------------------- ic


class PointsIC(_Model):
    """Fixed starting points (analytic toys): frame_id = point index."""

    kind: Literal["points"]
    points: list[list[float]] = Field(min_length=1)


class FramesIC(_Model):
    """An ensemble-frame file (`cytherea.ic.frames.save_frames`), e.g. from ``prepare``."""

    kind: Literal["frames"]
    path: str
    energy_window: tuple[Energy, Energy] | None = None
    min_pair_dist: Length | None = None
    remove_com_momentum: bool = True
    constraints: Literal["none", "from_system"] = "from_system"
    max_redraws: int = Field(20, ge=0)


class StructureIC(_Model):
    """``prepare``: start from the topology PDB's positions."""

    kind: Literal["structure"]


class EncounterIC(_Model):
    """``shoot.encounter``: frames of A and of B (separate frame files, atoms in
    the order of the combined system: A then B), B placed on the b sphere
    (`cytherea.ic.encounter.EncounterSampler`). ``label`` = (state of A,
    state of B), recorded as the origin label."""

    kind: Literal["encounter"]
    frames_A: str
    frames_B: str
    b: Length
    min_pair_dist: Length
    label: tuple[int, int] = (0, 0)
    energy_window: tuple[Energy, Energy] | None = None
    constraints: Literal["none", "from_system"] = "from_system"
    max_redraws: int = Field(20, ge=0)


ICSpec = Annotated[Union[PointsIC, FramesIC, StructureIC, EncounterIC], Field(discriminator="kind")]


# ----------------------------------------------------------- observables


class ObservableSpec(_Model):
    """``coord``: component ``index`` of the flattened coordinates; ``distance``:
    atoms (i, j), minimum image when ``periodic`` and the state has a box;
    ``dihedral``: atoms (a, b, c, d), radians in (-pi, pi]; ``com_distance``:
    distance between the mass-weighted centres of the atom-index ranges
    ``ranges = [[a0, a1], [b0, b1]]`` (half-open), minimum image as distance."""

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    kind: Literal["coord", "distance", "dihedral", "com_distance"]
    index: int | None = Field(None, ge=0)
    atoms: list[int] | None = None
    ranges: list[tuple[int, int]] | None = None
    periodic: bool = True

    @model_validator(mode="after")
    def _shape(self):
        if self.kind == "coord":
            if self.index is None or self.atoms is not None or self.ranges is not None:
                raise ValueError("a coord observable takes `index` (and no `atoms` / `ranges`)")
        elif self.kind == "com_distance":
            if self.ranges is None or len(self.ranges) != 2 or self.index is not None or self.atoms is not None:
                raise ValueError("a com_distance observable takes `ranges: [[a0, a1], [b0, b1]]` only")
            (a0, a1), (b0, b1) = self.ranges
            if not (0 <= a0 < a1 and 0 <= b0 < b1) or (a0 < b1 and b0 < a1):
                raise ValueError("com_distance ranges must be non-empty, non-negative and disjoint")
        else:
            if self.ranges is not None:
                raise ValueError(f"a {self.kind} observable takes no `ranges`")
            n = 2 if self.kind == "distance" else 4
            if self.atoms is None or len(self.atoms) != n or self.index is not None:
                raise ValueError(f"a {self.kind} observable takes `atoms` with {n} indices (and no `index`)")
            if len(set(self.atoms)) != n or min(self.atoms) < 0:
                raise ValueError(f"{self.kind} atoms must be {n} distinct indices >= 0")
        return self

    @property
    def dimension(self) -> str:
        return "angle" if self.kind == "dihedral" else "length"


class ObsSpecModel(_Model):
    dt_obs: Time
    store_stride: int = Field(1, ge=1)
    items: list[ObservableSpec] = Field(min_length=1)


# ------------------------------------------------------------------ stop


class Condition(_Model):
    """``observable <= le`` or ``observable >= ge`` (exactly one); the threshold
    has the observable's dimension and is converted after validation."""

    observable: str
    le: float | str | None = None
    ge: float | str | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.le is None) == (self.ge is None):
            raise ValueError("a condition takes exactly one of `le` / `ge`")
        return self


class RegionSpec(_Model):
    """All conditions hold (a box in observable space)."""

    name: str
    all_of: list[Condition] = Field(min_length=1)


class FixedLagSpec(_Model):
    kind: Literal["fixed_lag"]
    tau: Time


class AbsorbingABSpec(_Model):
    kind: Literal["absorbing_AB"]
    A: RegionSpec
    B: RegionSpec
    tau_persist: Time
    t_max: Time


class BSurfaceSpec(_Model):
    kind: Literal["b_surface"]
    reaction: RegionSpec
    r_observable: str
    q: Length
    tau_persist: Time
    t_max: Time


StopSpec = Annotated[Union[FixedLagSpec, AbsorbingABSpec, BSurfaceSpec], Field(discriminator="kind")]


# ------------------------------------------------------------- resampler


class NoResampler(_Model):
    kind: Literal["none"] = "none"


class WESpec(_Model):
    """BinnedWE on one progress observable; ``bin_edges`` in its dimension."""

    kind: Literal["we"]
    progress: str
    bin_edges: list[float | str] = Field(min_length=1)
    target_per_bin: int = Field(ge=1)
    tau_seg: Time
    n_iter: int = Field(ge=1)
    walkers_per_frame: int = Field(1, ge=1)


ResamplerSpec = Annotated[Union[NoResampler, WESpec], Field(discriminator="kind")]


# ---------------------------------------------------------------- budget


class BudgetSpec(_Model):
    """Shoot modes: ``shots_per_frame`` shots of every frame in ``frames``
    (``"all"`` or frame ids), keys ``ShotKey(seed, frame_id, shot_id, stage)``.
    ``prepare``: optional minimisation, ``equilibrate`` of burn-in, then
    ``n_frames`` frames every ``frame_interval`` into ``output_frames``."""

    shots_per_frame: int | None = Field(None, ge=1)
    frames: Literal["all"] | list[int] = "all"
    n_workers: int = Field(1, ge=1)
    stage: str = "main"
    minimize: bool = True
    equilibrate: Time | None = None
    n_frames: int | None = Field(None, ge=1)
    frame_interval: Time | None = None
    output_frames: str | None = None


# ------------------------------------------------------------------- run


class RunConfig(_Model):
    mode: Literal["prepare", "shoot.ensemble", "shoot.encounter", "shoot.surface"]
    units: Literal["openmm", "reduced"]
    system: SystemSpec
    physics: PhysicsSpec
    ic: ICSpec
    stop: StopSpec | None = None
    observables: ObsSpecModel | None = None
    resampler: ResamplerSpec = Field(default_factory=NoResampler)
    budget: BudgetSpec
    seed: int
    store_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict, base_dir: str | os.PathLike | None = None) -> "RunConfig":
        """Validate ``data`` (``base_dir``: relative paths in it are relative to this)."""
        if not isinstance(data, dict):
            raise ValueError(f"a config is a mapping, got {type(data).__name__}")
        ctx = {"units": data.get("units"), "base_dir": None if base_dir is None else os.fspath(base_dir)}
        return cls.model_validate(data, context=ctx)

    @model_validator(mode="after")
    def _rules(self, info: ValidationInfo):
        base = (info.context or {}).get("base_dir")
        self._check_kinds()
        self._check_mode()
        if self.observables is not None:
            self._check_observables(info)
        if base:
            self._resolve_paths(base)
        return self

    def _check_kinds(self) -> None:
        if self.system.kind == "analytic":
            if self.units != "reduced":
                raise ValueError("an analytic system needs units: reduced")
        elif self.units != "openmm":
            raise ValueError("an OpenMM system needs units: openmm")
        if self.physics.kind != self.system.kind:
            raise ValueError(f"physics.kind {self.physics.kind!r} does not match system.kind {self.system.kind!r}")
        if self.ic.kind == "points" and self.system.kind != "analytic":
            raise ValueError("ic.kind: points is for analytic systems; use frames for OpenMM")

    def _check_mode(self) -> None:
        m = self.mode
        if m == "prepare":
            if self.system.kind != "openmm":
                raise ValueError("prepare needs an OpenMM system")
            if self.physics.purpose != "equilibration":
                raise ValueError("prepare runs with physics.purpose: equilibration")
            if self.ic.kind != "structure":
                raise ValueError("prepare starts from ic.kind: structure")
            b = self.budget
            if b.output_frames is None or b.n_frames is None or b.frame_interval is None:
                raise ValueError("prepare needs budget.output_frames, budget.n_frames and budget.frame_interval")
            if self.resampler.kind != "none":
                raise ValueError("prepare takes no resampler")
            return
        if self.physics.purpose != "measurement":
            raise ValueError(f"{m} measures dynamics: physics.purpose must be measurement")
        for name in ("stop", "observables", "store_path"):
            if getattr(self, name) is None:
                raise ValueError(f"{m} needs `{name}`")
        if self.resampler.kind == "none" and self.budget.shots_per_frame is None:
            raise ValueError(f"{m} without a resampler needs budget.shots_per_frame")
        allowed = {
            "shoot.ensemble": ({"points", "frames"}, {"fixed_lag", "absorbing_AB"}),
            "shoot.surface": ({"points", "frames"}, {"absorbing_AB"}),
            "shoot.encounter": ({"encounter"}, {"b_surface"}),
        }[m]
        if self.ic.kind not in allowed[0]:
            raise ValueError(f"{m} takes ic.kind in {sorted(allowed[0])}, got {self.ic.kind!r}")
        if self.stop.kind not in allowed[1]:
            raise ValueError(f"{m} takes stop.kind in {sorted(allowed[1])}, got {self.stop.kind!r}")

    def _check_observables(self, info: ValidationInfo) -> None:
        units = (info.context or {}).get("units")
        items = {o.name: o for o in self.observables.items}
        if len(items) != len(self.observables.items):
            raise ValueError("observable names must be distinct")

        def threshold(name: str, value, where: str) -> float:
            if name not in items:
                raise ValueError(f"{where}: unknown observable {name!r}; defined: {sorted(items)}")
            try:
                return parse_quantity(value, items[name].dimension, units)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from None

        def region(r: RegionSpec, where: str) -> None:
            for i, c in enumerate(r.all_of):
                op = "le" if c.le is not None else "ge"
                setattr(c, op, threshold(c.observable, getattr(c, op), f"{where}.all_of[{i}].{op}"))

        s = self.stop
        if s is not None:
            if s.kind == "absorbing_AB":
                region(s.A, "stop.A")
                region(s.B, "stop.B")
            elif s.kind == "b_surface":
                region(s.reaction, "stop.reaction")
                if s.r_observable not in items:
                    raise ValueError(f"stop.r_observable: unknown observable {s.r_observable!r}")
        if self.resampler.kind == "we":
            w = self.resampler
            w.bin_edges = [threshold(w.progress, e, f"resampler.bin_edges[{i}]")
                           for i, e in enumerate(w.bin_edges)]
            if any(b <= a for a, b in zip(w.bin_edges, w.bin_edges[1:])):
                raise ValueError("resampler.bin_edges must increase")

    def _resolve_paths(self, base: str) -> None:
        def res(p: str | None) -> str | None:
            return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

        if self.system.kind == "openmm":
            self.system.system_xml = res(self.system.system_xml)
            self.system.topology_pdb = res(self.system.topology_pdb)
        if self.ic.kind == "frames":
            self.ic.path = res(self.ic.path)
        if self.ic.kind == "encounter":
            self.ic.frames_A, self.ic.frames_B = res(self.ic.frames_A), res(self.ic.frames_B)
        self.store_path = res(self.store_path)
        self.budget.output_frames = res(self.budget.output_frames)

    def normalized(self) -> dict:
        """Plain JSON form in internal units (what `config_hash` hashes)."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of `normalized()`: independent of key
        order and of the unit spelling of equal quantities."""
        return _config_hash(self.normalized())


def load_config(path: str | os.PathLike) -> RunConfig:
    """Read a YAML config; relative paths in it are relative to its directory.
    Raises pydantic.ValidationError (field paths in the message) on bad input."""
    path = os.fspath(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return RunConfig.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))
