"""`run_shot`: the VENUS "one trajectory" lifecycle (Task 7).

Sample an initial condition through the IC gate (`sampler.sample`) -> build a
propagator (`backend.build`) -> advance it in `obs.dt_obs`-sized chunks,
computing observables and feeding them to `stop` after each chunk -> once
`stop` returns a decision, assemble one `ShotRecord` and, if `store` was
given, append it.

Controller ruling R3 (observation cadence)
-------------------------------------------
Observables are computed by `run_shot` itself, once per chunk of
``round(dt_obs / propagator.dt)`` integrator steps, by calling
``propagator.get_state()`` -- there are no backend reporters. `dt_obs` must
be an integer multiple of the propagator's own `dt` to within ``1e-9``
relative; anything else is a `ValueError`.

Controller ruling R34 (effective dt comes from the propagator, not the
backend)
---------------------------------------------------------------------------
`dt_obs` is validated against `propagator.dt` -- the `Propagator` returned
by `backend.build(...)` -- rather than any attribute on `backend` itself.
This matters because a backend's *effective* step size can depend on the
`PhysicsConfig` passed to `build()` (e.g. an OpenMM backend reads its `dt`
from `physics_cfg.dt_ps`, not from anything fixed on the backend object), so
the only place the true, already-resolved step size is guaranteed to live is
the propagator `build()` actually returned. Consequently the validation
happens *after* `build()`, not before -- but still strictly before any
`Store` write, so a misconfigured `dt_obs` still has no persistent side
effect. (It does mean `sampler.sample(key)` and `backend.build(...)` have
already run by the time this can raise; that is unavoidable once the
effective `dt` can depend on `build()`'s inputs.)

The stop rule's `update(obs, t)` is called with `obs = {name:
float(fn(state)) for name, fn in obs.fns.items()}` (no `"t"` key -- `t` is
passed as `update`'s second, separate argument) and the engine clock
`t = step_index * propagator.dt` (see "Shot clock" below). The very first observation is
taken from `propagator.get_state()` *after* `build()`, not from the raw IC
(`istate.state`) -- a real propagator may adjust the state it was built from
(e.g. a constraint projection or energy minimization step) before its first
`get_state()` call, so the state `run_shot` actually starts observing from
is whatever the propagator reports, never the pre-build IC. This includes
the possibility that a rule fires immediately, with zero integration steps
run.

There is no separate max-steps guard: a shot runs until `stop.update`
returns a decision, however long that takes. A stop rule that can never
fire (e.g. `AbsorbingAB` built without a reachable `t_max`) is the caller's
configuration error, not something `run_shot` defends against -- see
`AbsorbingAB`/`BSurface`, which already require `t_max` in their
constructors.

`stop.reset()` is called once, right after `build()` (before the first
`update()`). This is what makes it safe to bind one long-lived `StopRule`
instance into a `shot_fn` (e.g. via `functools.partial(run_shot, ...,
stop=one_shared_rule, ...)`) and reuse it across many shots in a batch --
without this, the second shot's first `update()` call would raise
`RuntimeError` (a `StopRule` refuses `update()` after it has already
produced a decision, until `reset()`).

RNG substreams (controller ruling)
-----------------------------------
`backend.build(istate.state, physics_cfg, rng_key=key)` is called with the
`ShotKey` *itself* as the propagator's RNG key (the analytic backend derives
its own "propagate" substream from it, `derive_rng(key, "propagate")`). IC
sampling, inside `sampler.sample(key)`, independently derives its own
substreams from the same key (``"ic/frame"`` for the frame choice when
``key.frame_id == -1``, ``"ic/velocities/{k}"`` for velocity draw k). These
never collide: they are different, disjoint substream labels folded into
`derive_rng`, so the propagator's random draws and the IC gate's draws are
statistically independent even though both are keyed off the same
`ShotKey`.

ICRejectedError
----------------
If `sampler.sample(key)` raises `ICRejectedError` (every attempt, including
every redraw, failed the gate), `run_shot` records *nothing* for `key` and
lets the exception propagate unchanged -- it is the caller's (`run_batch`'s)
job to decide what an IC rejection means for a batch, not `run_shot`'s.

`store` is optional (controller ruling R33)
---------------------------------------------
`store: Store | None = None`. When given, `run_shot` appends the finished
record to it as its last step (this is the right thing to do when calling
`run_shot` directly, outside of any batch: `store` is then the one and only
record of truth for that shot). When `store is None`, `run_shot` computes
and returns the record without writing it anywhere.

Composing a `shot_fn` for `run_batch` (see `cytherea.exec.batch`) means
binding `run_shot` with `store=None`: `run_batch` is the *sole* writer of
whatever `store` **it** is given, appending each `shot_fn`-returned record
itself, exactly once, in the parent process, in `keys` order -- regardless
of `n_workers`. (An earlier version of this module had `shot_fn` bind a
private "scratch" store instead of `store=None`; that was found to still
leave a real correctness gap under a crash between `shot_fn` returning and
`run_batch`'s own append -- see `exec/batch.py`'s module docstring -- and
has been replaced by this simpler, gap-free contract: a `shot_fn` used with
`run_batch` should never write to *any* store itself.)

Shot clock (contract K1, fullreview A-C1)
------------------------------------------
The engine keeps its own clock: the observation after `step_index`
integration steps is at ``t = step_index * propagator.dt`` -- a
multiplication, never an accumulated sum -- starting from 0 at the
post-build observation. That `t` is what the stop rule sees, what
``observables["t"]`` stores and what `event_time` refers to; the
propagator's own ``state.t`` is never used. The initial state is always
built at ``t = 0.0``: a sampler that hands over ``InitialState.state.t != 0``
(pre-K1 samplers put the frame's trajectory time there; K1 moves it to
``meta["frame_time"]``) is rebased to 0 and the record gets a warning.
Previously a frame time of 50 made ``FixedLag(1.0)`` fire after zero steps.

Non-finite states (contract K4, fullreview A-I3)
-------------------------------------------------
After every observation the engine checks `t`, every observable value and
the state's ``x``/``v`` (and box) for NaN/inf. The first non-finite one
stops the shot with ``StopDecision("nonfinite", event_time=t)`` -- recorded
as ``stop_reason="nonfinite"`` with ``final_state_label=None`` -- instead of
letting a blown-up trajectory run to ``t_max`` and be stored as a
``"timeout"``. (Stop rules implementing K4 return the same decision for a
non-finite observable; the engine check additionally covers ``x``/``v``,
which rules never see; `offline_replay` of such a record still returns the
``"nonfinite"`` decision.) A backend that refuses to return a blown-up state
raises `NumericalInstabilityError` from ``run``/``get_state`` (OpenMM
CPU/CUDA; contract K10): the engine then appends a NaN observation at the
time the chunk was heading for, stops as ``"nonfinite"`` there and records
the message in ``warnings``. Any other exception propagates.

Physics config and provenance (contracts K8, K9; fullreview A-I1)
------------------------------------------------------------------
The backend must implement ``effective_config(cfg=None)`` and
``provenance(cfg=None)`` (the `PotentialBackend` protocol); otherwise
`run_shot` raises TypeError before anything runs. `physics_config_hash` is
``config_hash(backend.effective_config(physics_cfg))`` -- for ``None`` and
for an explicit cfg alike (contract K9, which supersedes K8's "hash the
explicit cfg"): the hash describes the dynamics that actually run, so an
explicit cfg the backend ignores (analytic) does not hide the backend's own
potential/dt/kT/gamma/mass, the OpenMM System and Topology enter through
their digests, and ``None`` and an equivalent explicit cfg hash the same.
`resolve_physics_config` is the one place this is computed (`run_shot`,
`cytherea.exec.batch` and the WE segment engine all call it).
``backend.provenance(physics_cfg)`` is recorded -- with the same cfg that
``build`` gets. If the passed cfg or ``effective_config(physics_cfg)``
has a ``purpose`` other than ``"measurement"`` the shot is rejected before
anything runs (design section 9: equilibration dynamics are not
measurements).

Protocol hash (ruling R39)
---------------------------
``record.protocol_hash`` is `protocol_hash(stop, obs, sampler)`: the
`config_hash` of `protocol_description` -- the stop rule's kind and
constructor parameters, the ObsSpec, and the IC sampler's configuration
including the frame pool's content. It is computed before sampling, so an
undescribable part (e.g. a Region with an opaque lambda predicate and no
spec -- use `spec_region`) raises ProtocolDescriptionError before anything
runs. Observable *functions* are opaque and enter only by name; the
backend enters through `physics_config_hash`. The `labeler` of `run_shot`
(when given) enters as ``"labeler"``: it must be described like a Region
predicate -- a `spec` attribute, e.g. `cytherea.observe.SpecLabeler` -- or
a `protocol_description()` method, else ProtocolDescriptionError (for
FixedLag shots the label is the measured outcome, so a changed state
definition must not be resumed silently; int2-m2). Without a labeler the
description, and so the hash, is unchanged from before.
`run_batch` refuses to resume against records with another protocol hash.
The pool content is hashed on every call (a few ms per 10 MB of
coordinates).

Record contents (contracts K2, K6)
-----------------------------------
``record.ic_meta`` is the sampler's ``InitialState.meta``, as given.
``record.observables_thinned`` is ``obs.store_stride > 1``: only an
unthinned (stride 1) record holds the series the stop rule saw, so only
it can be replayed offline. Observable values must be real scalars (Python
or numpy numbers, or 0-d arrays); they are converted to ``float`` as they
are observed -- the stop rule sees the same floats that are stored -- and a
non-scalar value raises ``TypeError`` at the first, pre-integration
observation. `ObsSpec` itself is validated before the sampler runs (an
observable may not be called ``"t"``; ``store_stride`` must be an int >= 1;
``dt_obs`` finite and positive).

Code version (fullreview A-I7)
-------------------------------
`code_version()` is ``"<package version>+src.<sha256 of the installed
cytherea sources, 12 hex>"``, plus ``".g<commit>"`` and ``"+dirty"`` when git
is available -- so it identifies the code that ran on every host (node 180
cannot run git here) instead of degrading to a bare package version.
`code_identity()` strips the informational git parts (``".g<commit>"`` and
the ``"+dirty"`` suffix); that is what resume compares (see
`cytherea.exec.batch`). It is computed once per process, when this module
is imported.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import math
import numbers
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cytherea.backends.base import (
    MDState,
    NumericalInstabilityError,
    PhysicsConfig,
    PotentialBackend,
)
from cytherea.ic import frames as _frames_mod
from cytherea.ic.sampler import EnsembleFrameSampler
from cytherea.keys import ShotKey, key_digest
from cytherea.observe.events import (
    Observables,
    ProtocolDescriptionError,
    Region,
    SpecLabeler,  # noqa: F401 -- re-exported (tests, callers)
    SpecPredicate,  # noqa: F401 -- re-exported (tests, callers)
    StopDecision,
    StopRule,
    region_description,
    spec_region,  # noqa: F401 -- re-exported
)
from cytherea.store import ShotRecord, Store, _plain, _plain_mapping, config_hash

# Relative tolerance for "dt_obs is an integer multiple of the propagator's
# dt" (controller ruling R3/R34).
_DT_OBS_REL_TOL = 1e-9


@dataclasses.dataclass
class ObsSpec:
    """What to observe, how often, and how much of it to keep.

    `fns`: name -> a function computing one scalar observable from an
    `MDState`. Each is called once per chunk (see module docstring); the
    resulting `{name: value}` dict (without `"t"`) is what gets passed to
    `stop.update(obs, t)`.

    `dt_obs`: observation cadence, in the propagator's own time units. Must
    be an integer multiple of `propagator.dt` (see `_steps_per_chunk`).

    `store_stride`: thin the recorded (not the stop-rule-observed --
    *every* observation is still fed to `stop`) series by keeping every
    `store_stride`-th observation, always including the first and the final
    one regardless of stride.
    """

    fns: dict[str, Callable[[MDState], float]]
    dt_obs: float
    store_stride: int


def _steps_per_chunk(dt_obs: float, propagator_dt: float) -> int:
    """`round(dt_obs / propagator_dt)`, requiring that ratio to be an integer
    within `_DT_OBS_REL_TOL` relative tolerance (controller ruling R3/R34).
    """
    if not (math.isfinite(propagator_dt) and propagator_dt > 0.0):
        raise ValueError(f"propagator dt must be finite and positive, got {propagator_dt!r}")
    if not (math.isfinite(dt_obs) and dt_obs > 0.0):
        raise ValueError(f"obs.dt_obs must be finite and positive, got {dt_obs!r}")

    ratio = dt_obs / propagator_dt
    n = round(ratio)
    if n < 1:
        raise ValueError(
            f"obs.dt_obs={dt_obs!r} must be at least one propagator step "
            f"(propagator dt={propagator_dt!r})"
        )
    tol = _DT_OBS_REL_TOL * max(abs(dt_obs), abs(propagator_dt))
    if abs(dt_obs - n * propagator_dt) > tol:
        raise ValueError(
            f"obs.dt_obs={dt_obs!r} is not an integer multiple of the "
            f"propagator's dt={propagator_dt!r} within {_DT_OBS_REL_TOL:.0e} "
            f"relative tolerance (closest integer multiple: n={n})"
        )
    return n


def _thin_indices(n: int, stride: int) -> list[int]:
    """Indices to keep out of `range(n)`: every `stride`-th one, plus the
    last index unconditionally (so the final observation is always kept even
    if the stride would otherwise skip past it). `n >= 1` always holds here
    (there is always at least the post-build observation).
    """
    if stride < 1:
        raise ValueError(f"obs.store_stride must be >= 1, got {stride!r}")
    idx = list(range(0, n, stride))
    if not idx or idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


_PKG_DIR = Path(__file__).resolve().parent.parent  # .../cytherea


def _source_digest() -> str | None:
    """sha256 over every ``.py`` file of the installed cytherea package
    (relative path + contents, in sorted path order). None if unreadable."""
    h = hashlib.sha256()
    try:
        for path in sorted(_PKG_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            h.update(path.relative_to(_PKG_DIR).as_posix().encode("utf-8") + b"\0")
            h.update(path.read_bytes() + b"\0")
    except OSError:
        return None
    return h.hexdigest()


# Computed at import, so it describes the code this process actually loaded
# even if the files change on disk later.
_SOURCE_DIGEST = _source_digest()


def _git_state() -> tuple[str | None, bool]:
    """(short commit, dirty) of the checkout holding the package, best
    effort: (None, False) when git is unavailable, slow or fails (e.g. on
    node 180, where the git directory is not reachable)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_PKG_DIR, capture_output=True, text=True, timeout=2.0,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return None, False
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=_PKG_DIR, capture_output=True, text=True, timeout=2.0,
        )
    except Exception:
        return None, False
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return head.stdout.strip(), dirty


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """See the module docstring ("Code version")."""
    try:
        version = importlib.metadata.version("cytherea")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    src = _SOURCE_DIGEST[:12] if _SOURCE_DIGEST else "unknown"
    out = f"{version}+src.{src}"
    commit, dirty = _git_state()
    if commit:
        out += f".g{commit}"
    if dirty:
        out += "+dirty"
    return out


_GIT_COMPONENT = re.compile(r"\.g[0-9a-f]+$")


def code_identity(version: str) -> str:
    """`version` without its informational git parts: the ``"+dirty"``
    suffix (contract K8) and, for `code_version()` strings, the
    ``".g<commit>"`` component (the source hash already identifies the
    code, and git is not available on every host)."""
    if version.endswith("+dirty"):
        version = version[: -len("+dirty")]
    if "+src." in version:
        version = _GIT_COMPONENT.sub("", version)
    return version


def _require_k8_backend(backend: PotentialBackend) -> None:
    """The backend must implement the K8 protocol: ``effective_config(cfg=None)``
    and ``provenance(cfg=None)``, both callable with one positional cfg."""
    for name in ("effective_config", "provenance"):
        fn = getattr(backend, name, None)
        if not callable(fn):
            raise TypeError(
                f"backend {type(backend).__name__} does not implement "
                f"PotentialBackend.{name}(cfg=None) (contract K8)"
            )
        try:
            inspect.signature(fn).bind(None)
        except TypeError:
            raise TypeError(
                f"backend {type(backend).__name__}.{name} must accept the physics "
                f"cfg as its one argument, {name}(cfg=None) (contract K8)"
            ) from None
        except ValueError:  # signature not introspectable (builtin): trust it
            pass


def resolve_physics_config(
    backend: PotentialBackend, physics_cfg: PhysicsConfig | None
) -> tuple[dict, str]:
    """``(effective, config_hash(effective))`` with ``effective =
    backend.effective_config(physics_cfg)``, for a shot built with
    ``backend.build(..., physics_cfg, ...)`` (contract K9: the same rule for
    ``None`` and an explicit cfg). This is the only place the physics hash
    is computed: `run_shot`, `cytherea.exec.batch` (to know the hash a
    `run_shot`-based `shot_fn` will record) and the WE segment engine all
    call it."""
    _require_k8_backend(backend)
    effective = backend.effective_config(physics_cfg)
    return effective, config_hash(effective)


def _check_measurement_purpose(*cfgs: object) -> None:
    for cfg in cfgs:
        if cfg is None:
            continue
        if isinstance(cfg, Mapping):
            purpose = cfg.get("purpose")
        else:
            purpose = getattr(cfg, "purpose", None)
        if purpose is not None and purpose != "measurement":
            raise ValueError(
                f"run_shot records measurement shots only, but the physics config in "
                f"effect has purpose={purpose!r} (design section 9: equilibration "
                "dynamics, e.g. gamma=1/ps, are not measurements)"
            )


# ---------------------------------------------------------------------------
# Shot protocol description and hash (controller ruling R39)
# ---------------------------------------------------------------------------


# ProtocolDescriptionError, SpecPredicate and spec_region live in
# cytherea.observe.events next to Region and the stop rules (whose public
# protocol_description() needs them); they are re-exported here unchanged.


def _array_desc(a: object) -> dict:
    arr = np.ascontiguousarray(np.asarray(a))
    return {
        "ndarray_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _describe(value: object, where: str) -> object:
    """Canonical plain-JSON description of `value` (see protocol_description)."""
    if value is None or isinstance(value, (bool, np.bool_, str, numbers.Number)):
        return _plain(value, where)
    if isinstance(value, np.ndarray):
        return _array_desc(value)
    desc = getattr(value, "protocol_description", None)
    if callable(desc):
        return {"class": type(value).__qualname__, "description": _describe(desc(), where)}
    if isinstance(value, Region):
        return _describe_region(value, where)
    if isinstance(value, Mapping):
        bad = [k for k in value if not isinstance(k, str)]
        if bad:
            # p1 m-8: str(k) would make {1: x} and {"1": x} hash alike
            raise ProtocolDescriptionError(
                f"{where}: mapping key {bad[0]!r} is not a str; protocol descriptions "
                "need str keys"
            )
        return {k: _describe(v, f"{where}[{k!r}]") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if callable(value):
        spec = getattr(value, "spec", None)
        if spec is None:
            raise ProtocolDescriptionError(
                f"{where}: {value!r} is an opaque callable; give it a `spec` "
                "attribute (or wrap it in SpecPredicate) so the protocol can be hashed"
            )
        return {"spec": _plain(spec, f"{where}.spec")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "class": type(value).__qualname__,
            **{f.name: _describe(getattr(value, f.name), f"{where}.{f.name}")
               for f in dataclasses.fields(value)},
        }
    # Public callable attributes are described too (through their `spec`),
    # never silently dropped (p1 m-8): an opaque one raises.
    public = {
        k: v for k, v in vars(value).items() if not k.startswith("_")
    } if hasattr(value, "__dict__") else {}
    if not public:
        raise ProtocolDescriptionError(
            f"{where}: cannot describe {type(value).__qualname__} (no public "
            "attributes); give it a protocol_description() method"
        )
    return {
        "class": type(value).__qualname__,
        **{k: _describe(v, f"{where}.{k}") for k, v in sorted(public.items())},
    }


def _describe_region(region: Region, where: str) -> dict:
    desc = region_description(region, where)  # raises for an opaque predicate
    return {"region": desc["region"], "spec": _plain(desc["spec"], f"{where}.spec")}


def _describe_stop_rule(stop: StopRule) -> dict:
    """``{"kind", "class", "params"}`` with params from the rule's public
    ``protocol_description()`` (FixedLag, AbsorbingAB and BSurface define it
    in observe/events.py; any other rule must too)."""
    desc = getattr(stop, "protocol_description", None)
    if not callable(desc):
        raise ProtocolDescriptionError(
            f"stop rule {type(stop).__qualname__} has no protocol_description() "
            "method; give it one returning its constructor parameters "
            "(ruling R39)"
        )
    return {
        "kind": getattr(stop, "kind", None),
        "class": type(stop).__qualname__,
        "params": _describe(desc(), "stop.protocol_description()"),
    }


def _describe_pool(pool: object) -> dict:
    frames = list(pool.frames)
    # int2-m3 / p1 m-1: an EnsembleFramePool freezes its frames and caches
    # this digest, so it is computed once per pool, not once per shot.
    cached = getattr(pool, "content_sha256", None)
    digest = cached() if callable(cached) else _frames_mod.frames_sha256(frames)
    out: dict = {"n_frames": len(frames), "frames_sha256": digest}
    state_of = getattr(pool, "state_of", None)
    if state_of is not None:
        label = getattr(pool, "state_label", None)
        labels = [label(f) if callable(label) else state_of(f) for f in frames]
        out["state_labels"] = _plain(labels, "pool state labels")
    return out


# its physics (incl. the OpenMM System/Topology digests, K9) is in physics_config_hash
_SAMPLER_EXCLUDED = frozenset({"backend"})


def _describe_sampler(sampler: object) -> dict:
    desc = getattr(sampler, "protocol_description", None)
    if callable(desc):
        return {"class": type(sampler).__qualname__,
                "description": _describe(desc(), "sampler.protocol_description()")}
    if not hasattr(sampler, "pool") or not hasattr(sampler, "__dict__"):
        raise ProtocolDescriptionError(
            f"sampler {type(sampler).__qualname__} has no frame pool to describe; "
            "give it a protocol_description() method"
        )
    out: dict = {"class": type(sampler).__qualname__, "pool": _describe_pool(sampler.pool)}
    for name, value in sorted(vars(sampler).items()):
        if name.startswith("_") or name == "pool" or name in _SAMPLER_EXCLUDED:
            continue
        out[name] = _describe(value, f"sampler.{name}")
    return out


def protocol_description(
    stop: StopRule, obs: ObsSpec, sampler: object, labeler: object = None
) -> dict:
    """Canonical plain-JSON description of a shot protocol (ruling R39):
    the stop rule (kind and every constructor parameter; Regions by name and
    `spec`), the ObsSpec (observable names, dt_obs, store_stride) and the
    IC sampler (class, every public attribute except the backend -- kT,
    masses (sha256), energy_window, min_pair_dist, state, max_redraws,
    constraints, remove_com_momentum, ... -- and the frame pool's content:
    frame ids, weights, topology refs, coordinates and boxes (sha256) and
    state labels) -- plus, when `labeler` is not None, ``"labeler"``: its
    `spec` (or `protocol_description()`; int2-m2). Raises
    ProtocolDescriptionError when a part cannot be described (an opaque
    labeler included); nothing is ever hashed by identity."""
    out = {
        "stop_rule": _describe_stop_rule(stop),
        "obs": {
            "observables": sorted(obs.fns),
            "dt_obs": _plain(obs.dt_obs, "obs.dt_obs"),
            "store_stride": _plain(obs.store_stride, "obs.store_stride"),
        },
        "sampler": _describe_sampler(sampler),
    }
    if labeler is not None:
        out["labeler"] = _describe_labeler(labeler)
    return out


def _describe_labeler(labeler: object) -> object:
    if not callable(labeler):
        raise ProtocolDescriptionError(f"labeler: {labeler!r} is not callable")
    if getattr(labeler, "spec", None) is None and not callable(
        getattr(labeler, "protocol_description", None)
    ):
        raise ProtocolDescriptionError(
            f"labeler: {labeler!r} is opaque; wrap it as cytherea.observe."
            "SpecLabeler(fn, spec) (or give it a `spec` attribute) so the state "
            "definition enters the protocol hash (R39, int2-m2)"
        )
    return _describe(labeler, "labeler")


def protocol_hash(
    stop: StopRule, obs: ObsSpec, sampler: object, labeler: object = None
) -> str:
    """``config_hash(protocol_description(stop, obs, sampler, labeler))``."""
    return config_hash(protocol_description(stop, obs, sampler, labeler))


def _validate_obs_spec(obs: ObsSpec) -> None:
    if "t" in obs.fns:
        raise ValueError(
            'ObsSpec.fns may not define an observable named "t": that name is '
            "the shot clock in the recorded observables"
        )
    stride = obs.store_stride
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, numbers.Integral):
        raise TypeError(f"obs.store_stride must be an int, got {stride!r}")
    if stride < 1:
        raise ValueError(f"obs.store_stride must be >= 1, got {stride!r}")
    dt_obs = obs.dt_obs
    if not isinstance(dt_obs, numbers.Real) or not (math.isfinite(dt_obs) and dt_obs > 0.0):
        raise ValueError(f"obs.dt_obs must be finite and positive, got {dt_obs!r}")


def _observable_value(name: str, value: object) -> float:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"observable {name!r} returned {type(value).__name__} {value!r}; "
            "observables must return a real scalar"
        )
    return float(value)


def _all_finite(t: float, obs_now: Observables, state: MDState) -> bool:
    if not math.isfinite(t) or not all(math.isfinite(v) for v in obs_now.values()):
        return False
    arrays = [state.x, state.v] if state.box is None else [state.x, state.v, state.box]
    return all(bool(np.all(np.isfinite(a))) for a in arrays)


def _validity_dict(validity: object) -> dict:
    """``validity.to_dict()`` (the sampler's own serialization, including
    ``rejected_attempts``) when available, else every dataclass field."""
    to_dict = getattr(validity, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if dataclasses.is_dataclass(validity) and not isinstance(validity, type):
        return dataclasses.asdict(validity)
    return {
        "ok": validity.ok,
        "reasons": validity.reasons,
        "n_redraws": validity.n_redraws,
        "checks": validity.checks,
    }


def run_shot(
    key: ShotKey,
    sampler: EnsembleFrameSampler,
    backend: PotentialBackend,
    stop: StopRule,
    obs: ObsSpec,
    physics_cfg: PhysicsConfig | None,
    store: Store | None = None,
    labeler: Callable[[Observables], str] | None = None,
) -> ShotRecord:
    """Run one trajectory shot end to end; return its `ShotRecord`, appending
    it to `store` first if `store` is not `None`.

    See the module docstring for the full contract (RNG substreams, dt_obs
    validation against `propagator.dt`, ICRejectedError propagation, and why
    `store` is optional).
    """
    # Everything that can be checked without running anything, first.
    _validate_obs_spec(obs)
    effective_cfg, physics_hash = resolve_physics_config(backend, physics_cfg)
    _check_measurement_purpose(physics_cfg, effective_cfg)
    proto_hash = protocol_hash(stop, obs, sampler, labeler)
    warnings: list[str] = []

    # ICRejectedError propagates unchanged here: nothing has been recorded
    # yet, and nothing below runs.
    istate, validity = sampler.sample(key)
    # Normalized now so that an unrepresentable value fails before dynamics.
    ic_validity = _plain_mapping(_validity_dict(validity), "ic_validity")
    ic_meta = _plain_mapping(istate.meta, "InitialState.meta")

    start = istate.state
    if start.t != 0.0:
        warnings.append(
            f"initial state t={start.t!r} was rebased to 0.0 (contract K1: the "
            "shot clock starts at 0; a frame's own time belongs in "
            "meta['frame_time'])"
        )
        start = dataclasses.replace(start, t=0.0)

    propagator = backend.build(start, physics_cfg, key)

    # Validated against the propagator's own (already-resolved) dt, and
    # still strictly before any Store write (controller ruling R34).
    dt = propagator.dt
    n_steps = _steps_per_chunk(obs.dt_obs, dt)

    stop.reset()

    times: list[float] = []
    series: dict[str, list[float]] = {name: [] for name in obs.fns}
    last_obs: Observables = {}
    decision: StopDecision | None = None
    step_index = 0

    # The first observation comes from the propagator's own post-build
    # state, not the raw IC (controller ruling R35 #4).
    state = propagator.get_state()
    while True:
        t_now = step_index * dt  # contract K1: multiplied, never accumulated
        obs_now = {name: _observable_value(name, fn(state)) for name, fn in obs.fns.items()}
        times.append(t_now)
        for name, value in obs_now.items():
            series[name].append(value)
        last_obs = obs_now

        if not _all_finite(t_now, obs_now, state):
            decision = StopDecision(
                reason="nonfinite", event_time=t_now if math.isfinite(t_now) else None
            )
            break
        decision = stop.update(obs_now, t_now)
        if decision is not None:
            break

        try:
            propagator.run(n_steps)
            state = propagator.get_state()
        except NumericalInstabilityError as exc:
            # contract K10: the backend refused to return the blown-up state
            # (OpenMM CPU/CUDA). Record the observation this chunk was heading
            # for as non-finite, exactly as a NaN state would have been seen.
            step_index += n_steps
            t_now = step_index * dt
            times.append(t_now)
            for name in obs.fns:
                series[name].append(math.nan)
            last_obs = {name: math.nan for name in obs.fns}
            warnings.append(f"NumericalInstabilityError at t={t_now!r}: {exc}")
            decision = StopDecision(reason="nonfinite", event_time=t_now)
            break
        step_index += n_steps

    kept = _thin_indices(len(times), obs.store_stride)
    observables: dict[str, list[float]] = {"t": [times[i] for i in kept]}
    for name in obs.fns:
        observables[name] = [series[name][i] for i in kept]

    nonfinite = decision.reason == "nonfinite"
    record = ShotRecord(
        key_digest=key_digest(key),
        key=dataclasses.asdict(key),
        kind="shot",
        frame_id=istate.frame_id,
        origin_label=getattr(istate, "origin_label", None),
        ic_validity=ic_validity,
        stop_rule_kind=stop.kind,
        stop_reason=decision.reason,
        event_time=decision.event_time,
        physics_config_hash=physics_hash,
        protocol_hash=proto_hash,
        backend_provenance=backend.provenance(physics_cfg),
        code_version=code_version(),
        observables=observables,
        final_state_label=(
            labeler(last_obs) if labeler is not None and not nonfinite else None
        ),
        ic_meta=ic_meta,
        observables_thinned=obs.store_stride > 1,
        warnings=warnings,
    )
    if store is not None:
        store.append(record)
    return record
