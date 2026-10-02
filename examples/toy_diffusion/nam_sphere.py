"""3D free diffusion with a NAM b-surface stop rule (A0 acceptance 9.1).

Setup (reduced units, declared explicitly in `CONFIG["units"]`):

- one point particle, `FreeParticle(dim=3)`, overdamped dynamics
  (`AnalyticBackend(integrator="overdamped")`), D = kT / (m gamma) = 1;
- reaction region: |r| <= a (persistence 0: first observation inside);
- start: on the sphere |r| = b = 2a;
- escape: |r| >= q = 8a (`BSurface`), timeout at `t_max`.

Exact answers (Smoluchowski): the probability of reaching |r| = a before
|r| = q from |r| = b is

    beta_exact(q) = (1/b - 1/q) / (1/a - 1/q),

and the NAM-corrected infinite-q value is beta_inf = a / b (design 4.3,
`cytherea.estimate.association.nam_beta_inf`).

Starting directions. The engine draws initial conditions from an
`EnsembleFramePool` through `EnsembleFrameSampler` (with its IC gate), so
the b-sphere is represented as a pool of `n_directions` frames placed on a
Fibonacci (golden-spiral) lattice of |r| = b -- a deterministic, very nearly
uniform point set -- all with equal weight; each shot picks one uniformly at
random through the sampler's keyed RNG (keys with ``frame_id = -1``,
contract K3; the frames have equal weight, so `shot_weights` gives every
shot weight 1, and `estimate_kon` is called with
``allow_clustered_frames=True`` -- several shots share a direction, which
is harmless because the outcome does not depend on it). (For free diffusion the outcome is
independent of direction by symmetry, so the direction set only needs to be
unbiased, not random; the lattice avoids adding an extra RNG stream.)
Velocities drawn by the sampler are ignored by the overdamped integrator.

Observation: `dt_obs = dt` (every step). Absorbing-boundary detection on a
discretely observed Brownian path misses excursions between observations,
which effectively moves each absorbing surface outwards (away from the
start) by ~0.5826 * sqrt(2 D dt_obs) (Broadie-Glasserman-Kou continuity
correction); `beta_discrete_prediction` gives the resulting expected beta.
This is an O(sqrt(dt)) bias, which is why the acceptance test also runs at
dt/2. Every observation is stored (`store_stride = 1`) so that the online
stop decisions can be replayed offline (acceptance 9.4).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

from cytherea.backends.analytic import AnalyticBackend, FreeParticle
from cytherea.backends.base import MDState
from cytherea.engine.shot import ObsSpec, run_shot
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import EnsembleFrameSampler
from cytherea.observe.events import BSurface, Region

CONFIG: dict = {
    "units": "reduced",
    "a": 1.0,  # reaction-sphere radius
    "b": 2.0,  # start (b-surface) radius, = 2a
    "q": 8.0,  # escape radius, = 8a
    "kT": 1.0,
    "gamma": 1.0,  # a rate; D = kT/(mass*gamma) = 1
    "mass": 1.0,
    "dt": 5.0e-4,  # sqrt(2 D dt) = 0.032 a
    "tau_persist": 0.0,
    "t_max": 200.0,  # >> mean absorption time 5.5 a^2/D
    "n_directions": 1024,
    "n_shots": 4000,
}

# -zeta(1/2)/sqrt(2 pi): discrete-monitoring boundary shift, in units of the
# per-observation displacement std (Broadie, Glasserman & Kou 1997).
BGK_SHIFT = 0.5825971579390106


def radius(state: MDState) -> float:
    """Observable "r": distance of the particle from the origin."""
    return float(np.linalg.norm(state.x))


@dataclasses.dataclass(frozen=True)
class RadiusAtMost:
    """Picklable Region predicate: obs["r"] <= cutoff."""

    cutoff: float

    def __call__(self, obs: dict) -> bool:
        return obs["r"] <= self.cutoff

    @property
    def spec(self) -> str:  # enters the protocol hash (ruling R39)
        return f"r <= {self.cutoff!r}"


def fibonacci_sphere(n: int, radius_: float) -> np.ndarray:
    """n nearly uniform points on the sphere of the given radius, shape (n, 3)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(1.0 - z * z)
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return radius_ * np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def diffusion_coefficient(cfg: dict = CONFIG) -> float:
    return cfg["kT"] / (cfg["mass"] * cfg["gamma"])


def beta_exact(a: float, b: float, q: float) -> float:
    """P(reach |r| = a before |r| = q | start at |r| = b), 3D free diffusion."""
    return (1.0 / b - 1.0 / q) / (1.0 / a - 1.0 / q)


def beta_discrete_prediction(cfg: dict = CONFIG) -> float:
    """beta_exact with both absorbing surfaces moved outwards by the discrete-
    monitoring shift BGK_SHIFT * sqrt(2 D dt) (module docstring)."""
    shift = BGK_SHIFT * math.sqrt(2.0 * diffusion_coefficient(cfg) * cfg["dt"])
    return beta_exact(cfg["a"] - shift, cfg["b"], cfg["q"] + shift)


def make_backend(cfg: dict = CONFIG) -> AnalyticBackend:
    return AnalyticBackend(
        FreeParticle(dim=3),
        integrator="overdamped",
        dt=cfg["dt"],
        kT=cfg["kT"],
        gamma=cfg["gamma"],
        mass=cfg["mass"],
    )


def make_sampler(backend: AnalyticBackend, cfg: dict = CONFIG) -> EnsembleFrameSampler:
    frames = [
        EnsembleFrame(
            coordinates=p,
            box=None,
            topology_ref="point_particle_3d",
            temperature=cfg["kT"],
            weight=1.0,
            source_id="b_sphere_fibonacci",
            frame_id=k,
            time=0.0,
        )
        for k, p in enumerate(fibonacci_sphere(cfg["n_directions"], cfg["b"]))
    ]
    return EnsembleFrameSampler(
        pool=EnsembleFramePool(frames),
        masses=np.array([cfg["mass"]]),
        kT=cfg["kT"],
        backend=backend,
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=False,  # single particle, non-(n_atoms, 3) coordinates
    )


def make_stop(cfg: dict = CONFIG) -> BSurface:
    """A fresh stop rule (also used by the offline replay)."""
    return BSurface(
        reaction=Region("reaction", RadiusAtMost(cfg["a"])),
        r_name="r",
        q=cfg["q"],
        tau_persist=cfg["tau_persist"],
        t_max=cfg["t_max"],
    )


def make_obs(cfg: dict = CONFIG) -> ObsSpec:
    return ObsSpec(fns={"r": radius}, dt_obs=cfg["dt"], store_stride=1)


def make_shot_fn(cfg: dict = CONFIG):
    """Picklable `shot_fn` for `run_batch` (run_shot bound with store=None)."""
    backend = make_backend(cfg)
    return functools.partial(
        run_shot,
        sampler=make_sampler(backend, cfg),
        backend=backend,
        stop=make_stop(cfg),
        obs=make_obs(cfg),
        physics_cfg=None,
        store=None,
    )
