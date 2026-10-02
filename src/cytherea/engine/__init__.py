"""Engine: the one-trajectory shooting lifecycle.

See `shot.py` (`ObsSpec`, `run_shot`): sample an initial condition through
the IC gate, build a propagator, advance it in `obs.dt_obs` chunks with
online stop-rule checks, and append exactly one `ShotRecord` to a `Store`.
"""
