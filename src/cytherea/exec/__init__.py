"""Exec: the resumable, order-independent batch shot executor.

See `batch.py` (`run_batch`, `ShotFailure`): run many shots (serially or
across a process pool), skip keys already recorded in the `Store`, and
report IC-gate rejections as failures without stopping the batch.
"""
