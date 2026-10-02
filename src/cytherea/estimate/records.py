"""From stored records to estimator inputs (contract K11, spec S1).

`shot_weights` is the one place frame weights become shot weights:

* enumerated design (``key["frame_id"] >= 0``: every frame is shot on
  purpose, typically K_k times): shot weight = ``record.weight *
  ic_meta["frame_weight"] / K_k``, with K_k the number of records of frame k
  in the given set, so every frame contributes its own weight in total
  (design 4.1, ``w_k / K_k``);
* weighted draw (``key["frame_id"] == -1``: the sampler already drew frames
  with probability proportional to their weight): shot weight =
  ``record.weight`` (applying the frame weight again would count it twice).

Mixing the two designs in one set raises ``ValueError``. A set in which no
record carries ``ic_meta["frame_weight"]`` (hand-built records, or records
written before contract K2) keeps ``record.weight``; a set in which only
some enumerated records carry it raises. The ``record.weight`` factor is
the WE / importance weight (1.0 for plain shots).

`records_to_groups` builds the ``{state: {frame: [values]}}`` input of
`cytherea.estimate.hierarchical_bootstrap` (S1: states fixed, frames
resampled within their state); `records_to_transitions` builds the keyword
arguments of `cytherea.estimate.estimate_T`, always including
``stop_reasons`` so that a nonfinite shot invalidates the estimate (K4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from cytherea.store import ShotRecord


def _key_frame_id(r: ShotRecord) -> int | None:
    fid = r.key.get("frame_id") if isinstance(r.key, Mapping) else None
    return None if fid is None else int(fid)


def _sort_key(value: Hashable):
    return (type(value).__name__, value)


def shot_weights(records: Sequence[ShotRecord]) -> np.ndarray:
    """Per-record weights for pooled estimators (module docstring, K11)."""
    records = list(records)
    if not records:
        return np.zeros(0)
    w = np.array([float(r.weight) for r in records])
    has_fw = ["frame_weight" in r.ic_meta for r in records]
    if not any(has_fw):
        return w
    kf = [_key_frame_id(r) for r in records]
    enumerated = [f is not None and f >= 0 for f in kf]
    if any(enumerated) and not all(enumerated):
        raise ValueError(
            "records mix the enumerated design (key frame_id >= 0) and weighted frame draws "
            "(key frame_id == -1); their frame weights combine differently (K11) -- "
            "estimate them separately"
        )
    if not all(enumerated):
        return w  # frames were drawn proportionally to their weight
    if not all(has_fw):
        missing = next(r.key_digest for r, h in zip(records, has_fw) if not h)
        raise ValueError(
            f"record {missing[:12]} has no ic_meta['frame_weight'] while others do (K2/K11)"
        )
    per_frame = Counter(r.frame_id for r in records)
    fw = np.array([float(r.ic_meta["frame_weight"]) for r in records])
    k = np.array([per_frame[r.frame_id] for r in records], dtype=float)
    out = w * fw / k
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError("frame weights must be finite and >= 0")
    return out


def _reject_for_groups(records: Sequence[ShotRecord]) -> None:
    for r in records:
        if r.kind == "segment":
            raise ValueError("WE segments are correlated by ancestry; bootstrap over WE runs instead")
        if r.stop_reason == "nonfinite":
            raise ValueError(
                f"record {r.key_digest[:12]} stopped 'nonfinite' (contract K4): the estimate is "
                "invalid; fix the run instead of dropping it"
            )


def records_to_groups(
    records: Iterable[ShotRecord],
    value: Callable[[ShotRecord], float],
    state_of: Callable[[ShotRecord], Hashable] | None = None,
) -> dict[Hashable, dict[Hashable, list[float]]]:
    """``{state: {frame_id: [value(record), ...]}}`` for
    `hierarchical_bootstrap` (spec S1), in canonical order: states and
    frames sorted, shots of a frame in ``key_digest`` order, so the result
    depends on the record set only. The state is ``ic_meta["state"]``
    unless `state_of` is given. Segments and nonfinite records raise."""
    records = sorted(records, key=lambda r: r.key_digest)
    _reject_for_groups(records)
    get_state = state_of or (lambda r: r.ic_meta.get("state"))
    raw: dict[Hashable, dict[Hashable, list[float]]] = {}
    for r in records:
        raw.setdefault(get_state(r), {}).setdefault(r.frame_id, []).append(float(value(r)))
    return {
        s: {f: raw[s][f] for f in sorted(raw[s], key=_sort_key)}
        for s in sorted(raw, key=_sort_key)
    }


def records_to_transitions(
    records: Iterable[ShotRecord],
    state_index: Mapping[Hashable, int],
    weights: Sequence[float] | None = None,
) -> dict:
    """Keyword arguments ``start_states, end_states, weights, frame_ids,
    stop_reasons`` of `estimate_T` from fixed-lag shot records: start state
    ``state_index[ic_meta["state"]]``, end state
    ``state_index[final_state_label]``, weights from `shot_weights` (or
    `weights`, one per record in the order given -- they are reordered with
    the records), frame ids from ``record.frame_id``. Output arrays are in
    key-digest order. A nonfinite record keeps its slot with end state =
    start state as a placeholder; ``estimate_T`` drops it, counts it and sets
    ``valid=False`` (K4)."""
    records = list(records)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(records),):
            raise ValueError("weights must have one entry per record")
    order = sorted(range(len(records)), key=lambda i: records[i].key_digest)
    records = [records[i] for i in order]
    if weights is not None:
        weights = weights[order]
    for r in records:
        if r.kind == "segment":
            raise ValueError("WE segments are not fixed-lag shots")
        if r.stop_reason not in ("fixed_lag", "nonfinite"):
            raise ValueError(f"record {r.key_digest[:12]}: stop_reason {r.stop_reason!r} is not fixed_lag")
    start = np.array([state_index[r.ic_meta.get("state")] for r in records], dtype=np.int64)
    end = np.array(
        [s if r.stop_reason == "nonfinite" else state_index[r.final_state_label]
         for s, r in zip(start, records)],
        dtype=np.int64,
    )
    w = shot_weights(records) if weights is None else weights
    return {
        "start_states": start,
        "end_states": end,
        "weights": w,
        "frame_ids": np.array([r.frame_id for r in records], dtype=np.int64),
        "stop_reasons": [r.stop_reason for r in records],
    }
