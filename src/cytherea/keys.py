"""Keyed, process-independent RNG derivation.

Every random draw anywhere in cytherea must be traceable to a small,
explicit "key" describing *what* is being sampled (which shot, which
segment of which walker, which iteration, ...). Given the same key, two
calls to :func:`derive_rng` -- in the same process, a different process,
or a different order relative to other keys -- must produce bit-identical
draws. This is what makes initial conditions reproducible (design doc
section 7) and is why nothing in this module (or anything derived from
it) may consult global RNG state: no ``np.random.*`` convenience API, no
``random`` module, no unseeded OpenMM integrator.

The recipe is: canonically encode the key -> sha256 digest (stable across
interpreters and processes, unlike Python's salted ``hash()``) -> fold in
an optional ``substream`` label -> use the result as entropy for a
``numpy.random.SeedSequence`` feeding a ``PCG64`` bit generator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import operator

import numpy as np

# ASCII unit separator: extremely unlikely to appear inside a legitimate
# field value (run ids, stage names, ...), which keeps the canonical
# encoding free of field-boundary ambiguity (e.g. "a" + "b" vs "ab").
_FIELD_SEP = "\x1f"


def _normalize_fields(key: object) -> None:
    """Coerce every field of a key dataclass to its exact declared type.

    Fullreview A-I5: ``ShotKey(np.int64(1), ...)`` used to compare *equal* to
    ``ShotKey(1, ...)`` (and hash equal) while encoding to a different
    digest -- ``repr(np.int64(1))`` is ``"np.int64(1)"`` under numpy 2 -- so
    dict/set dedup and the Store/RNG disagreed about whether they were the
    same key, and the digest depended on the numpy version. Now:

    - ``int`` fields accept anything implementing ``__index__`` (Python
      ints, numpy integer scalars) and store a plain ``int``; ``bool`` and
      ``numpy.bool_`` are rejected (``True`` is not a seed), as are floats
      and strings.
    - ``str`` fields accept ``str`` (including subclasses such as
      ``numpy.str_``) and store a plain ``str``; anything else is rejected.

    So after construction every field is an exact ``int``/``str`` and the
    canonical ``repr``-based encoding below is type-stable.
    """
    for field in dataclasses.fields(key):
        value = getattr(key, field.name)
        if field.type in ("int", int):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{type(key).__name__}.{field.name} must be an integer, "
                    f"got bool {value!r}"
                )
            try:
                coerced: object = operator.index(value)
            except TypeError:
                raise TypeError(
                    f"{type(key).__name__}.{field.name} must be an integer, "
                    f"got {type(value).__name__} {value!r}"
                ) from None
        elif field.type in ("str", str):
            if not isinstance(value, str):
                raise TypeError(
                    f"{type(key).__name__}.{field.name} must be a str, "
                    f"got {type(value).__name__} {value!r}"
                )
            coerced = str.__str__(value)  # exact str, even for a str subclass
        else:  # pragma: no cover - every key field is int or str
            raise TypeError(f"unsupported key field type {field.type!r}")
        object.__setattr__(key, field.name, coerced)


@dataclasses.dataclass(frozen=True)
class ShotKey:
    global_seed: int
    frame_id: int
    shot_id: int
    stage: str

    def __post_init__(self) -> None:
        _normalize_fields(self)


@dataclasses.dataclass(frozen=True)
class SegmentKey:
    """Key of one WE segment (contract K7).

    ``global_seed`` is part of the key so that two WE runs that share a
    ``run_id`` but not a seed never share propagation noise (fullreview
    A-I9): replicas must be independent for a bootstrap over runs to be
    valid.
    """

    global_seed: int
    run_id: str
    iteration: int
    walker_id: int

    def __post_init__(self) -> None:
        _normalize_fields(self)


@dataclasses.dataclass(frozen=True)
class IterKey:
    global_seed: int
    run_id: str
    iteration: int

    def __post_init__(self) -> None:
        _normalize_fields(self)


Key = ShotKey | SegmentKey | IterKey


def _canonical_bytes(key: Key) -> bytes:
    """Encode ``key`` into a stable byte string.

    Stability requirements (see module docstring / task-1 brief):
    - Must not depend on Python's salted ``hash()``.
    - Must be identical across processes and Python invocations.
    - Must include the key's class name, so that two different key
      types can never collide even if some future key type happened to
      have fields with equal names and values.

    Dataclass field order is fixed by the class definition, so iterating
    ``dataclasses.fields(key)`` is itself stable. Each field is encoded
    as ``name=repr(value)``, which also disambiguates values that differ
    only in type (e.g. the int ``1`` vs the string ``"1"``). Fields are
    exact ``int``/``str`` by construction (``_normalize_fields``), so
    ``repr`` never sees a numpy scalar.
    """
    parts = [type(key).__name__]
    for field in dataclasses.fields(key):
        value = getattr(key, field.name)
        parts.append(f"{field.name}={value!r}")
    canonical = _FIELD_SEP.join(parts)
    return canonical.encode("utf-8")


def key_digest(key: Key) -> str:
    """Return the canonical sha256 hex digest identifying ``key``."""
    return hashlib.sha256(_canonical_bytes(key)).hexdigest()


def _entropy_from_hex(digest_hex: str) -> list[int]:
    """Split a hex digest into a list of 32-bit integers for SeedSequence."""
    return [
        int(digest_hex[i : i + 8], 16) for i in range(0, len(digest_hex), 8)
    ]


def derive_rng(key: Key, substream: str = "") -> np.random.Generator:
    """Derive a PCG64 Generator deterministically from ``key`` (+ ``substream``).

    Never touches numpy's or Python's global RNG state. Identical
    ``(key, substream)`` always yields identical draws, regardless of
    process, call order, or what else has been drawn from other keys.
    """
    digest = key_digest(key)
    combined = f"{digest}{_FIELD_SEP}{substream}".encode("utf-8")
    combined_digest = hashlib.sha256(combined).hexdigest()
    seed_sequence = np.random.SeedSequence(_entropy_from_hex(combined_digest))
    return np.random.Generator(np.random.PCG64(seed_sequence))
