"""Quantities in configuration files (Task 13, Review Focus 4).

Internal units are OpenMM's: nm, ps, kJ/mol, K, rad. A config declares
``units: openmm`` or ``units: reduced``:

* ``openmm``: every dimensioned value is a string ``"<number> <unit>"``
  (``"20 angstrom"``, ``"2 fs"``, ``"0.1 /ps"``, ``"60 deg"``). A bare number is
  an error -- it is never silently read as nm.
* ``reduced`` (analytic toys): dimensioned values are plain numbers in the
  toy's own units, and a string with a unit is an error.

`parse_quantity(value, dimension, units)` returns the value in internal
units; the unit table below is the whole vocabulary.
"""

from __future__ import annotations

import math
import re

#: dimension -> {unit spelling: factor to the internal unit}
UNITS: dict[str, dict[str, float]] = {
    "length": {"nm": 1.0, "nanometer": 1.0, "nanometers": 1.0, "angstrom": 0.1, "angstroms": 0.1,
               "Å": 0.1, "A": 0.1, "pm": 1e-3},
    "time": {"ps": 1.0, "picosecond": 1.0, "picoseconds": 1.0, "fs": 1e-3, "femtosecond": 1e-3,
             "femtoseconds": 1e-3, "ns": 1e3, "nanosecond": 1e3, "nanoseconds": 1e3,
             "us": 1e6, "µs": 1e6, "microsecond": 1e6, "microseconds": 1e6},
    "inverse_time": {"/ps": 1.0, "1/ps": 1.0, "ps^-1": 1.0, "ps-1": 1.0, "/fs": 1e3, "1/fs": 1e3,
                     "fs^-1": 1e3, "/ns": 1e-3, "1/ns": 1e-3, "ns^-1": 1e-3},
    "temperature": {"K": 1.0, "kelvin": 1.0},
    "energy": {"kJ/mol": 1.0, "kj/mol": 1.0, "kcal/mol": 4.184},
    "angle": {"rad": 1.0, "radian": 1.0, "radians": 1.0, "deg": math.pi / 180.0, "degree": math.pi / 180.0,
              "degrees": math.pi / 180.0, "°": math.pi / 180.0},
}

UNIT_SYSTEMS = ("openmm", "reduced")
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_QUANTITY = re.compile(rf"^\s*({_NUMBER})\s*(\S.*?)\s*$")


def parse_quantity(value: object, dimension: str, units: str) -> float:
    """``value`` in internal units (module docstring); ValueError otherwise."""
    if dimension not in UNITS:
        raise ValueError(f"unknown dimension {dimension!r}")
    if units not in UNIT_SYSTEMS:
        raise ValueError(f"units must be one of {UNIT_SYSTEMS}, got {units!r}")
    if isinstance(value, bool):
        raise ValueError(f"expected a {dimension}, got {value!r}")
    if units == "reduced":
        if isinstance(value, (int, float)):
            out = float(value)
        else:
            raise ValueError(
                f"units: reduced takes plain numbers for a {dimension}, got {value!r} "
                "(a unit here would mix unit systems)"
            )
    else:
        if isinstance(value, (int, float)):
            raise ValueError(
                f"a {dimension} needs an explicit unit under units: openmm, e.g. "
                f"'{value} {next(iter(UNITS[dimension]))}'; got the bare number {value!r}"
            )
        if not isinstance(value, str):
            raise ValueError(f"expected '<number> <unit>' for a {dimension}, got {value!r}")
        m = _QUANTITY.match(value)
        if m is None:
            raise ValueError(f"expected '<number> <unit>' for a {dimension}, got {value!r}")
        number, unit = m.groups()
        table = UNITS[dimension]
        if unit not in table:
            raise ValueError(
                f"unit {unit!r} is not a {dimension} unit; known: {', '.join(table)}"
            )
        out = float(number) * table[unit]
    if not math.isfinite(out):
        raise ValueError(f"{dimension} must be finite, got {value!r}")
    return out
