"""Absorbing milestone networks (design 4.3 / 4.5, Task 11)."""

from cytherea.network.absorb import (
    MarkovReport,
    StageNetwork,
    build_transitions,
    markov_test,
    solve_absorption,
)

__all__ = ["MarkovReport", "StageNetwork", "build_transitions", "markov_test", "solve_absorption"]
