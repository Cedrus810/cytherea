"""Tests for cytherea.keys: keyed RNG derivation (Task 1).

Every later task derives all randomness from `derive_rng`, so these tests
pin down the exact reproducibility contract from the brief's test table
(task-1-brief.md, section "必测用例").
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from cytherea.keys import IterKey, SegmentKey, ShotKey, derive_rng, key_digest

SEGMENT_GOLDEN = "fcee20fadd7c96486d1ba42975656d201b02e058349fdd3fbac41d642096088c"  # frozen 2026-10-01, see test below


# --- 1.1: same key called twice gives identical draws -----------------

def test_same_key_twice_gives_identical_draws():
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    a = derive_rng(key).random(8)
    b = derive_rng(key).random(8)
    assert np.array_equal(a, b)


# --- 1.2: changing only shot_id, or only stage, changes the draws -----

def test_different_shot_id_gives_different_draws():
    k1 = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    k2 = ShotKey(global_seed=1, frame_id=2, shot_id=4, stage="ic")
    a = derive_rng(k1).random(8)
    b = derive_rng(k2).random(8)
    assert np.all(a != b)


def test_different_stage_gives_different_draws():
    k1 = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    k2 = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="prod")
    a = derive_rng(k1).random(8)
    b = derive_rng(k2).random(8)
    assert np.all(a != b)


# --- 1.3: order of generation across keys does not matter -------------

def test_generation_order_does_not_matter():
    key_a = ShotKey(global_seed=1, frame_id=1, shot_id=1, stage="ic")
    key_b = SegmentKey(global_seed=1, run_id="r0", iteration=1, walker_id=2)
    key_c = IterKey(global_seed=7, run_id="r0", iteration=1)

    # generate A, B, C in that order
    draw_a_1 = derive_rng(key_a).random(8)
    draw_b_1 = derive_rng(key_b).random(8)
    draw_c_1 = derive_rng(key_c).random(8)

    # generate C, A, B in a different order
    draw_c_2 = derive_rng(key_c).random(8)
    draw_a_2 = derive_rng(key_a).random(8)
    draw_b_2 = derive_rng(key_b).random(8)

    assert np.array_equal(draw_a_1, draw_a_2)
    assert np.array_equal(draw_b_1, draw_b_2)
    assert np.array_equal(draw_c_1, draw_c_2)


# --- 1.4: multiprocessing subprocess reproduces main-process result ---

def _draw_in_subprocess(queue: "mp.Queue") -> None:
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    queue.put(derive_rng(key).random(8))


def test_multiprocessing_subprocess_matches_main_process():
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    main_draw = derive_rng(key).random(8)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_draw_in_subprocess, args=(queue,))
    proc.start()
    subprocess_draw = queue.get(timeout=60)
    proc.join(timeout=60)

    assert proc.exitcode == 0
    assert np.array_equal(main_draw, subprocess_draw)


# --- 1.5: golden digest, frozen once and hard-coded --------------------

def test_shot_key_digest_matches_frozen_golden_value():
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    # Frozen 2026-09-30 after first implementation of key_digest's canonical
    # encoding. If this assertion ever fails, the canonical encoding changed
    # and every downstream derived stream would silently shift -- that is
    # exactly the regression this test exists to catch.
    assert key_digest(key) == (
        "20f7512775a7c728974cc4b92a5d76e0"
        "b082a79de193db9a1b4c627ebc6539bb"
    )


# --- 1.6: different substreams give different draws --------------------

def test_different_substreams_give_different_draws():
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    a = derive_rng(key, substream="a").random(8)
    b = derive_rng(key, substream="b").random(8)
    assert np.all(a != b)


# --- extra: different key types with equal-looking fields never collide

def test_different_key_types_never_collide_in_digest():
    # SegmentKey and IterKey have disjoint field sets, so this mostly
    # exercises that key_digest folds in the class name at all times.
    a = SegmentKey(global_seed=1, run_id="x", iteration=1, walker_id=2)
    b = IterKey(global_seed=1, run_id="x", iteration=1)
    assert key_digest(a) != key_digest(b)


# --- fix wave P1 (fullreview A-I5, A-I9 / contract K7, A-m11) ------------


def test_segment_key_fields_are_global_seed_run_id_iteration_walker_id():
    # Contract K7: SegmentKey(global_seed, run_id, iteration, walker_id).
    import dataclasses

    assert [f.name for f in dataclasses.fields(SegmentKey)] == [
        "global_seed", "run_id", "iteration", "walker_id",
    ]


def test_segment_keys_differing_only_in_global_seed_give_different_draws():
    # A-I9: two WE replicas with the same run_id but different global_seed
    # must not share segment noise.
    a = SegmentKey(global_seed=1, run_id="we", iteration=0, walker_id=0)
    b = SegmentKey(global_seed=2, run_id="we", iteration=0, walker_id=0)
    assert key_digest(a) != key_digest(b)
    assert np.all(derive_rng(a).random(8) != derive_rng(b).random(8))


def test_numpy_integer_fields_normalize_to_the_python_int_key():
    # A-I5: ShotKey(np.int64(1), ...) compared equal to ShotKey(1, ...) but
    # had a different digest and RNG stream.
    k_py = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    k_np = ShotKey(global_seed=np.int64(1), frame_id=np.int32(2), shot_id=np.uint8(3), stage="ic")
    assert k_np == k_py
    assert type(k_np.global_seed) is int and type(k_np.shot_id) is int
    assert key_digest(k_np) == key_digest(k_py)
    assert np.array_equal(derive_rng(k_np).random(4), derive_rng(k_py).random(4))
    s_np = SegmentKey(global_seed=np.int64(5), run_id=np.str_("r"), iteration=np.int64(1), walker_id=np.int64(2))
    assert type(s_np.run_id) is str
    assert key_digest(s_np) == key_digest(SegmentKey(5, "r", 1, 2))


@pytest.mark.parametrize(
    "fields",
    [
        dict(global_seed=True, frame_id=2, shot_id=3, stage="ic"),     # bool is not an int here
        dict(global_seed="1", frame_id=2, shot_id=3, stage="ic"),      # str for an int field
        dict(global_seed=1.0, frame_id=2, shot_id=3, stage="ic"),      # float for an int field
        dict(global_seed=1, frame_id=2, shot_id=3, stage=7),           # int for a str field
        dict(global_seed=1, frame_id=np.bool_(True), shot_id=3, stage="ic"),
    ],
)
def test_key_fields_of_the_wrong_type_are_rejected(fields):
    with pytest.raises(TypeError):
        ShotKey(**fields)


def test_segment_and_iter_key_digests_match_frozen_golden_values():
    # Frozen 2026-10-01 (fix wave P1). SegmentKey gained global_seed (K7),
    # so its digest was re-frozen here; IterKey is unchanged since Task 1.
    assert key_digest(SegmentKey(global_seed=7, run_id="r0", iteration=1, walker_id=2)) == (
        SEGMENT_GOLDEN
    )
    assert key_digest(IterKey(global_seed=7, run_id="r0", iteration=1)) == (
        "ad6dad9eba1dcfe93d0f3286680456788a2c01d658a9c29e73d9a6d9a3ce213f"
    )


def test_derive_rng_output_matches_frozen_golden_draws():
    # A-m11: NEP 19 does not promise Generator distribution streams across
    # numpy versions; this pins them so an environment change that would
    # silently change every IC is caught. Frozen with numpy 2.4.3.
    key = ShotKey(global_seed=1, frame_id=2, shot_id=3, stage="ic")
    assert derive_rng(key).random(3).tolist() == [
        0.15810323368561308, 0.826980754151264, 0.9421681487161812,
    ]
    assert derive_rng(key, "x").normal(size=2).tolist() == [
        0.7124378512942477, -1.1811246723109268,
    ]
    assert derive_rng(key).integers(0, 2**31, 2).tolist() == [153660741, 339524109]
