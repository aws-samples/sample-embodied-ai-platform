"""Exact streaming statistics: Welford's algorithm, Chan's parallel-variance
merge, exact uint8 histograms, and the mergeable per-episode partial.

Every fixture below is an inline data literal (no fixture files), following
`test_profile.py`'s convention.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from data.contracts.containers import Disposition
from data.contracts.profile import load_profile
from data.tests.frame_fixtures import build_synthetic_episode
from data.writers.lerobot_v21 import LeRobotV21Writer
from data.writers.stats import (
    STATS_KEYS,
    Uint8Histogram,
    WelfordPartial,
    disposition_included_in_merge,
    finalize_stats,
    merge_partials,
    retained_value_bytes_per_episode,
    write_partial_file,
)

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
PROFILE = load_profile(PROFILE_PATH)


def _partial_from_samples(samples: list[list[float]]) -> WelfordPartial:
    width = len(samples[0])
    partial = WelfordPartial.empty(width)
    for sample in samples:
        partial.update(np.array(sample, dtype=np.float32))
    return partial


# --- Test 1: single-sample feature.


def test_single_sample_partial() -> None:
    partial = _partial_from_samples([[5.0]])
    stats = finalize_stats(partial)
    assert stats["count"] == [1]
    assert stats["mean"] == [5.0]
    assert stats["std"] == [0.0]
    assert stats["min"] == [5.0]
    assert stats["max"] == [5.0]
    assert stats["q01"] == [5.0]
    assert stats["q99"] == [5.0]
    assert set(stats) == set(STATS_KEYS)


# --- Test 2: two-sample feature — q01 is the lower sample, q99 the upper.


def test_two_sample_quantiles_are_exact_bounds() -> None:
    partial = _partial_from_samples([[1.0], [9.0]])
    stats = finalize_stats(partial)
    assert stats["q01"] == [1.0]
    assert stats["q99"] == [9.0]


# --- Test 3: merged mean/std over two synthetic episodes equal NumPy's
# values over the concatenation.


def test_merge_matches_numpy_over_concatenation() -> None:
    rng = np.random.default_rng(42)
    a_samples = rng.normal(loc=10.0, scale=2.0, size=(30, 3))
    b_samples = rng.normal(loc=10.0, scale=2.0, size=(40, 3))

    partial_a = WelfordPartial.empty(3)
    for row in a_samples:
        partial_a.update(row)
    partial_b = WelfordPartial.empty(3)
    for row in b_samples:
        partial_b.update(row)

    merged = merge_partials(partial_a, partial_b)
    concatenated = np.concatenate([a_samples, b_samples], axis=0)

    np.testing.assert_allclose(merged.mean, concatenated.mean(axis=0), atol=1e-6)
    merged_std = np.sqrt(merged.m2 / merged.count)
    np.testing.assert_allclose(merged_std, concatenated.std(axis=0), atol=1e-6)


# --- Test 4: same equality on a large-magnitude-offset series (the case a
# naive sum-of-squares accumulator catastrophically cancels).


def test_merge_matches_numpy_on_large_magnitude_offset_series() -> None:
    rng = np.random.default_rng(7)
    offset = 1e8
    a_samples = offset + rng.normal(loc=0.0, scale=1.0, size=(50, 2))
    b_samples = offset + rng.normal(loc=0.0, scale=1.0, size=(60, 2))

    partial_a = WelfordPartial.empty(2)
    for row in a_samples:
        partial_a.update(row)
    partial_b = WelfordPartial.empty(2)
    for row in b_samples:
        partial_b.update(row)

    merged = merge_partials(partial_a, partial_b)
    concatenated = np.concatenate([a_samples, b_samples], axis=0)

    np.testing.assert_allclose(merged.mean, concatenated.mean(axis=0), atol=1e-6)
    merged_std = np.sqrt(merged.m2 / merged.count)
    np.testing.assert_allclose(merged_std, concatenated.std(axis=0), atol=1e-6)


# --- Test 5: merging a zero-count partial with a populated one returns
# the populated one unchanged, in both argument orders.


def test_merge_with_zero_count_partial_is_identity() -> None:
    populated = _partial_from_samples([[1.0, 2.0], [3.0, 4.0]])
    empty = WelfordPartial.empty(2)

    result_left = merge_partials(empty, populated)
    result_right = merge_partials(populated, empty)

    for result in (result_left, result_right):
        assert result.count == populated.count
        np.testing.assert_array_equal(result.mean, populated.mean)
        np.testing.assert_array_equal(result.m2, populated.m2)


# --- Test 6: a constant-valued feature yields zero std with all seven
# statistics present — no channel removed.


def test_constant_feature_yields_zero_std_all_stats_present() -> None:
    partial = _partial_from_samples([[3.0], [3.0], [3.0]])
    stats = finalize_stats(partial)
    assert stats["std"] == [0.0]
    assert set(stats) == set(STATS_KEYS)
    assert stats["min"] == [3.0]
    assert stats["max"] == [3.0]


# --- Test 7: uint8 histogram has exactly 256 bins/channel, bin sums equal
# sample count, and quantiles match a direct NumPy computation over the
# same rank definition.


def test_uint8_histogram_exact_quantiles() -> None:
    rng = np.random.default_rng(3)
    samples = rng.integers(0, 256, size=(500, 4, 4, 3), dtype=np.uint8)

    hist = Uint8Histogram.empty(channels=3)
    for frame in samples:
        hist.update(frame)

    assert hist.counts.shape == (3, 256)
    total_pixels_per_channel = 500 * 4 * 4
    assert hist.counts.sum(axis=1).tolist() == [total_pixels_per_channel] * 3

    flat_per_channel = samples.reshape(-1, 3)
    for c in range(3):
        sorted_vals = np.sort(flat_per_channel[:, c])
        n = sorted_vals.shape[0]
        expected_q01 = sorted_vals[int(0.01 * (n - 1))]
        expected_q99_idx = min(max(int(np.ceil(0.99 * (n - 1))), 0), n - 1)
        expected_q99 = sorted_vals[expected_q99_idx]
        got_q01 = hist.quantile(0.01, "lower")[c]
        got_q99 = hist.quantile(0.99, "higher")[c]
        assert got_q01 == expected_q01
        assert got_q99 == expected_q99


# --- Test 8/9: meta/stats.json key set equals exactly info.json's
# float-dtype feature set, and keys are sorted / byte-identical across runs.


def _write_two_episode_dataset(out_dir: Path) -> None:
    writer = LeRobotV21Writer(out_dir, PROFILE)
    for n in (4, 3):
        episode = build_synthetic_episode(frame_count=n, fps=25.0, profile=PROFILE, task_prompt="t")
        writer.write_episode(episode)
    writer.finalize()


def test_stats_json_key_set_matches_info_json_float_features(tmp_path: Path) -> None:
    _write_two_episode_dataset(tmp_path)

    with open(tmp_path / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    expected_keys = {name for name, feat in info["features"].items() if "float" in feat["dtype"]}

    with open(tmp_path / "meta" / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    assert set(stats.keys()) == expected_keys
    for feat_name in expected_keys:
        assert set(stats[feat_name].keys()) == set(STATS_KEYS)
    for name, feat in info["features"].items():
        if feat["dtype"] == "video":
            assert name not in stats


def test_stats_json_keys_sorted_and_byte_identical_across_runs(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    _write_two_episode_dataset(dir_a)
    _write_two_episode_dataset(dir_b)

    stats_a = (dir_a / "meta" / "stats.json").read_text(encoding="utf-8")
    stats_b = (dir_b / "meta" / "stats.json").read_text(encoding="utf-8")
    assert stats_a == stats_b

    parsed = json.loads(stats_a)
    assert list(parsed.keys()) == sorted(parsed.keys())


# --- Test 10: a per-episode partial file is written, carries the
# second-moment term (not a finished std), and the applied disposition.


def test_per_episode_partial_file_written_and_named_in_output_paths(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    episode = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    result = writer.write_episode(episode)

    assert "stats_partial" in result.output_paths
    partial_path = Path(result.output_paths["stats_partial"])
    assert partial_path.exists()

    with open(partial_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert "m2" in payload["features"]["observation.state"]
    assert "std" not in payload["features"]["observation.state"]
    assert payload["disposition"] in (None, "PASS", "PASS_WITH_FLAGS", "REJECT")
    assert payload["included_in_merge"] is True


# --- Test 11: REJECT is excluded from the merge; PASS_WITH_FLAGS is
# included.


def test_reject_disposition_excluded_pass_with_flags_included(tmp_path: Path) -> None:
    assert disposition_included_in_merge(Disposition.REJECT) is False
    assert disposition_included_in_merge(Disposition.PASS_WITH_FLAGS) is True
    assert disposition_included_in_merge(Disposition.PASS) is True
    assert disposition_included_in_merge(None) is True

    writer = LeRobotV21Writer(tmp_path, PROFILE)
    kept_episode = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        task_prompt="t",
        disposition=Disposition.PASS_WITH_FLAGS,
        state_values={"left_arm": [np.full(7, 100.0, dtype=np.float32)] * 2},
    )
    rejected_episode = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        task_prompt="t",
        disposition=Disposition.REJECT,
        state_values={"left_arm": [np.full(7, -999.0, dtype=np.float32)] * 2},
    )
    writer.write_episode(kept_episode)
    writer.write_episode(rejected_episode)
    writer.finalize()

    with open(tmp_path / "meta" / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    # -999.0 must never influence the merged state stats since that
    # episode's disposition is REJECT.
    assert stats["observation.state"]["min"][0] != pytest.approx(-999.0)
    assert stats["observation.state"]["max"][0] == pytest.approx(100.0)


# --- Test 12: the memory ceiling expression matches the actual retained
# buffer size for one shipped-profile episode.


def test_memory_ceiling_matches_retained_buffer_size() -> None:
    # Derived from the shipped profile's own state_vector_order -- never a
    # hardcoded literal (Phase 3 D-09: a written-down total is what produced
    # the three unreconciled emitted-width figures this correction exists to
    # close).
    state_topics_by_key = {f.state_key: f for f in PROFILE.state_topics.values()}
    state_width = sum(
        len(state_topics_by_key[key].joint_names) for key in PROFILE.state_vector_order
    )
    frame_count = 17
    partial = WelfordPartial.empty(state_width)
    for _ in range(frame_count):
        partial.update(np.zeros(state_width, dtype=np.float32))

    expected_bytes = retained_value_bytes_per_episode(state_width, frame_count)
    assert partial.values.nbytes == expected_bytes


def test_write_partial_file_roundtrip(tmp_path: Path) -> None:
    partial = _partial_from_samples([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "meta" / "stats_partials" / "episode_000000.json"
    write_partial_file(
        path,
        disposition=Disposition.PASS,
        included_in_merge=True,
        feature_partials={"observation.state": partial},
    )
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["disposition"] == "PASS"
    assert payload["included_in_merge"] is True
