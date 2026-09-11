"""VALID-09/D-13 tests: independent numpy recomputation of `meta/stats.json`.

Most behaviours are driven by `data/tests/mutators.py::perturb_stats_value`
against `build_varying_baseline_dataset` prefixes. A few structural
behaviours (a spurious video-feature entry, a missing/extra feature key)
edit `meta/stats.json`/`meta/info.json` directly with small local helpers —
neither is a numeric-value mutation an existing mutator already covers.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import build_varying_baseline_dataset, copy_dataset, perturb_stats_value
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.validation.stats_oracle import (
    EXACT_STATS_KEYS,
    MEAN_STD_RTOL,
    check_stats_recomputation,
    recompute_feature_stats,
)
from data.writers.stats import _rank_quantile

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _read_stats(dataset_dir: Path) -> dict[str, Any]:
    with open(dataset_dir / "meta" / "stats.json", encoding="utf-8") as f:
        return json.load(f)


def _write_stats(dataset_dir: Path, payload: dict[str, Any]) -> None:
    path = dataset_dir / "meta" / "stats.json"
    tmp = path.parent / f".{path.name}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp, path)


def _inject_video_feature_into_stats(dataset_dir: Path, video_key: str) -> None:
    """Adds a spurious camera-feature entry to `meta/stats.json` — no
    existing mutator does this (they all target real float channels)."""
    payload = _read_stats(dataset_dir)
    payload[video_key] = {
        "mean": [0.5],
        "std": [0.1],
        "min": [0.0],
        "max": [1.0],
        "q01": [0.0],
        "q99": [1.0],
        "count": [1],
    }
    _write_stats(dataset_dir, payload)


def _drop_stats_feature(dataset_dir: Path, feature: str) -> None:
    payload = _read_stats(dataset_dir)
    del payload[feature]
    _write_stats(dataset_dir, payload)


def _hash_meta_tree(dataset_dir: Path) -> dict[str, str]:
    meta_dir = dataset_dir / "meta"
    digests: dict[str, str] = {}
    for path in sorted(meta_dir.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(meta_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


# ---------------------------------------------------------------------------
# Test 1: a clean prefix passes, every statistic compared per channel with
# its applied tolerance recorded.
# ---------------------------------------------------------------------------


def test_clean_prefix_passes_with_recorded_tolerances(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_stats_recomputation(ctx)[0]

    assert result.passed is True
    assert result.evidence["unevaluated_features"] == []
    feature_names = {pf["feature"] for pf in result.evidence["per_feature"]}
    assert feature_names == {"observation.state", "action", "timestamp"}
    for pf in result.evidence["per_feature"]:
        assert pf["mismatches"] == []
        for key in EXACT_STATS_KEYS:
            assert pf["comparisons"][key]["tolerance"] == "exact"
            assert pf["comparisons"][key]["equal"] is True
        for key in ("mean", "std"):
            assert pf["comparisons"][key]["tolerance"] == f"rtol={MEAN_STD_RTOL}"
            assert pf["comparisons"][key]["equal"] is True
        assert pf["comparisons"]["count"]["tolerance"] == "exact"


# ---------------------------------------------------------------------------
# Test 2: perturbing mean beyond rtol fails, naming feature/statistic/
# channel/both values.
# ---------------------------------------------------------------------------


def test_perturb_mean_beyond_tolerance_fails(tmp_path: Path, profile: EmbodimentProfile) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6, episodes=2)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    perturb_stats_value(mutated, feature="observation.state", stat="mean", channel_index=0, delta=5.0)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.passed is False
    pf = next(p for p in result.evidence["per_feature"] if p["feature"] == "observation.state")
    mismatch = next(m for m in pf["mismatches"] if m["statistic"] == "mean")
    assert mismatch["expected"][0] != mismatch["actual"][0]
    assert pf["comparisons"]["mean"]["equal"] is False


# ---------------------------------------------------------------------------
# Test 3: perturbing min by any nonzero amount fails — exact comparison.
# ---------------------------------------------------------------------------


def test_perturb_min_by_any_amount_fails(tmp_path: Path, profile: EmbodimentProfile) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6, episodes=2)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    perturb_stats_value(mutated, feature="action", stat="min", channel_index=0, delta=1e-4)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.passed is False
    pf = next(p for p in result.evidence["per_feature"] if p["feature"] == "action")
    assert pf["comparisons"]["min"]["equal"] is False


# ---------------------------------------------------------------------------
# Test 4: perturbing mean by less than rtol passes.
# ---------------------------------------------------------------------------


def test_perturb_mean_within_tolerance_passes(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    stats_before = _read_stats(dataset_dir)
    mean_value = stats_before["action"]["mean"][0]
    # A delta well under MEAN_STD_RTOL relative to the real magnitude —
    # legitimate float32-accumulation-order noise, not a real defect.
    tiny_delta = abs(mean_value) * (MEAN_STD_RTOL / 100) if mean_value else 1e-10
    perturb_stats_value(mutated, feature="action", stat="mean", channel_index=0, delta=tiny_delta)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    pf = next(p for p in result.evidence["per_feature"] if p["feature"] == "action")
    assert pf["comparisons"]["mean"]["equal"] is True
    assert pf["mismatches"] == []
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 5: q01/q99 match exactly via the writer's rank convention; numpy's
# default quantile interpolation is shown to disagree on the same data.
# ---------------------------------------------------------------------------


def test_rank_quantile_convention_matters_not_incidental() -> None:
    sorted_values = np.array([1.0, 2.0, 3.0, 4.0])
    writer_q01 = _rank_quantile(sorted_values, 0.01, "lower")
    writer_q99 = _rank_quantile(sorted_values, 0.99, "higher")
    numpy_default_q01 = np.quantile(sorted_values, 0.01)
    numpy_default_q99 = np.quantile(sorted_values, 0.99)

    # The load-bearing proof: swapping to numpy's own convention would
    # disagree with the writer's on the very same data.
    assert float(writer_q01) != float(numpy_default_q01)
    assert float(writer_q99) != float(numpy_default_q99)


def test_clean_prefix_q01_q99_match_exactly(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_stats_recomputation(ctx)[0]

    assert result.passed is True
    for pf in result.evidence["per_feature"]:
        assert pf["comparisons"]["q01"]["equal"] is True
        assert pf["comparisons"]["q99"]["equal"] is True


def test_recompute_feature_stats_uses_rank_quantile_not_numpy_default() -> None:
    column = np.array([[1.0], [2.0], [3.0], [4.0]])
    recomputed = recompute_feature_stats(column)
    sorted_column = np.sort(column, axis=0)
    expected_q01 = _rank_quantile(sorted_column, 0.01, "lower")
    expected_q99 = _rank_quantile(sorted_column, 0.99, "higher")
    assert np.array_equal(recomputed["q01"], expected_q01)
    assert np.array_equal(recomputed["q99"], expected_q99)
    assert not np.array_equal(recomputed["q01"], np.quantile(column, 0.01, axis=0))


# ---------------------------------------------------------------------------
# Test 6: a camera feature entry in meta/stats.json fails, naming it.
# ---------------------------------------------------------------------------


def test_video_feature_in_stats_json_fails_naming_it(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    _inject_video_feature_into_stats(dataset_dir, "observation.images.head")

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.passed is False
    assert result.evidence["extra_in_stats_json"] == ["observation.images.head"]
    assert result.evidence["missing_from_stats_json"] == []


# ---------------------------------------------------------------------------
# Test 7: a missing float feature and an extra unrelated key each fail,
# naming the direction of the mismatch.
# ---------------------------------------------------------------------------


def test_missing_and_extra_stats_keys_fail_naming_direction(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    _drop_stats_feature(dataset_dir, "timestamp")
    payload = _read_stats(dataset_dir)
    payload["not_a_real_feature"] = payload["action"]
    _write_stats(dataset_dir, payload)

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.passed is False
    assert "timestamp" in result.evidence["missing_from_stats_json"]
    assert "not_a_real_feature" in result.evidence["extra_in_stats_json"]


# ---------------------------------------------------------------------------
# Test 8: read-only — digests unchanged, two runs produce identical
# evidence.
# ---------------------------------------------------------------------------


def test_check_is_read_only_and_deterministic(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    digests_before = _hash_meta_tree(dataset_dir)
    first = check_stats_recomputation(ctx)[0]
    digests_after_first = _hash_meta_tree(dataset_dir)
    second = check_stats_recomputation(ctx)[0]
    digests_after_second = _hash_meta_tree(dataset_dir)

    assert digests_before == digests_after_first == digests_after_second
    assert first.evidence == second.evidence
    assert first.passed == second.passed


# ---------------------------------------------------------------------------
# Zero-count merged partial routes to the unevaluated marker, not a raise
# propagated from the writer's own zero-count `finalize_stats` guard.
# ---------------------------------------------------------------------------


def test_zero_count_merged_partial_is_unevaluated_not_raised(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    import shutil

    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    # Every feature's merged partial becomes unreachable (count 0 for the
    # purposes of this check) once the partials directory itself is gone —
    # `meta/stats.json`/`meta/info.json` still agree on the feature set, so
    # this exercises the zero-count branch specifically, not the missing/
    # extra-feature branch.
    shutil.rmtree(dataset_dir / "meta" / "stats_partials")

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.passed is False
    assert SKIP_REASON_KEY in result.evidence
    assert len(result.evidence["unevaluated_features"]) == 3
