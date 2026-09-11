"""VALID-06/07/10 tests: the three dataset-scoped read-back checks, each
driven by a proven mutator against `build_varying_baseline_dataset`
prefixes (plan 03-01, Task 3).
"""

import json
from pathlib import Path

import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import (
    build_varying_baseline_dataset,
    copy_dataset,
    corrupt_episodes_length,
    drop_tasks_row,
    inject_nan_into_parquet,
    inject_nan_into_stats,
)
from data.validation.context import ValidationContext
from data.validation.inventory import (
    check_episode_length_agreement,
    check_nan_inf,
    check_task_index_join,
)
from data.validation.registry import UNEVALUATED_KEY

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


# ---------------------------------------------------------------------------
# Test 1: clean multi-episode prefix passes all three checks with per-
# episode evidence in ascending episode_index order.
# ---------------------------------------------------------------------------


def test_clean_multi_episode_prefix_passes_all_three_checks(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=3)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    length_result = check_episode_length_agreement(ctx)[0]
    assert length_result.passed is True
    assert length_result.evidence["episode_count"] == 3
    assert length_result.evidence["mismatches"] == []

    nan_result = check_nan_inf(ctx)[0]
    assert nan_result.passed is True
    assert nan_result.evidence["total_non_finite"] == 0

    join_result = check_task_index_join(ctx)[0]
    assert join_result.passed is True
    assert join_result.evidence["orphan_task_indices"] == []


# ---------------------------------------------------------------------------
# Test 2: corrupt_episodes_length fails naming declared length and actual
# row count.
# ---------------------------------------------------------------------------


def test_corrupted_episode_length_fails_naming_both_numbers(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=2)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    corrupt_episodes_length(mutated, episode_index=0, delta=1)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_episode_length_agreement(ctx)[0]

    assert result.passed is False
    mismatch = result.evidence["mismatches"][0]
    assert mismatch["episode_index"] == 0
    assert mismatch["declared_length"] != mismatch["actual_row_count"]


# ---------------------------------------------------------------------------
# Test 3: drop_tasks_row fails naming the orphan task_index and the highest
# present index.
# ---------------------------------------------------------------------------


def test_dropped_tasks_row_fails_naming_orphan_and_highest_present(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=1)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    drop_tasks_row(mutated, task_index=0)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_task_index_join(ctx)[0]

    assert result.passed is False
    assert 0 in result.evidence["orphan_task_indices"]
    assert result.evidence["highest_present_task_index"] is None


# ---------------------------------------------------------------------------
# Test 4: a tasks.jsonl row whose task_index is the string "0" fails naming
# the type mismatch rather than matching integer 0.
# ---------------------------------------------------------------------------


def test_string_task_index_fails_naming_type_mismatch(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=1)
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    rows[0]["task_index"] = "0"
    tasks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_task_index_join(ctx)[0]

    assert result.passed is False
    type_error = result.evidence["type_errors"][0]
    assert type_error["value"] == "0"
    assert type_error["type"] == "str"


# ---------------------------------------------------------------------------
# Test 5: inject_nan_into_parquet / inject_nan_into_stats each fail
# anomalies.nan_inf naming the feature and location.
# ---------------------------------------------------------------------------


def test_nan_in_parquet_fails_naming_feature_and_location(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=1)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    inject_nan_into_parquet(mutated, feature="action", row_index=1, channel_index=2)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_nan_inf(ctx)[0]

    assert result.passed is False
    finding = next(f for f in result.evidence["findings"] if f["feature"] == "action")
    assert finding["row_index"] == 1
    assert finding["channel_index"] == 2


def test_nan_in_stats_fails_naming_feature_and_statistic(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=5, episodes=1)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    inject_nan_into_stats(mutated, feature="observation.state", stat="mean", channel_index=3)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_nan_inf(ctx)[0]

    assert result.passed is False
    finding = next(
        f for f in result.evidence["findings"] if f.get("feature") == "observation.state"
    )
    assert finding["stat"] == "mean"
    assert finding["channel_index"] == 3


# ---------------------------------------------------------------------------
# Test 6: a zero-episode prefix reports all three checks unevaluated,
# passed False.
# ---------------------------------------------------------------------------


def test_zero_episode_prefix_reports_all_three_unevaluated(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = tmp_path / "empty"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "data_engine").mkdir()
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    length_result = check_episode_length_agreement(ctx)[0]
    assert length_result.evidence[UNEVALUATED_KEY] is True
    assert length_result.passed is False

    nan_result = check_nan_inf(ctx)[0]
    assert nan_result.evidence[UNEVALUATED_KEY] is True
    assert nan_result.passed is False

    join_result = check_task_index_join(ctx)[0]
    assert join_result.evidence[UNEVALUATED_KEY] is True
    assert join_result.passed is False


# ---------------------------------------------------------------------------
# Test 7: an unparseable third line in meta/episodes.jsonl fails naming line
# 3, and validate_dataset still returns rather than raising.
# ---------------------------------------------------------------------------


def test_unparseable_episodes_line_fails_naming_line_number_not_raising(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=3, episodes=3)
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    lines = episodes_path.read_text().splitlines()
    lines[2] = "{not valid json"
    episodes_path.write_text("\n".join(lines) + "\n")

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_episode_length_agreement(ctx)[0]

    assert result.passed is False
    assert result.evidence["line"] == 3

    from data.validation.validate import validate_dataset

    outcome = validate_dataset(dataset_dir, profile)
    assert outcome.exit_code != 0
