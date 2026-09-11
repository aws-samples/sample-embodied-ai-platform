"""Tests for `data/validation/diff.py::diff_datasets` (REGR-02).

REGR-02 is the one regression requirement fully testable in CI — no
skip-if-absent path anywhere in this module, unlike `test_validation_
fingerprint.py`'s REGR-01 coverage (D-18).
"""

import json
from pathlib import Path

import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import build_varying_baseline_dataset, copy_dataset
from data.validation.diff import diff_datasets

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _read_info(prefix: Path) -> dict:
    with open(prefix / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def _write_info(prefix: Path, payload: dict) -> None:
    with open(prefix / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Test 1: diffing a prefix against a byte-identical copy of itself returns
# an empty mapping.
# ---------------------------------------------------------------------------


def test_identical_copy_produces_empty_diff(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=4)
    copy_dir = copy_dataset(dataset_dir, tmp_path / "copy")

    assert diff_datasets(dataset_dir, copy_dir) == {}


# ---------------------------------------------------------------------------
# Test 2: a prefix with one fewer episode reports the episode count with
# both values and the per-episode length distribution delta.
# ---------------------------------------------------------------------------


def test_one_fewer_episode_reports_count_and_length_distribution(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(
        tmp_path / "a", profile, camera_frames=4, episodes=2
    )
    prefix_b = build_varying_baseline_dataset(
        tmp_path / "b", profile, camera_frames=4, episodes=1
    )

    diff = diff_datasets(prefix_a, prefix_b)

    assert diff["total_episodes"] == {"a": 2, "b": 1}
    assert "length_distribution" in diff
    assert diff["length_distribution"]["a"]["count"] == 2
    assert diff["length_distribution"]["b"]["count"] == 1


# ---------------------------------------------------------------------------
# Test 3: a prefix whose one feature has a different dtype reports that
# feature and both dtypes; a different shape reports both shapes.
# ---------------------------------------------------------------------------


def test_differing_dtype_and_shape_report_both_values(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(tmp_path / "a", profile, camera_frames=4)
    prefix_b = copy_dataset(prefix_a, tmp_path / "b")

    info_b = _read_info(prefix_b)
    info_b["features"]["action"]["dtype"] = "float64"
    info_b["features"]["timestamp"]["shape"] = [2]
    _write_info(prefix_b, info_b)

    diff = diff_datasets(prefix_a, prefix_b)

    assert diff["feature_changes"]["action"]["dtype"] == {"a": "float32", "b": "float64"}
    assert diff["feature_changes"]["timestamp"]["shape"] == {"a": [1], "b": [2]}


# ---------------------------------------------------------------------------
# Test 4: a prefix with an added or removed camera reports the camera-set
# difference with both sides sorted.
# ---------------------------------------------------------------------------


def test_removed_camera_reports_sorted_camera_set_difference(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    narrowed_profile = profile.model_copy(
        update={
            "cameras": {
                **profile.cameras,
                "wrist_right": profile.cameras["wrist_right"].model_copy(
                    update={"emit": False}
                ),
            }
        }
    )
    prefix_a = build_varying_baseline_dataset(tmp_path / "a", profile, camera_frames=4)
    prefix_b = build_varying_baseline_dataset(tmp_path / "b", narrowed_profile, camera_frames=4)

    diff = diff_datasets(prefix_a, prefix_b)

    assert diff["camera_set"] == {
        "a": sorted(
            [
                "observation.images.head",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            ]
        ),
        "b": sorted(["observation.images.head", "observation.images.wrist_left"]),
    }


# ---------------------------------------------------------------------------
# Test 5: an fps of 29.95 against 30.0 reports both values — no tolerance
# absorbs it.
# ---------------------------------------------------------------------------


def test_fps_difference_reported_exactly_no_tolerance(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(tmp_path / "a", profile, camera_frames=4)
    prefix_b = copy_dataset(prefix_a, tmp_path / "b")

    info_a = _read_info(prefix_a)
    info_a["fps"] = 29.95
    _write_info(prefix_a, info_a)

    info_b = _read_info(prefix_b)
    info_b["fps"] = 30.0
    _write_info(prefix_b, info_b)

    diff = diff_datasets(prefix_a, prefix_b)

    assert diff["fps"] == {"a": 29.95, "b": 30.0}


# ---------------------------------------------------------------------------
# Test 6: two feature keys differing only in letter case appear as two
# names in the set difference.
# ---------------------------------------------------------------------------


def test_case_differing_feature_keys_appear_as_two_names(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(tmp_path / "a", profile, camera_frames=4)
    prefix_b = copy_dataset(prefix_a, tmp_path / "b")

    info_b = _read_info(prefix_b)
    info_b["features"]["Action"] = info_b["features"].pop("action")
    _write_info(prefix_b, info_b)

    diff = diff_datasets(prefix_a, prefix_b)

    assert "action" in diff["feature_names"]["a"]
    assert "Action" not in diff["feature_names"]["a"]
    assert "Action" in diff["feature_names"]["b"]
    assert "action" not in diff["feature_names"]["b"]


# ---------------------------------------------------------------------------
# Test 7: a missing meta/info.json on either side produces a named-reason
# entry rather than an empty mapping.
# ---------------------------------------------------------------------------


def test_missing_info_json_produces_named_reason_not_empty_mapping(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(tmp_path / "a", profile, camera_frames=4)
    prefix_b = copy_dataset(prefix_a, tmp_path / "b")
    (prefix_b / "meta" / "info.json").unlink()

    diff = diff_datasets(prefix_a, prefix_b)

    assert diff != {}
    assert "error" in diff
    assert "info.json" in diff["error"]
    assert "'b'" in diff["error"]


# ---------------------------------------------------------------------------
# Test 8: a zero-episode prefix diffed against a populated one reports both
# episode counts and the full feature-name set difference.
# ---------------------------------------------------------------------------


def test_zero_episode_prefix_reports_counts_and_feature_name_difference(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    from data.writers.lerobot_v21 import LeRobotV21Writer

    narrowed_profile = profile.model_copy(
        update={
            "cameras": {
                **profile.cameras,
                "wrist_right": profile.cameras["wrist_right"].model_copy(
                    update={"emit": False}
                ),
            }
        }
    )
    zero_dir = tmp_path / "zero"
    zero_dir.mkdir()
    LeRobotV21Writer(zero_dir, narrowed_profile, stdin_isatty=lambda: False).finalize()

    populated_dir = build_varying_baseline_dataset(tmp_path / "populated", profile, camera_frames=4)

    diff = diff_datasets(zero_dir, populated_dir)

    assert diff["total_episodes"] == {"a": 0, "b": 1}
    assert diff["camera_set"]["a"] != diff["camera_set"]["b"]
    assert "feature_names" in diff
    assert set(diff["feature_names"]["a"]) < set(diff["feature_names"]["b"])
    assert diff["feature_names"]["a"] == sorted(diff["feature_names"]["a"])
    assert diff["feature_names"]["b"] == sorted(diff["feature_names"]["b"])


# ---------------------------------------------------------------------------
# Test 9: diff_datasets(a, b) and diff_datasets(b, a) report the same set of
# changed keys with the two sides swapped.
# ---------------------------------------------------------------------------


def test_diff_is_symmetric_with_sides_swapped(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    prefix_a = build_varying_baseline_dataset(
        tmp_path / "a", profile, camera_frames=4, episodes=2
    )
    prefix_b = build_varying_baseline_dataset(
        tmp_path / "b", profile, camera_frames=4, episodes=1
    )

    diff_ab = diff_datasets(prefix_a, prefix_b)
    diff_ba = diff_datasets(prefix_b, prefix_a)

    assert set(diff_ab.keys()) == set(diff_ba.keys())
    for key in ("total_episodes", "total_frames"):
        assert diff_ab[key]["a"] == diff_ba[key]["b"]
        assert diff_ab[key]["b"] == diff_ba[key]["a"]
    assert diff_ab["length_distribution"]["a"] == diff_ba["length_distribution"]["b"]
    assert diff_ab["length_distribution"]["b"] == diff_ba["length_distribution"]["a"]
