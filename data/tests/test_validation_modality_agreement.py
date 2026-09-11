"""Tests for `data/validation/modality_agreement.py`:
`check_modality_agreement` (VALID-08) and `check_modality_slice_range`
(VALID-05).

Every fixture in this module assigns each declared state/action field a
DISTINCT, widely-separated numeric base value (left_arm ~10, right_arm
~100, torso ~1000, left_gripper ~-50, right_gripper ~-500) — deliberately
far enough apart that a one-index shift in the on-disk `meta/modality.json`
slice map pulls a value belonging to a neighboring field far outside the
target field's own `meta/stats.json`-derived bounds, making
`check_modality_slice_range`'s physical-range mechanism exercised for real
rather than by coincidence. Hand-edited variants (Tests 3-8) mutate a
copied prefix's `meta/modality.json`/`meta/modality_config.py` directly in
this module, per the plan's own instruction; `overlap_modality_slice` (Test
2) is the one mutator reused from `data/tests/mutators.py`.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.provenance.recorder import ProvenanceRecorder
from data.tests.frame_fixtures import build_synthetic_episode
from data.tests.mutators import build_fixture_source_mcap, copy_dataset, overlap_modality_slice
from data.validation.context import ValidationContext
from data.validation.modality_agreement import (
    check_modality_agreement,
    check_modality_slice_range,
)
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.writers.lerobot_v21 import LeRobotV21Writer

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"

_STATE_BASES = {
    "left_arm": 10.0,
    "right_arm": 100.0,
    "torso": 1000.0,
    "left_gripper": -50.0,
    "right_gripper": -500.0,
}
_ACTION_BASES = {
    "left_arm": 10.0,
    "right_arm": 100.0,
    "left_gripper": -50.0,
    "right_gripper": -500.0,
}


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _distinct_state_action_values(
    profile: EmbodimentProfile, frame_count: int
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    state_by_key = {f.state_key: f for f in profile.state_topics.values()}
    state_values: dict[str, list[np.ndarray]] = {}
    for key in profile.state_vector_order:
        width = len(state_by_key[key].joint_names)
        base = _STATE_BASES[key]
        state_values[key] = [
            np.full(width, base + i * 0.01, dtype=np.float32) for i in range(frame_count)
        ]

    action_by_key = {f.action_key: f for f in profile.action_topics.values()}
    action_values: dict[str, list[np.ndarray]] = {}
    for key in profile.action_primary:
        width = len(action_by_key[key].joint_names)
        base = _ACTION_BASES[key]
        action_values[key] = [
            np.full(width, base + i * 0.01, dtype=np.float32) for i in range(frame_count)
        ]
    return state_values, action_values


def _build_dataset(
    tmp_path: Path, profile: EmbodimentProfile, *, frame_count: int = 3, episodes: int = 1
) -> Path:
    """A structural, distinct-per-field baseline: real values, real
    provenance records (so `ValidationContext.episode_indices` can discover
    the written episode), no dependency on decoding real camera pixels
    (`check_modality_agreement`/`check_modality_slice_range` never touch
    video)."""
    import dataclasses

    from data.contracts.containers import CanonicalEpisode

    dest = tmp_path / "dataset"
    dest.mkdir()
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=max(frame_count, 1))
    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    recorder = ProvenanceRecorder(dest, profile)

    for _ in range(episodes):
        state_values, action_values = _distinct_state_action_values(profile, frame_count)
        episode = build_synthetic_episode(
            frame_count=frame_count,
            fps=10.0,
            profile=profile,
            state_values=state_values,
            action_values=action_values,
            task_prompt="t",
        )
        header = dataclasses.replace(episode.header, episode_id=str(source_mcap))
        rebuilt = CanonicalEpisode(header=header, frames=episode.frames)
        write_result = writer.write_episode(rebuilt)
        recorder.record(header, write_result)

    writer.finalize()
    return dest


def _build_zero_episode_dataset(tmp_path: Path, profile: EmbodimentProfile) -> Path:
    dest = tmp_path / "zero_episode_dataset"
    dest.mkdir()
    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    writer.finalize()
    return dest


def _ctx(dataset_dir: Path, profile: EmbodimentProfile) -> ValidationContext:
    return ValidationContext(out_dir=dataset_dir, profile=profile)


def _read_modality_json(dataset_dir: Path) -> dict[str, Any]:
    with open(dataset_dir / "meta" / "modality.json", encoding="utf-8") as f:
        return json.load(f)


def _write_modality_json(dataset_dir: Path, payload: dict[str, Any]) -> None:
    with open(dataset_dir / "meta" / "modality.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


def _shift_field(dataset_dir: Path, scope: str, key: str, delta: int) -> None:
    """Hand-edit: shifts one field's `start` and `end` by `delta`, keeping
    its width unchanged."""
    payload = _read_modality_json(dataset_dir)
    payload[scope][key]["start"] += delta
    payload[scope][key]["end"] += delta
    _write_modality_json(dataset_dir, payload)


def _shift_all_fields(dataset_dir: Path, scope: str, delta: int) -> None:
    payload = _read_modality_json(dataset_dir)
    for entry in payload[scope].values():
        entry["start"] += delta
        entry["end"] += delta
    _write_modality_json(dataset_dir, payload)


def _widen_last_field_end(dataset_dir: Path, scope: str, delta: int) -> str:
    payload = _read_modality_json(dataset_dir)
    last_key = max(payload[scope], key=lambda k: payload[scope][k]["end"])
    payload[scope][last_key]["end"] += delta
    _write_modality_json(dataset_dir, payload)
    return last_key


def _rename_json_key(dataset_dir: Path, scope: str, old_key: str, new_key: str) -> None:
    payload = _read_modality_json(dataset_dir)
    payload[scope][new_key] = payload[scope].pop(old_key)
    _write_modality_json(dataset_dir, payload)


def _reorder_python_config_keys(
    dataset_dir: Path, original_order: list[str], reordered_order: list[str]
) -> None:
    """Hand-edit: rewrites the `modality_config.py` source text, swapping
    ONE `modality_keys=[...]` list literal for a reordered one — a targeted
    string replace of the exact `repr()` text `_generate_modality_config_source`
    writes, unique to the scope being reordered (state/action/video/language
    lists never share content)."""
    config_path = dataset_dir / "meta" / "modality_config.py"
    source = config_path.read_text(encoding="utf-8")
    original_repr = repr(original_order)
    assert original_repr in source, f"{original_repr!r} not found verbatim in {config_path}"
    reordered_repr = repr(reordered_order)
    new_source = source.replace(original_repr, reordered_repr, 1)
    config_path.write_text(new_source, encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: a clean prefix passes both checks; per-field evidence is emitted
# in ascending `start` order.
# ---------------------------------------------------------------------------


def test_clean_prefix_passes_both_checks_in_ascending_start_order(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    ctx = _ctx(dataset_dir, profile)

    (agreement_result,) = check_modality_agreement(ctx)
    assert agreement_result.passed, agreement_result.evidence
    assert agreement_result.evidence["violations"] == []

    (slice_result,) = check_modality_slice_range(ctx)
    assert slice_result.passed, slice_result.evidence
    assert slice_result.evidence["adjacency_violations"] == []

    state_field_order = [
        f["field"] for f in slice_result.evidence["per_field"] if f["scope"] == "state"
    ]
    assert state_field_order == list(profile.state_vector_order)
    action_field_order = [
        f["field"] for f in slice_result.evidence["per_field"] if f["scope"] == "action"
    ]
    assert action_field_order == list(profile.action_primary)

    for field_evidence in slice_result.evidence["per_field"]:
        assert field_evidence["passed"] is True
        assert "blind_spot" in field_evidence


# ---------------------------------------------------------------------------
# Test 2: overlap_modality_slice makes consistency.modality_agreement fail
# naming both overlapping fields and both boundaries.
# ---------------------------------------------------------------------------


def test_overlap_modality_slice_fails_agreement_naming_both_fields_and_boundaries(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    overlap_modality_slice(mutated_dir, key="left_arm", scope="state", by=1)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    overlaps = [v for v in result.evidence["violations"] if v["kind"] == "overlap"]
    assert overlaps, result.evidence["violations"]
    overlap = overlaps[0]
    assert {overlap["field_a"], overlap["field_b"]} == {"left_arm", "right_arm"}
    assert overlap["end_a"] is not None
    assert overlap["start_b"] is not None


# ---------------------------------------------------------------------------
# Test 3: a one-index gap between two adjacent fields fails naming the gap.
# ---------------------------------------------------------------------------


def test_one_index_gap_fails_agreement_naming_the_gap(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    _shift_field(mutated_dir, "state", "right_arm", delta=1)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    gaps = [v for v in result.evidence["violations"] if v["kind"] == "gap"]
    assert gaps, result.evidence["violations"]
    gap = gaps[0]
    assert gap["field_a"] == "left_arm"
    assert gap["field_b"] == "right_arm"


# ---------------------------------------------------------------------------
# Test 4: a slice map whose first field starts at 1 instead of 0 fails
# naming that field and its start.
# ---------------------------------------------------------------------------


def test_first_field_start_of_one_fails_agreement_naming_field_and_start(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    payload = _read_modality_json(mutated_dir)
    first_key = min(payload["state"], key=lambda k: payload["state"][k]["start"])
    payload["state"][first_key]["start"] += 1
    _write_modality_json(mutated_dir, payload)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    not_zero = [
        v for v in result.evidence["violations"] if v["kind"] == "does_not_start_at_zero"
    ]
    assert not_zero, result.evidence["violations"]
    assert not_zero[0]["field"] == first_key
    assert not_zero[0]["start"] == 1


# ---------------------------------------------------------------------------
# Test 5: a slice map whose widths sum to one more than the parquet column
# width fails naming both totals.
# ---------------------------------------------------------------------------


def test_width_sum_one_greater_fails_agreement_naming_both_totals(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    _widen_last_field_end(mutated_dir, "state", delta=1)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    width_mismatches = [
        v for v in result.evidence["violations"] if v["kind"] == "width_sum_mismatch"
    ]
    assert width_mismatches, result.evidence["violations"]
    mismatch = width_mismatches[0]
    assert mismatch["actual_total"] == 21
    assert mismatch["declared_width"] == 20


# ---------------------------------------------------------------------------
# Test 6: renaming one key in meta/modality.json so it no longer appears in
# the generated Python config's modality_keys fails naming the key and
# which artifact is missing it.
# ---------------------------------------------------------------------------


def test_renamed_json_key_fails_agreement_naming_key_and_missing_artifact(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    _rename_json_key(mutated_dir, "state", "left_gripper", "left_gripper_renamed")

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    key_mismatches = [
        v
        for v in result.evidence["violations"]
        if v["kind"] == "key_set_mismatch" and v.get("scope") == "state"
    ]
    assert key_mismatches, result.evidence["violations"]
    mismatch = key_mismatches[0]
    assert "left_gripper" in mismatch["missing_from_json"]
    assert "left_gripper_renamed" in mismatch["missing_from_python_config"]
    assert "left_gripper_renamed" in mismatch["missing_from_profile"]


# ---------------------------------------------------------------------------
# Test 7: reordering the generated Python config's modality_keys list
# relative to the profile's declared ordering fails naming all three
# orderings.
# ---------------------------------------------------------------------------


def test_reordered_python_config_keys_fails_agreement_naming_all_three_orderings(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")

    original_order = list(profile.state_vector_order)
    reordered_order = [original_order[1], original_order[0], *original_order[2:]]
    _reorder_python_config_keys(mutated_dir, original_order, reordered_order)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    ordering_mismatches = [
        v
        for v in result.evidence["violations"]
        if v["kind"] == "ordering_mismatch" and v.get("scope") == "state"
    ]
    assert ordering_mismatches, result.evidence["violations"]
    mismatch = ordering_mismatches[0]
    assert mismatch["profile_order"] == original_order
    assert mismatch["python_config_order"] == reordered_order
    assert mismatch["json_ascending_start_order"] == original_order


# ---------------------------------------------------------------------------
# Test 8: a key differing only in letter case between the two artifacts is
# reported as two different keys, not matched.
# ---------------------------------------------------------------------------


def test_case_differing_key_reported_as_two_different_keys(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    _rename_json_key(mutated_dir, "state", "left_gripper", "Left_Gripper")

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_agreement(ctx)
    assert not result.passed

    key_mismatches = [
        v
        for v in result.evidence["violations"]
        if v["kind"] == "key_set_mismatch" and v.get("scope") == "state"
    ]
    assert key_mismatches, result.evidence["violations"]
    mismatch = key_mismatches[0]
    assert "Left_Gripper" in mismatch["json_keys"]
    assert "left_gripper" not in mismatch["json_keys"]
    assert "left_gripper" in mismatch["python_config_keys"]
    assert "Left_Gripper" not in mismatch["python_config_keys"]


# ---------------------------------------------------------------------------
# Test 9: a shifted slice map — every range moved by one index without
# changing widths — makes consistency.modality_slice_range fail because the
# sliced values then fall outside the range independently derived for that
# field.
# ---------------------------------------------------------------------------


def test_shifted_slice_map_fails_slice_range_with_out_of_range_values(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_dataset(tmp_path, profile)
    mutated_dir = copy_dataset(dataset_dir, tmp_path / "mutated")
    _shift_all_fields(mutated_dir, "state", delta=1)
    _shift_all_fields(mutated_dir, "action", delta=1)

    ctx = _ctx(mutated_dir, profile)
    (result,) = check_modality_slice_range(ctx)
    assert not result.passed

    failed_fields = [f for f in result.evidence["per_field"] if not f["passed"]]
    assert failed_fields, result.evidence["per_field"]

    # left_arm's shifted slice must pull in a value from right_arm's very
    # different numeric range — a genuine value_out_of_range failure, not
    # merely a boundary-overflow artifact on whichever field lands last.
    left_arm_entries = [
        f for f in result.evidence["per_field"] if f["scope"] == "state" and f["field"] == "left_arm"
    ]
    assert left_arm_entries
    assert left_arm_entries[0]["passed"] is False
    assert left_arm_entries[0]["reason"] == "value_out_of_range"


# ---------------------------------------------------------------------------
# Test 10: a zero-episode dataset reports consistency.modality_slice_range
# as explicitly unevaluated, while consistency.modality_agreement still
# runs.
# ---------------------------------------------------------------------------


def test_zero_episode_prefix_slice_range_unevaluated_agreement_still_runs(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = _build_zero_episode_dataset(tmp_path, profile)
    ctx = _ctx(dataset_dir, profile)

    (slice_result,) = check_modality_slice_range(ctx)
    assert slice_result.evidence[UNEVALUATED_KEY] is True
    assert "zero-episode" in slice_result.evidence[SKIP_REASON_KEY]

    (agreement_result,) = check_modality_agreement(ctx)
    assert UNEVALUATED_KEY not in agreement_result.evidence
    assert agreement_result.passed is True
    assert agreement_result.evidence["violations"] == []


# ---------------------------------------------------------------------------
# Test 11: the agreement check does not import the generated Python config
# module — proven by staying green in this environment, where that module's
# own top-level imports (targeting the uninstalled gr00t package) would
# raise if actually executed.
# ---------------------------------------------------------------------------


def test_agreement_check_does_not_import_generated_module(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    import sys

    assert "gr00t" not in sys.modules, "gr00t must not be importable in this test environment"

    dataset_dir = _build_dataset(tmp_path, profile)
    ctx = _ctx(dataset_dir, profile)

    # If check_modality_agreement executed meta/modality_config.py's source
    # (rather than parsing it with `ast`), this call would raise
    # ModuleNotFoundError for `gr00t` — it does not.
    (result,) = check_modality_agreement(ctx)
    assert result.passed is True
    assert "gr00t" not in sys.modules
