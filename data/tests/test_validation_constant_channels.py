"""VALID-04/D-07/D-08 tests: dataset-wide constant-channel detection.

Most behaviours are driven by `data/tests/mutators.py::force_constant_channel`
against `build_varying_baseline_dataset` prefixes, matching plan 03-04's
own instruction. A few boundary/edge behaviours (the exact one-float32-ULP
probe, the two `likely_cause` classifications, and the all-excluded merge)
need finer control than any existing mutator provides and instead edit the
persisted `meta/stats_partials/episode_*.json` sidecar directly — the exact
file `load_merged_partials` reads, so this is white-box testing of the real
code path, not a bypass of it.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import build_varying_baseline_dataset, copy_dataset, force_constant_channel
from data.validation.constant_channels import (
    CHECK_NAME,
    PROFILE_KEY_FOR_FEATURE,
    check_constant_channel,
    channel_names_for_feature,
)
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY, ValidationRegistry
from data.validation.validate import validate_dataset

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


def _profile_path_without_declared_lag(tmp_path: Path, profile: EmbodimentProfile) -> Path:
    """Mirrors `test_validation_frame_integrity.py`'s helper of the same
    name: writes a temp copy of the shipped profile with
    `expected_action_state_lag` unset, so `consistency.action_state_lag`
    reports `UNEVALUATED` (per VALID-03 Option B) rather than genuinely
    failing on this fixture's unrelated-by-construction ramp signal. Needed
    here so the exit-code test below is a clean, single-check comparison
    of `statistics.constant_channel` alone, not incidentally of VALID-03."""
    dumped = profile.model_dump(mode="json")
    dumped.pop("expected_action_state_lag", None)
    path = tmp_path / "profile_no_declared_lag.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dumped, f)
    return path


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _partial_path(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / "meta" / "stats_partials" / f"episode_{episode_index:06d}.json"


def _load_partial_payload(dataset_dir: Path, episode_index: int) -> dict[str, Any]:
    with open(_partial_path(dataset_dir, episode_index), encoding="utf-8") as f:
        return json.load(f)


def _save_partial_payload(dataset_dir: Path, episode_index: int, payload: dict[str, Any]) -> None:
    path = _partial_path(dataset_dir, episode_index)
    tmp = path.parent / f".{path.name}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _force_partial_channel_constant(
    dataset_dir: Path, episode_index: int, *, feature: str, channel_index: int, value: float
) -> None:
    """Directly edits one episode's persisted partial so its `mean`/`min`/
    `max`/`sum`/every retained `values` row all read `value` at
    `channel_index` for `feature` — a finer-grained edit than
    `force_constant_channel` (which also rewrites the parquet, unneeded for
    a check that only reads `meta/stats_partials/`), used for the exact
    `likely_cause` boundary values (`0.0` vs a nonzero constant)."""
    payload = _load_partial_payload(dataset_dir, episode_index)
    entry = payload["features"][feature]
    count = entry["count"]
    entry["mean"][channel_index] = value
    entry["min"][channel_index] = value
    entry["max"][channel_index] = value
    entry["m2"][channel_index] = 0.0
    entry["sum"][channel_index] = value * count
    for row in entry["values"]:
        row[channel_index] = value
    _save_partial_payload(dataset_dir, episode_index, payload)


def _set_partial_min_max(
    dataset_dir: Path,
    episode_index: int,
    *,
    feature: str,
    channel_index: int,
    min_value: float,
    max_value: float,
) -> None:
    """Directly sets one channel's merged `min`/`max` in one episode's
    persisted partial, without touching `mean`/`values`/`m2` — used for the
    exact-equality boundary probe (Test 5), which needs `min` and `max` to
    differ by precisely one float32 unit in the last place, a value no
    natural dataset construction can guarantee."""
    payload = _load_partial_payload(dataset_dir, episode_index)
    entry = payload["features"][feature]
    entry["min"][channel_index] = min_value
    entry["max"][channel_index] = max_value
    _save_partial_payload(dataset_dir, episode_index, payload)


def _mark_all_partials_excluded(dataset_dir: Path) -> None:
    """Sets `included_in_merge: false` (and `disposition: "REJECT"`, for
    evidentiary consistency) on every episode's persisted partial —
    simulates "every episode in this prefix was REJECTed" without needing
    to fabricate a failing ingest-time check to actually earn that
    disposition; `load_merged_partials` only ever reads the flag, so this
    exercises its real branch."""
    partials_dir = dataset_dir / "meta" / "stats_partials"
    for path in sorted(partials_dir.glob("episode_*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        payload["included_in_merge"] = False
        payload["disposition"] = "REJECT"
        tmp = path.parent / f".{path.name}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Test 1: every channel varies -> passes, emitted width recorded.
# ---------------------------------------------------------------------------


def test_clean_varying_prefix_passes_with_measured_width(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_constant_channel(ctx)[0]

    assert result.passed is True
    assert result.evidence["flagged_channels"] == []
    expected_state_width = len(channel_names_for_feature(profile, "observation.state"))
    expected_action_width = len(channel_names_for_feature(profile, "action"))
    assert expected_state_width == 20
    assert expected_action_width == 16
    assert result.evidence["emitted_width_measured"] == expected_state_width
    assert result.evidence["action_emitted_width_measured"] == expected_action_width


# ---------------------------------------------------------------------------
# Test 2/3: force_constant_channel on a state channel and an action
# channel each fail, naming the profile key to edit.
# ---------------------------------------------------------------------------


def test_forced_constant_state_channel_fails_naming_state_vector_order(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6, episodes=1)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    force_constant_channel(mutated, feature="observation.state", channel_index=3)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert result.passed is False
    flagged = next(
        f for f in result.evidence["flagged_channels"] if f["feature"] == "observation.state"
    )
    assert flagged["channel_index"] == 3
    assert flagged["profile_key"] == "state_vector_order"
    assert flagged["profile_key"] == PROFILE_KEY_FOR_FEATURE["observation.state"]
    assert flagged["joint_name"] == channel_names_for_feature(profile, "observation.state")[3]
    assert flagged["merged_min"] == flagged["merged_max"]


def test_forced_constant_action_channel_fails_naming_action_primary(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6, episodes=1)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    force_constant_channel(mutated, feature="action", channel_index=2)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert result.passed is False
    flagged = next(f for f in result.evidence["flagged_channels"] if f["feature"] == "action")
    assert flagged["channel_index"] == 2
    assert flagged["profile_key"] == "action_primary"
    assert flagged["profile_key"] == PROFILE_KEY_FOR_FEATURE["action"]


# ---------------------------------------------------------------------------
# Test 4: constant in one episode, varying in another -> passes dataset-wide.
# ---------------------------------------------------------------------------


def test_constant_in_one_episode_varying_in_another_passes(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    force_constant_channel(dataset_dir, feature="observation.state", channel_index=1, episode_index=0)

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert result.passed is True
    assert not any(
        f["feature"] == "observation.state" and f["channel_index"] == 1
        for f in result.evidence["flagged_channels"]
    )


# ---------------------------------------------------------------------------
# Test 5: min/max differing by exactly one float32 ULP passes — exact
# equality, never a variance floor.
# ---------------------------------------------------------------------------


def test_one_float32_ulp_apart_passes(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    base = np.float32(1.0)
    next_up = np.nextafter(base, np.float32(np.inf))
    assert float(base) != float(next_up)  # sanity: distinct float64 widenings

    _set_partial_min_max(
        dataset_dir,
        0,
        feature="observation.state",
        channel_index=5,
        min_value=float(base),
        max_value=float(next_up),
    )

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert not any(
        f["feature"] == "observation.state" and f["channel_index"] == 5
        for f in result.evidence["flagged_channels"]
    )


# ---------------------------------------------------------------------------
# Test 6: failure evidence distinguishes the two causes; both fail.
# ---------------------------------------------------------------------------


def test_likely_cause_distinguishes_dead_sensor_from_intentionally_still(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    _force_partial_channel_constant(
        dataset_dir, 0, feature="observation.state", channel_index=0, value=0.0
    )
    _force_partial_channel_constant(
        dataset_dir, 0, feature="observation.state", channel_index=1, value=7.5
    )

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert result.passed is False
    flagged_by_index = {
        f["channel_index"]: f
        for f in result.evidence["flagged_channels"]
        if f["feature"] == "observation.state"
    }
    assert flagged_by_index[0]["likely_cause"] == "likely_dead_sensor"
    assert flagged_by_index[1]["likely_cause"] == "possibly_intentionally_still"
    # Both dispose as failures regardless of likely_cause.
    assert result.passed is False


# ---------------------------------------------------------------------------
# Test 7: every partial excluded from the merge -> unevaluated, not a
# zero-count division.
# ---------------------------------------------------------------------------


def test_all_partials_excluded_is_unevaluated(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=2
    )
    _mark_all_partials_excluded(dataset_dir)

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_constant_channel(ctx)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.passed is False
    assert SKIP_REASON_KEY in result.evidence


# ---------------------------------------------------------------------------
# Test 8: zero-width feature -> unevaluated, naming the zero width.
# ---------------------------------------------------------------------------


def test_zero_width_features_is_unevaluated(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    zero_profile = profile.model_copy(update={"state_vector_order": [], "action_primary": []})
    ctx = ValidationContext(out_dir=tmp_path / "empty", profile=zero_profile)

    result = check_constant_channel(ctx)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.passed is False
    assert "zero-width" in result.evidence[SKIP_REASON_KEY]


# ---------------------------------------------------------------------------
# Test 9 (G-03): severity is advisory, asserted directly rather than by
# reading the source.
# ---------------------------------------------------------------------------


def test_severity_is_advisory_on_every_emitted_result(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=6, episodes=1
    )
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    force_constant_channel(mutated, feature="observation.state", channel_index=4)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    flagged_result = check_constant_channel(ctx)[0]
    assert flagged_result.severity == "advisory"
    assert flagged_result.passed is False

    clean_ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    clean_result = check_constant_channel(clean_ctx)[0]
    assert clean_result.severity == "advisory"
    assert clean_result.passed is True

    entry = next(
        c for c in ValidationRegistry.default().checks if c.name == CHECK_NAME
    )
    assert entry.severity == "advisory"


# ---------------------------------------------------------------------------
# Test 10 (G-03): validate_dataset's exit code stays 0 when the constant-
# channel check is the only failing check -- the registry-level proof that
# an advisory failure alone never drives REJECT/non-zero exit.
# ---------------------------------------------------------------------------


def test_validate_dataset_exit_code_zero_when_only_constant_channel_fails(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    no_decl_profile_path = _profile_path_without_declared_lag(tmp_path, profile)
    no_decl_profile = load_profile(no_decl_profile_path)

    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", no_decl_profile, camera_frames=6
    )
    force_constant_channel(dataset_dir, feature="observation.state", channel_index=4)

    outcome = validate_dataset(dataset_dir, no_decl_profile)

    constant_channel_result = next(
        r for r in outcome.dataset_results if r.check_name == CHECK_NAME
    )
    assert constant_channel_result.passed is False
    assert constant_channel_result.severity == "advisory"
    assert outcome.exit_code == 0

    report_path = dataset_dir / "meta" / "data_engine" / "validation_report.json"
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["disposition"] != "REJECT"
