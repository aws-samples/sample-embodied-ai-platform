"""Tests for `data.tools.measure_action_state_pairing` — Defect B of
D-GAP-01 (VALID-03).

Two groups, per `03-11-PLAN.md` Task 1:

1. **Fresh-clone coverage** (runs with no `data/samples/` present): the
   structural-completeness, magnitude-direction, source-scan and
   camera-topic-absence tests, plus the two clock tests built on a
   synthetic fixture whose two clocks (`log_time`/`header_stamp`) are
   independently controlled by construction.
2. **Real-recording coverage** (skip-if-absent, mirroring
   `test_adapter_galaxea.py::test_real_sample_recording_if_present`'s
   D-18-accepted shape): asserts only the decision-independent facts —
   Finding 1's direction and the clock-skew report's structure. It does
   NOT assert a pass/fail verdict under any particular pass criterion —
   that is Task 3's, once the developer has decided one.
"""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.contracts.profile import load_profile
from data.tests.mcap_fixture import (
    SCHEMA_NAME_JOINT_STATE,
    SCHEMA_NAME_MOTOR_CONTROL,
    SCHEMA_TEXT_JOINT_STATE,
    SCHEMA_TEXT_MOTOR_CONTROL,
    TopicSpec,
    build_fixture_mcap,
    joint_state_messages,
    motor_control_messages,
    varying_joint_positions,
    varying_motion_joint_topics,
)
from data.tests.mutators import _CAMERA_TOPICS, copy_dataset, shift_action_column
from data.tools.inspect_mcap import align_series
from data.tools.measure_action_state_pairing import (
    CANDIDATE_PAIRINGS,
    CLOCK_SOURCES,
    _derive_pairs,
    _field_map,
    _stream_buffers,
    clock_skew_ns,
    measure_pairings,
)
from data.validation.context import ValidationContext
from data.validation.correlation import check_action_state_lag
from data.validation.registry import UNEVALUATED_KEY
from data.writers.lerobot_v21 import _CHUNK_INDEX, _DATA_PATH_TEMPLATE

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"
_MODULE_PATH = "data/tools/measure_action_state_pairing.py"
_STEP_NS = 40_000_000


def _profile():
    return load_profile(PROFILE_PATH)


def _real_files() -> list[str]:
    return sorted(glob.glob("data/samples/vr-teleop-20260904/*/*.mcap"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Runs the CLI as `python -m data.tools.measure_action_state_pairing`
    (never as a bare script path) — the module imports `data.*` siblings by
    package-qualified name, which only resolves when Python treats the
    repo root as the top-level package directory, exactly what `-m` does
    (and exactly what this plan's own `<verify>` commands invoke)."""
    return subprocess.run(
        [sys.executable, "-m", "data.tools.measure_action_state_pairing", *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Fresh-clone: source scan / topic scoping
# ---------------------------------------------------------------------------


def test_measure_action_state_pairing_imports_neither_cv2_nor_av() -> None:
    with open(_MODULE_PATH) as f:
        source = f.read()
    for forbidden in ("import cv2", "import av", "from cv2", "from av"):
        assert forbidden not in source


def test_measure_pairings_requests_no_camera_topic() -> None:
    profile = _profile()
    pairs = _derive_pairs(profile)
    requested = set(_field_map(pairs))
    assert requested.isdisjoint(_CAMERA_TOPICS)


# ---------------------------------------------------------------------------
# Fresh-clone: structural completeness + magnitude direction (synthetic)
# ---------------------------------------------------------------------------


def _build_varying_fixture(tmp_path, *, joint_samples: int = 40) -> str:
    fixture_path = tmp_path / "varying.mcap"
    topics: dict[str, TopicSpec | None] = dict(
        varying_motion_joint_topics(joint_samples=joint_samples)
    )
    build_fixture_mcap(
        str(fixture_path), camera_frames=joint_samples, joint_samples=joint_samples, topics=topics
    )
    return str(fixture_path)


def test_measure_pairings_synthetic_fixture_structurally_complete(tmp_path) -> None:
    profile = _profile()
    fixture_path = _build_varying_fixture(tmp_path)

    report = measure_pairings(fixture_path, profile, max_samples=200, decimate=1)

    assert report["clock"] == "log_time"
    action_keys = set(profile.action_topics)
    assert action_keys <= set(report["action_keys"])
    for key in action_keys:
        entry = report["action_keys"][key]
        assert "skip_reason" not in entry, entry
        assert entry["aligned_rows"] > 2
        assert set(entry["pairings"]) == set(CANDIDATE_PAIRINGS)
        assert "argmin_lag" in entry["tracking_error_profile"]
        assert entry["median_abs_state"] > 0
        assert entry["median_abs_action"] > 0


def test_median_abs_action_less_than_median_abs_state_on_synthetic_right_arm(tmp_path) -> None:
    """On `varying_motion_joint_topics`'s fixture, the action series IS the
    achieved delta of the paired state series (by construction) — its
    magnitude must be far below the state's own absolute magnitude, the
    mirror image of Finding 1 on real data."""
    profile = _profile()
    fixture_path = _build_varying_fixture(tmp_path)

    report = measure_pairings(fixture_path, profile, max_samples=200)
    entry = report["action_keys"]["right_arm"]
    assert entry["median_abs_action"] < entry["median_abs_state"]


# ---------------------------------------------------------------------------
# Fresh-clone: clock coverage (synthetic, both clocks controlled by construction)
# ---------------------------------------------------------------------------


def _build_clock_offset_fixture(tmp_path, *, offset_steps: int = 1, count: int = 30) -> str:
    """A fixture where `/hdas/feedback_arm_right`'s `header.stamp` equals its
    own `log_time` exactly (offset 0), while `/motion_control/control_arm_right`
    carries the SAME position values but its `header.stamp` is offset
    `-offset_steps * _STEP_NS` from its own `log_time` — i.e. `header_stamp
    = log_time - offset_steps * step`, matching the direction Finding 4/7
    hypothesize (the recorder's `log_time` trails behind the true publish
    time by some latency).

    `/motion_target/target_joint_state_arm_right` is written empty so
    `control_arm_right` is the resolved (only populated) action source —
    the same fallback shape every real recording exhibits for `left_arm`.
    """
    # mcap_ros2's own hardcoded builtin_interfaces/Time definition declares
    # `sec` as uint32 (not the int32 the real ROS2 spec and this fixture's
    # own schema text declare) — verified directly against
    # mcap_ros2._dynamic.TimeDefinition. A negative `sec` therefore fails to
    # encode regardless of what schema text is registered. `start_ns` is
    # chosen large enough that `start_ns + i*step - offset_steps*step` never
    # goes negative for any `i`/`offset_steps` this module uses.
    start_ns = 10 * _STEP_NS
    positions = varying_joint_positions(count=count, joint_count=7, seed=99, base=2.0)
    fixture_path = tmp_path / "clock_offset.mcap"
    topics: dict[str, TopicSpec | None] = {
        "/hdas/feedback_arm_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(
                count=count,
                positions=positions,
                joint_count=7,
                name="right_arm",
                start_ns=start_ns,
                step_ns=_STEP_NS,
                header_stamp_offset_ns=0,
            ),
        ),
        "/motion_target/target_joint_state_arm_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE, SCHEMA_TEXT_JOINT_STATE, []
        ),
        "/motion_control/control_arm_right": TopicSpec(
            SCHEMA_NAME_MOTOR_CONTROL,
            SCHEMA_TEXT_MOTOR_CONTROL,
            motor_control_messages(
                count=count,
                p_des=positions,
                joint_count=7,
                start_ns=start_ns,
                step_ns=_STEP_NS,
                header_stamp_offset_ns=-offset_steps * _STEP_NS,
            ),
        ),
    }
    build_fixture_mcap(str(fixture_path), camera_frames=3, joint_samples=3, topics=topics)
    return str(fixture_path)


def test_clock_skew_ns_recovers_injected_offset_exactly(tmp_path) -> None:
    fixture_path = _build_clock_offset_fixture(tmp_path, offset_steps=1)
    profile = _profile()

    report = clock_skew_ns(fixture_path, profile, max_samples=200)

    action_row = report["topics"]["/motion_control/control_arm_right"]
    assert action_row["median_ns"] == pytest.approx(float(_STEP_NS), abs=1e-6)
    assert action_row["absent_or_zero_stamp_count"] == 0

    state_row = report["topics"]["/hdas/feedback_arm_right"]
    assert state_row["median_ns"] == pytest.approx(0.0, abs=1e-6)
    assert report["grid_period_ns"] > 0
    assert action_row["median_grid_periods"] == pytest.approx(
        _STEP_NS / report["grid_period_ns"], abs=1e-9
    )


def test_measure_pairings_header_stamp_clock_shifts_argmin_lag_by_exactly_one(tmp_path) -> None:
    fixture_path = _build_clock_offset_fixture(tmp_path, offset_steps=1)
    profile = _profile()

    log_time_report = measure_pairings(
        fixture_path, profile, max_samples=200, clock="log_time"
    )
    header_report = measure_pairings(
        fixture_path, profile, max_samples=200, clock="header_stamp"
    )

    assert log_time_report["clock"] == "log_time"
    assert header_report["clock"] == "header_stamp"

    log_entry = log_time_report["action_keys"]["right_arm"]
    header_entry = header_report["action_keys"]["right_arm"]
    assert "skip_reason" not in log_entry, log_entry
    assert "skip_reason" not in header_entry, header_entry

    log_argmin = log_entry["tracking_error_profile"]["argmin_lag"]
    header_argmin = header_entry["tracking_error_profile"]["argmin_lag"]

    assert log_argmin is not None and header_argmin is not None
    assert header_argmin - log_argmin == -1


# ---------------------------------------------------------------------------
# Real-recording coverage (skip-if-absent, D-18 shape) — decision-independent only
# ---------------------------------------------------------------------------


def test_real_recording_right_arm_action_is_absolute_position_not_delta() -> None:
    """The one finding already settled (Finding 1): the executable form of
    "action is an absolute position target, not a delta" — asserts only
    this direction, never a pass/fail verdict under any pass criterion."""
    real_files = _real_files()
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    profile = _profile()
    report = measure_pairings(
        real_files[0], profile, max_samples=1200, skip_samples=700, decimate=1
    )
    entry = report["action_keys"]["right_arm"]
    assert "skip_reason" not in entry, entry
    assert entry["aligned_rows"] > 2

    assert entry["median_abs_action"] <= 2.0 * entry["median_abs_state"]
    assert entry["median_abs_action"] >= 0.5 * entry["median_abs_state"]
    assert entry["median_abs_state_delta"] * 100.0 <= entry["median_abs_state"]


def test_real_recording_clock_skew_structural_report() -> None:
    """Presence and structure only — Finding 7's magnitude/direction is the
    open empirical question this measurement exists to answer, so this test
    deliberately does not pin it either way. An absent header.stamp on the
    real topics is itself a valid (inconclusive) Finding 7 answer, not a
    test failure."""
    real_files = _real_files()
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    profile = _profile()
    report = clock_skew_ns(real_files[0], profile, max_samples=1200, skip_samples=700)

    row = report["topics"]["/hdas/feedback_arm_right"]
    assert isinstance(row["sample_count"], int) and row["sample_count"] > 0
    assert isinstance(row["absent_or_zero_stamp_count"], int)
    assert row["median_ns"] is None or isinstance(row["median_ns"], float)
    assert row["median_grid_periods"] is None or isinstance(row["median_grid_periods"], float)


# ---------------------------------------------------------------------------
# Plan 03-11 Task 3: the criterion-specific real-recording assertion Task 1
# deliberately deferred until Task 2 decided the pass criterion (Option B —
# declared expected_action_state_lag, compared against tracking_error_lag).
# Builds observation.state/action parquet columns directly from REAL
# streamed right_arm data (no camera decode, no video encode — bypassing
# the full writer entirely, matching this module's own zero-decode-cost
# design) so this stays fast even against the 322MB real recordings.
# ---------------------------------------------------------------------------


def _real_right_arm_aligned_series(
    path: str, profile, *, max_samples: int = 1200, skip_samples: int = 700, decimate: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Real right_arm `(state, aligned_action)` arrays at the persisted
    ~30Hz grid — the same internal streaming/alignment
    `measure_pairings(..., decimate=5)` uses, reimplemented at the array
    level (rather than the aggregate-statistics level `measure_pairings`
    returns) so a real parquet file can be built from it."""
    pairs = _derive_pairs(profile)
    pair = pairs["right_arm"]
    assert pair.state_field is not None and pair.action_field is not None
    field_by_topic = _field_map({"right_arm": pair})
    buffers = _stream_buffers(
        path, field_by_topic, max_samples=max_samples, skip_samples=skip_samples, clock="log_time"
    )

    state_buf = buffers[pair.state_field.topic][::decimate]
    winning_source = next(
        source for source in pair.action_field.sources if len(buffers.get(source.topic, [])) > 0
    )
    action_buf = buffers[winning_source.topic]

    state_ts = np.array([t for t, _ in state_buf], dtype=np.float64)
    state_vals = np.stack([v for _, v in state_buf])
    action_ts = np.array([t for t, _ in action_buf], dtype=np.float64)
    action_vals = np.stack([v for _, v in action_buf])

    aligned_action = align_series(state_ts, state_vals, action_ts, action_vals)
    return state_vals.astype(np.float32), aligned_action.astype(np.float32)


def _write_full_width_parquet_from_right_arm(
    dataset_dir: Path, profile, *, state_right_arm: np.ndarray, action_right_arm: np.ndarray
) -> Path:
    """Builds `observation.state`(20-wide)/`action`(16-wide) parquet
    columns in the profile's own `state_vector_order`/`action_primary`
    layout, placing REAL right_arm data in the right_arm slice and zeros
    everywhere else — a faithful reduced-fidelity reconstruction, since
    Finding 5 (`03-11-PLAN.md`) already measures `left_arm` and both
    grippers as exactly constant-zero in every real recording, and
    `torso` is not action-aligned so its content never affects
    `check_action_state_lag`'s verdict."""
    row_count = state_right_arm.shape[0]
    assert action_right_arm.shape[0] == row_count

    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    state_widths = [
        (key, len(by_state_key[key].joint_names)) for key in profile.state_vector_order
    ]
    state_total = sum(w for _, w in state_widths)
    by_action_key_field = {f.action_key: f for f in profile.action_topics.values()}
    action_widths = [
        (key, len(by_action_key_field[key].joint_names)) for key in profile.action_primary
    ]
    action_total = sum(w for _, w in action_widths)

    state_matrix = np.zeros((row_count, state_total), dtype=np.float32)
    offset = 0
    for key, width in state_widths:
        if key == "right_arm":
            state_matrix[:, offset : offset + width] = state_right_arm
        offset += width

    action_matrix = np.zeros((row_count, action_total), dtype=np.float32)
    offset = 0
    for key, width in action_widths:
        if key == "right_arm":
            action_matrix[:, offset : offset + width] = action_right_arm
        offset += width

    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), state_total)),
            pa.field("action", pa.list_(pa.float32(), action_total)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "observation.state": pa.array(
                state_matrix.tolist(), type=schema.field("observation.state").type
            ),
            "action": pa.array(action_matrix.tolist(), type=schema.field("action").type),
            "timestamp": pa.array([float(i) for i in range(row_count)], type=pa.float32()),
            "frame_index": pa.array(list(range(row_count)), type=pa.int64()),
            "episode_index": pa.array([0] * row_count, type=pa.int64()),
            "index": pa.array(list(range(row_count)), type=pa.int64()),
            "task_index": pa.array([0] * row_count, type=pa.int64()),
        },
        schema=schema,
    )
    path = dataset_dir / _DATA_PATH_TEMPLATE.format(episode_chunk=_CHUNK_INDEX, episode_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return dataset_dir


def test_real_recording_leading_lag_criterion_discriminates(tmp_path) -> None:
    """Under the decided Option B criterion
    (`data/validation/correlation.py::check_action_state_lag`, comparing
    `tracking_error_lag` against `profile.expected_action_state_lag ==
    -4`), a real unshifted recording produces the verdict the criterion
    specifies, and a deliberate one-tick rotation of the action series
    produces a genuinely different verdict — the assertion Task 1
    deliberately deferred until Task 2 decided the criterion."""
    real_files = _real_files()
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    profile = _profile()
    assert profile.expected_action_state_lag == -4

    state_right_arm, action_right_arm = _real_right_arm_aligned_series(real_files[0], profile)

    clean_dir = _write_full_width_parquet_from_right_arm(
        (Path(tmp_path) / "clean_real"),
        profile,
        state_right_arm=state_right_arm,
        action_right_arm=action_right_arm,
    )
    mutated_dir = copy_dataset(clean_dir, Path(tmp_path) / "mutated_real")
    shift_action_column(mutated_dir, episode_index=0, shift=1)

    clean_ctx = ValidationContext(out_dir=clean_dir, profile=profile)
    mutated_ctx = ValidationContext(out_dir=mutated_dir, profile=profile)

    clean_result = check_action_state_lag(clean_ctx, 0)[0]
    shifted_result = check_action_state_lag(mutated_ctx, 0)[0]

    assert UNEVALUATED_KEY not in clean_result.evidence, clean_result.evidence
    assert UNEVALUATED_KEY not in shifted_result.evidence, shifted_result.evidence
    assert clean_result.evidence["expected_lag"] == -4
    assert clean_result.evidence["measured_lag"] != shifted_result.evidence["measured_lag"]
    # The verdict the decided criterion specifies for unshifted real data:
    # passed iff the measured lag matches the declared -4 exactly.
    assert clean_result.passed is (clean_result.evidence["measured_lag"] == -4)
    # A one-tick rotation must not measure the SAME lag as clean — that
    # would fail to discriminate the exact defect VALID-03 exists to catch.
    assert shifted_result.evidence["measured_lag"] != clean_result.evidence["measured_lag"]


# ---------------------------------------------------------------------------
# CLI contracts
# ---------------------------------------------------------------------------


def test_cli_help_lists_all_seven_flags() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    for flag in (
        "--profile",
        "--max-samples",
        "--skip-samples",
        "--decimate",
        "--clock",
        "--clock-skew",
        "--json",
    ):
        assert flag in result.stdout


def test_cli_decimate_and_json_on_real_recording_reports_peak_lag_for_every_pairing() -> None:
    real_files = _real_files()
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    result = _run_cli(real_files[0], "--decimate", "5", "--skip-samples", "700", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    entry = payload[0]["action_keys"]["right_arm"]
    for name in CANDIDATE_PAIRINGS:
        assert "peak_lag" in entry["pairings"][name]


def test_cli_clock_skew_json_on_real_recording() -> None:
    real_files = _real_files()
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    result = _run_cli(real_files[0], "--clock-skew", "--skip-samples", "700", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    row = payload[0]["topics"]["/hdas/feedback_arm_right"]
    assert "median_ns" in row
    assert "median_grid_periods" in row
    assert isinstance(row["absent_or_zero_stamp_count"], int)


def test_clock_sources_are_exactly_log_time_and_header_stamp() -> None:
    assert list(CLOCK_SOURCES) == ["log_time", "header_stamp"]
