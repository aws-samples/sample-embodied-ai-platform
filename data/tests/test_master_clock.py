"""Tests for `MasterClockSynchronizer`'s full per-field resample dispatch,
secondary-camera nearest-match with a gap bound, and the populated
`ClockReport` (02-04, SYNC-01..05 / PROV-03).

Every test builds `raw_streams` directly — the exact plain-dict shape
`data/adapters/galaxea.py`'s decode pass produces (see that module's
`_run_decode_pass` docstring) — rather than routing through the full
`GalaxeaInputAdapter`. This decouples these tests from the caller (per this
plan's own note that it changes only the callee) and lets each test isolate
one behavior with an exact, hand-picked timestamp/value fixture instead of
inheriting `mcap_fixture.py`'s defaults.
"""

import json
import re
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from data.contracts.profile import EmbodimentProfile
from data.sync.master_clock import (
    MasterClockSynchronizer,
    MasterClockSynchronizerError,
    _resample_series,
)

STEP_NS = 40_000_000  # 25 Hz, matching the shipped profile's measured native rate


def _profile(
    *,
    target_fps: float | None = None,
    chassis_required: bool = False,
    gripper_action_required: bool = True,
    all_optional: bool = False,
) -> EmbodimentProfile:
    """A minimal, fully self-contained profile — one arm (interp), one
    gripper (zoh), an excluded-from-emission chassis field, one secondary
    camera (nearest, 20ms gap bound), and one ordered-fallback action field
    — small enough for exact closed-form assertions, independent of the
    shipped Galaxea profile (which stays reserved for the adapter's own
    tests)."""
    arm_required = not all_optional
    gripper_required = not all_optional
    action_arm_required = not all_optional
    action_gripper_required = False if all_optional else gripper_action_required
    return EmbodimentProfile.model_validate(
        {
            "embodiment": "test_rig",
            "schema_version": 1,
            "robot_type": "test_rig",
            "master_clock": {"topic": "/cam/master", "target_fps": target_fps},
            "cameras": {
                "cam_master": {
                    "topic": "/cam/master",
                    "role": "master_clock",
                    "required": True,
                    "emit": True,
                    "video_key": "observation.images.master",
                },
                "cam_secondary": {
                    "topic": "/cam/secondary",
                    "role": "secondary",
                    # required/emit both False (04-04-PLAN.md, G-01/VERIFY-01):
                    # this profile's own master_clock declares no
                    # max_camera_coverage_deficit_ticks (defaults to 0), so a
                    # constraining (required or emit) camera with ANY gap
                    # would now truncate the grid at the first miss. This
                    # fixture's test exercises the per-tick nearest-match/
                    # gap-bound annotation in isolation, not the coverage
                    # trim (which has its own dedicated coverage in
                    # data/tests/test_sync_camera_coverage.py) -- so
                    # cam_secondary is deliberately non-constraining here.
                    "required": False,
                    "emit": False,
                    "video_key": "observation.images.secondary",
                    "sync_policy": "nearest",
                    "max_sync_gap_ms": 20,
                },
            },
            "state_topics": {
                "arm": {
                    "topic": "/state/arm",
                    "field": "position",
                    "joint_names": ["j1", "j2"],
                    "required": arm_required,
                    "resample": "interp",
                    "state_key": "arm",
                    "include_in_observation_state": True,
                },
                "gripper": {
                    "topic": "/state/gripper",
                    "field": "position",
                    "joint_names": ["g1"],
                    "required": gripper_required,
                    "resample": "zoh",
                    "state_key": "gripper",
                    "include_in_observation_state": True,
                },
                "chassis": {
                    "topic": "/state/chassis",
                    "field": "position",
                    "joint_names": ["x", "y"],
                    "required": chassis_required,
                    "resample": "interp",
                    "state_key": "chassis",
                    "include_in_observation_state": False,
                },
            },
            "action_topics": {
                "arm": {
                    "sources": [
                        {"topic": "/action/arm", "field": "position", "message_type": "x"},
                    ],
                    "joint_names": ["j1", "j2"],
                    "required": action_arm_required,
                    "resample": "interp",
                    "action_key": "arm",
                },
                "gripper_action": {
                    "sources": [
                        {
                            "topic": "/action/gripper_pref",
                            "field": "position",
                            "message_type": "x",
                        },
                        {
                            "topic": "/action/gripper_fallback",
                            "field": "p_des",
                            "message_type": "y",
                        },
                    ],
                    "joint_names": ["g1"],
                    "required": action_gripper_required,
                    "resample": "zoh",
                    "action_key": "gripper_action",
                },
            },
            "state_vector_order": ["arm", "gripper"],
            "action_primary": ["arm", "gripper_action"],
        }
    )


def _camera_entry(
    topic: str, role: str, video_key: str, frame_ts: list[int], *, prefix: str = "frame"
) -> dict[str, Any]:
    return {
        "topic": topic,
        "role": role,
        "video_key": video_key,
        "frames": [(ts, f"{prefix}_{i}") for i, ts in enumerate(frame_ts)],
    }


def _series_entry(
    topic: str, resample: str, joint_names: list[str], samples: list[tuple[int, list[float]]]
) -> dict[str, Any]:
    return {
        "topic": topic,
        "resample": resample,
        "joint_names": joint_names,
        "samples": [(ts, np.array(v, dtype=np.float32)) for ts, v in samples],
    }


def _raw_streams(
    *,
    master_frame_ts: list[int],
    secondary_frame_ts: list[int] | None = None,
    arm_samples: list[tuple[int, list[float]]] | None = None,
    gripper_samples: list[tuple[int, list[float]]] | None = None,
    chassis_samples: list[tuple[int, list[float]]] | None = None,
    action_arm_samples: list[tuple[int, list[float]]] | None = None,
    action_gripper_topic: str = "/action/gripper_pref",
    action_gripper_samples: list[tuple[int, list[float]]] | None = None,
    include_chassis: bool = True,
    include_gripper_action: bool = True,
) -> dict[str, Any]:
    cameras = {
        "cam_master": _camera_entry(
            "/cam/master", "master_clock", "observation.images.master", master_frame_ts,
            prefix="master",
        ),
    }
    if secondary_frame_ts is not None:
        cameras["cam_secondary"] = _camera_entry(
            "/cam/secondary", "secondary", "observation.images.secondary", secondary_frame_ts,
            prefix="secondary",
        )

    state: dict[str, Any] = {
        "arm": {
            **_series_entry("/state/arm", "interp", ["j1", "j2"], arm_samples or []),
            "state_key": "arm",
        },
        "gripper": {
            **_series_entry("/state/gripper", "zoh", ["g1"], gripper_samples or []),
            "state_key": "gripper",
        },
    }
    if include_chassis:
        state["chassis"] = {
            **_series_entry("/state/chassis", "interp", ["x", "y"], chassis_samples or []),
            "state_key": "chassis",
        }

    action: dict[str, Any] = {
        "arm": _series_entry("/action/arm", "interp", ["j1", "j2"], action_arm_samples or []),
    }
    if include_gripper_action:
        action["gripper_action"] = _series_entry(
            action_gripper_topic, "zoh", ["g1"], action_gripper_samples or []
        )

    return {"cameras": cameras, "state": state, "action": action, "meta": {}}


# ---------------------------------------------------------------------------
# Task 1: grid construction, per-field dispatch, idealized timestamps.
# ---------------------------------------------------------------------------


def test_rate_detected_from_master_camera_when_target_fps_unset() -> None:
    profile = _profile(target_fps=None)
    ts = [i * STEP_NS for i in range(5)]
    raw = _raw_streams(
        master_frame_ts=ts,
        arm_samples=[(t, [1.0, 2.0]) for t in ts],
        gripper_samples=[(t, [0.0]) for t in ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in ts],
        action_gripper_samples=[(t, [0.0]) for t in ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    assert episode.header.fps == pytest.approx(25.0, rel=1e-9)


def test_target_fps_used_verbatim_ignoring_detected_rate() -> None:
    profile = _profile(target_fps=30.0)
    ts = [i * STEP_NS for i in range(3)]  # would detect 25Hz if measured
    raw = _raw_streams(
        master_frame_ts=ts,
        arm_samples=[(t, [1.0, 2.0]) for t in ts],
        gripper_samples=[(t, [0.0]) for t in ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in ts],
        action_gripper_samples=[(t, [0.0]) for t in ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    assert episode.header.fps == pytest.approx(30.0, rel=1e-9)


def test_one_frame_master_clock_aborts_naming_topic_when_target_fps_unset() -> None:
    profile = _profile(target_fps=None)
    raw = _raw_streams(
        master_frame_ts=[0],
        arm_samples=[(0, [1.0, 2.0])],
        gripper_samples=[(0, [0.0])],
        action_arm_samples=[(0, [1.0, 2.0])],
        action_gripper_samples=[(0, [0.0])],
    )
    with pytest.raises(MasterClockSynchronizerError) as excinfo:
        MasterClockSynchronizer().synchronize(raw, profile)
    message = str(excinfo.value)
    assert "/cam/master" in message
    assert "rate cannot be detected" in message


def test_interp_field_matches_closed_form_on_multi_joint_different_slopes() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(4)]
    sample_ts = [i * 20_000_000 for i in range(8)]  # denser than the grid, covers its span
    arm_samples = [(t, [t / 1e9, -2.0 * t / 1e9]) for t in sample_ts]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=arm_samples,
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=arm_samples,
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    for f in episode.frames:
        t_s = f.timestamp_ns / 1e9
        expected = np.array([t_s, -2.0 * t_s], dtype=np.float32)
        np.testing.assert_allclose(f.state["arm"], expected, atol=1e-5)
        np.testing.assert_allclose(f.action["arm"], expected, atol=1e-5)


def test_interp_dispatch_is_one_vectorized_call_never_per_joint() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    arm_samples = [(t, [1.0, 2.0]) for t in [i * 20_000_000 for i in range(5)]]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=arm_samples,
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=arm_samples,
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    with patch("data.sync.master_clock.make_interp_spline") as spy:
        spy.return_value = lambda x: np.zeros((len(x), 2), dtype=np.float64)
        MasterClockSynchronizer().synchronize(raw, profile)
    # Exactly one call per interp field (state.arm, action.arm) — never one
    # call per joint, which would be 4 (2 joints x 2 fields) instead of 2.
    assert spy.call_count == 2


def test_zoh_gripper_never_interpolates_only_exact_discrete_values() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * 20_000_000 for i in range(5)]  # 0,20,40,60,80 ms
    gripper_samples = [(0, [0.0]), (50_000_000, [1.0])]  # steps mid-grid
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=gripper_samples,
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=gripper_samples,
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    values = {float(f.state["gripper"][0]) for f in episode.frames}
    assert values <= {0.0, 1.0}
    assert values == {0.0, 1.0}


def test_boundary_ticks_outside_source_span_resolve_without_nan() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(5)]
    # Source samples only cover the middle of the grid's span.
    arm_samples = [(60_000_000, [1.0, 2.0]), (100_000_000, [3.0, 4.0])]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=arm_samples,
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=arm_samples,
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    for f in episode.frames:
        assert not np.any(np.isnan(f.state["arm"]))
        assert not np.any(np.isnan(f.action["arm"]))


def test_emitted_timestamps_derive_from_grid_position_never_capture_time() -> None:
    profile = _profile(target_fps=None)
    ideal = [i * STEP_NS for i in range(5)]
    offsets = [2_000_000, 3_000_000, -2_000_000, 1_000_000, -1_000_000]
    jittered = [t + o for t, o in zip(ideal, offsets)]
    raw = _raw_streams(
        master_frame_ts=jittered,
        arm_samples=[(t, [1.0, 2.0]) for t in jittered],
        gripper_samples=[(t, [0.0]) for t in jittered],
        action_arm_samples=[(t, [1.0, 2.0]) for t in jittered],
        action_gripper_samples=[(t, [0.0]) for t in jittered],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frames = list(episode.frames)
    assert len(frames) == 5
    for i, f in enumerate(frames):
        assert f.timestamp_ns != jittered[i]
        assert f.frame_index == i


def test_secondary_camera_attached_inside_gap_bound_omitted_outside() -> None:
    profile = _profile(target_fps=None)  # cam_secondary declares a 20ms gap bound
    grid_ts = [0, 40_000_000, 80_000_000, 120_000_000]
    secondary_ts = [5_000_000, 105_000_000]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        secondary_frame_ts=secondary_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frames = list(episode.frames)
    key = "observation.images.secondary"

    assert key in frames[0].images  # nearest is 5ms away — inside 20ms bound
    assert key not in frames[1].images  # nearest is 35ms away — outside
    assert key not in frames[2].images  # nearest is 25ms away — outside
    assert key in frames[3].images  # nearest is 15ms away — inside

    assert frames[1].meta["secondary_camera_gaps_ns"]["cam_secondary"] == 35_000_000
    assert frames[2].meta["secondary_camera_gaps_ns"]["cam_secondary"] == 25_000_000
    assert "secondary_camera_gaps_ns" not in frames[0].meta
    assert "secondary_camera_gaps_ns" not in frames[3].meta


def test_excluded_field_resampled_and_reported_but_absent_from_emitted_state() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        chassis_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frame = next(iter(episode.frames))
    assert "chassis" not in frame.state
    report = episode.header.global_meta["clock_report"]
    assert "state.chassis" in report.per_field_source_age_ns


def test_optional_field_zero_samples_absent_not_zero_filled() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        chassis_samples=[],  # present entry, zero samples
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frame = next(iter(episode.frames))
    assert "chassis" not in frame.state
    report = episode.header.global_meta["clock_report"]
    assert "state.chassis" not in report.per_field_source_age_ns


def test_frames_generator_is_single_pass_and_matches_grid_length() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(4)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frames = list(episode.frames)
    assert len(frames) == 4
    assert list(episode.frames) == []


# ---------------------------------------------------------------------------
# Task 2: the populated ClockReport.
# ---------------------------------------------------------------------------


def test_clock_report_grid_hz_and_single_tick_episode() -> None:
    profile = _profile(target_fps=25.0)  # a declared rate makes 1 frame legal
    raw = _raw_streams(
        master_frame_ts=[0],
        arm_samples=[(0, [1.0, 2.0])],
        gripper_samples=[(0, [0.0])],
        action_arm_samples=[(0, [1.0, 2.0])],
        action_gripper_samples=[(0, [0.0])],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.grid_hz == pytest.approx(25.0, rel=1e-9)
    assert len(list(episode.frames)) == 1


def test_per_field_source_age_covers_every_resampled_field_sorted_int_keys() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        chassis_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    ages = episode.header.global_meta["clock_report"].per_field_source_age_ns
    assert set(ages) == {"state.arm", "state.gripper", "state.chassis", "action.arm", "action.gripper_action"}
    assert list(ages) == sorted(ages)
    for v in ages.values():
        assert isinstance(v, int)


def test_source_age_zero_when_sample_exactly_on_tick() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.per_field_source_age_ns["state.gripper"] == 0


def test_source_age_measures_exact_12ms_prior_sample_for_zoh() -> None:
    profile = _profile(target_fps=25.0)  # single tick, so the mean is exact
    raw = _raw_streams(
        master_frame_ts=[0],
        arm_samples=[(0, [1.0, 2.0])],
        gripper_samples=[(-12_000_000, [5.0])],
        action_arm_samples=[(0, [1.0, 2.0])],
        action_gripper_samples=[(0, [0.0])],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.per_field_source_age_ns["state.gripper"] == 12_000_000


def test_resolved_action_source_names_winning_topic_when_preferred_populated() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [0, STEP_NS]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_topic="/action/gripper_pref",
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.resolved_action_source["arm"] == "/action/arm"
    assert report.resolved_action_source["gripper_action"] == "/action/gripper_pref"


def test_resolved_action_source_names_fallback_topic_when_that_is_the_winner() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [0, STEP_NS]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_topic="/action/gripper_fallback",
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.resolved_action_source["gripper_action"] == "/action/gripper_fallback"


def test_optional_action_field_no_source_records_empty_string_no_segment() -> None:
    profile = _profile(target_fps=None, gripper_action_required=False)
    grid_ts = [0, STEP_NS]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        include_gripper_action=False,
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.resolved_action_source["gripper_action"] == ""
    frame = next(iter(episode.frames))
    assert "gripper_action" not in frame.action


def test_required_action_field_no_source_aborts_naming_field_and_every_source() -> None:
    profile = _profile(target_fps=None, gripper_action_required=True)
    grid_ts = [0, STEP_NS]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        include_gripper_action=False,
    )
    with pytest.raises(MasterClockSynchronizerError) as excinfo:
        MasterClockSynchronizer().synchronize(raw, profile)
    message = str(excinfo.value)
    assert "gripper_action" in message
    assert "/action/gripper_pref" in message
    assert "/action/gripper_fallback" in message


def test_skew_is_int_and_a_deliberately_large_skew_still_completes() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(-500_000_000, [0.0])],  # 500ms before everything
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert isinstance(report.max_skew_ns, int)
    assert isinstance(report.mean_skew_ns, int)
    assert report.max_skew_ns >= 500_000_000
    assert len(list(episode.frames)) == 3
    assert episode.header.global_meta["skew_rounding_mode"] == "floor"


def test_equal_timestamps_record_zero_skew() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.max_skew_ns == 0
    assert report.mean_skew_ns == 0
    assert episode.header.global_meta["single_stream"] is False


def test_single_stream_episode_records_zero_skew_with_explicit_marker() -> None:
    profile = _profile(target_fps=None, all_optional=True)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(master_frame_ts=grid_ts, include_chassis=False, include_gripper_action=False)
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    report = episode.header.global_meta["clock_report"]
    assert report.max_skew_ns == 0
    assert report.mean_skew_ns == 0
    assert episode.header.global_meta["single_stream"] is True


def test_resample_mode_recorded_alongside_source_age_in_global_meta() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    mode_map = episode.header.global_meta["per_field_resample_mode"]
    assert mode_map["state.arm"] == "interp"
    assert mode_map["state.gripper"] == "zoh"
    assert mode_map["action.arm"] == "interp"
    assert mode_map["action.gripper_action"] == "zoh"


def test_clock_report_byte_identical_across_two_synchronize_calls() -> None:
    profile = _profile(target_fps=None)
    grid_ts = [i * STEP_NS for i in range(3)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        chassis_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    report_1 = MasterClockSynchronizer().synchronize(raw, profile).header.global_meta["clock_report"]
    report_2 = MasterClockSynchronizer().synchronize(raw, profile).header.global_meta["clock_report"]
    dumped_1 = json.dumps(asdict(report_1), sort_keys=True)
    dumped_2 = json.dumps(asdict(report_2), sort_keys=True)
    assert dumped_1 == dumped_2


# ---------------------------------------------------------------------------
# 03-07 gap closure (CR-01): `_resample_series`'s `interp` path must convert
# to float64 only relative to a local reference point, never as an absolute
# Unix-epoch nanosecond value. `float64`'s ULP at real 2026 epoch magnitude
# (~1.7e18 ns) is ~256 ns, so any fixture built from small timestamps (like
# every fixture above, all built from STEP_NS starting at t=0) can never
# exercise this bug — these tests deliberately use real epoch magnitude.
# ---------------------------------------------------------------------------

_EPOCH_NS = 1_788_000_000_000_000_000  # real 2026-era Unix-epoch nanoseconds


def test_interp_epoch_magnitude_sub_ulp_spacing_does_not_raise() -> None:
    """Six samples 200ns apart at epoch magnitude: pre-fix code raises
    ValueError('Expect x to not have duplicates') because 200ns is well
    under float64's ~256ns ULP at this magnitude, so distinct int64
    timestamps collapse to the identical float64 knot value."""
    ts = np.array([_EPOCH_NS + i * 200 for i in range(6)], dtype=np.int64)
    values = np.cumsum(np.ones((6, 3), dtype=np.float32), axis=0).astype(np.float32)
    result = _resample_series(ts, values, ts, "interp")
    assert result.shape == (6, 3)


def test_interp_epoch_magnitude_exact_at_matching_grid_ticks() -> None:
    """Mismatched-reference guard: grid ticks exactly equal to the sample
    timestamps must return the exact sample values (zero error). Making the
    knots reference-relative while leaving the evaluation points absolute
    does not raise, but silently returns values wrong by ~1e16 — this test
    is what catches that half-fix, not merely absence of an exception."""
    ts = np.array([_EPOCH_NS + i * 200 for i in range(6)], dtype=np.int64)
    values = np.cumsum(np.ones((6, 3), dtype=np.float32), axis=0).astype(np.float32)
    result = _resample_series(ts, values, ts, "interp")
    assert np.abs(result - values).max() == 0.0


def test_interp_epoch_magnitude_clips_to_endpoints_not_extrapolate() -> None:
    """A grid tick before the first sample and one after the last both clamp
    to the endpoint sample values, at epoch magnitude — clip-not-extrapolate
    behavior must survive the reference-relative rewrite unchanged."""
    d = 33_333_333  # ~30 Hz spacing
    ts = np.array([_EPOCH_NS + i * d for i in range(4)], dtype=np.int64)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    grid = np.array([_EPOCH_NS - d, _EPOCH_NS + 3 * d + d], dtype=np.int64)
    result = _resample_series(ts, values, grid, "interp")
    np.testing.assert_array_equal(result[0], values[0])
    np.testing.assert_array_equal(result[1], values[-1])


def test_interp_epoch_magnitude_30hz_midpoint_matches_arithmetic_mean() -> None:
    """No regression at realistic camera rates: samples 33ms (~30Hz) apart at
    epoch magnitude interpolate at a midpoint tick to the arithmetic mean of
    the bracketing samples, within float32 tolerance."""
    d = 33_333_333
    ts = np.array([_EPOCH_NS + i * d for i in range(4)], dtype=np.int64)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    grid = np.array([_EPOCH_NS + d // 2], dtype=np.int64)
    result = _resample_series(ts, values, grid, "interp")
    expected = (values[0] + values[1]) / 2
    np.testing.assert_allclose(result[0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 03-07 Task 2: no other timestamp path in the module casts an absolute
# epoch value to float, and the fix leaves synchronize()'s emitted output
# unchanged from a caller's perspective.
# ---------------------------------------------------------------------------


def test_synchronize_epoch_magnitude_end_to_end_emits_grid_relative_timestamps() -> None:
    """A full synchronize() call at real epoch magnitude, with both an
    interp-mode state field and a zoh-mode state field, must complete and
    emit CanonicalFrame.timestamp_ns values that are grid-relative (start at
    0, step by the grid period) — never carrying the epoch offset — with a
    non-negative integer ClockReport skew."""
    profile = _profile(target_fps=None)
    grid_ts = [_EPOCH_NS + i * STEP_NS for i in range(4)]
    raw = _raw_streams(
        master_frame_ts=grid_ts,
        arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        gripper_samples=[(t, [0.0]) for t in grid_ts],
        action_arm_samples=[(t, [1.0, 2.0]) for t in grid_ts],
        action_gripper_samples=[(t, [0.0]) for t in grid_ts],
    )
    episode = MasterClockSynchronizer().synchronize(raw, profile)
    frames = list(episode.frames)
    assert len(frames) == 4
    for i, f in enumerate(frames):
        assert f.timestamp_ns == pytest.approx(i * STEP_NS, abs=1)
        assert f.timestamp_ns < _EPOCH_NS  # never leaks the absolute offset
    report = episode.header.global_meta["clock_report"]
    assert isinstance(report.max_skew_ns, int) and report.max_skew_ns >= 0
    assert isinstance(report.mean_skew_ns, int) and report.mean_skew_ns >= 0


def test_interp_epoch_magnitude_matches_hand_computed_linear_interpolation() -> None:
    """Characterization test pinning that the fix is behaviour-preserving at
    ordinary rates: at 30Hz-equivalent epoch-magnitude spacing (the spacing
    at which the pre-fix code did NOT raise), the resampled output equals
    hand-computed linear interpolation at each grid tick."""
    d = 33_333_333
    ts = np.array([_EPOCH_NS + i * d for i in range(5)], dtype=np.int64)
    values = np.array([[float(i), float(2 * i)] for i in range(5)], dtype=np.float32)
    grid = np.array(
        [_EPOCH_NS + i * d + d // 3 for i in range(4)], dtype=np.int64
    )  # a third of the way into each interval
    result = _resample_series(ts, values, grid, "interp")
    for i in range(4):
        frac = (d // 3) / d
        expected = values[i] + frac * (values[i + 1] - values[i])
        np.testing.assert_allclose(result[i], expected, atol=1e-3)


def test_no_unreferenced_float64_cast_of_a_timestamp_array_in_module() -> None:
    """Source-level gate: every `astype(np.float64)` occurrence in
    master_clock.py that casts a TIMESTAMP-bearing array (name matching
    `ts_ns`/`clipped`/`grid`), excluding comment lines, must appear on a
    line that also contains a subtraction — the textual signature of a
    reference-relative conversion. Scoped to timestamp-bearing names so the
    legitimate `sample_values` float64 cast (not a timestamp) is not
    flagged. Prevents a future edit from reintroducing an absolute-epoch
    float64 cast."""
    import data.sync.master_clock as module

    source_path = module.__file__
    assert source_path is not None
    lines = open(source_path).read().splitlines()
    offending = [
        (i + 1, line.strip())
        for i, line in enumerate(lines)
        if "astype(np.float64)" in line
        and not line.strip().startswith("#")
        and re.search(r"ts_ns|clipped|grid", line)
        and "-" not in line
    ]
    assert offending == [], f"float64 cast with no reference subtraction: {offending}"
