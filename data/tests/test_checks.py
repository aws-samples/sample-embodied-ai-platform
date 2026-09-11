"""The single fatal/advisory check registry (INTEG-01), its thresholds
(D-07/D-08), and the CRC regression test 02-RESEARCH.md Pitfall 1 names
explicitly. Every test builds its own fixture and runs it through
`GalaxeaInputAdapter().parse()` — the registry is exercised through the
adapter, not directly, since fatal checks must abort inside `parse()`
before any payload decode (INTEG-01).
"""

import numpy as np
import pytest

import data.tests.mcap_fixture as fixture_mod
from data.adapters.checks import AdvisoryCheckContext, CheckRegistry, _check_frozen_duplicate_frames
from data.adapters.galaxea import GalaxeaAdapterError, GalaxeaInputAdapter
from data.contracts.containers import CheckResult, Disposition
from data.contracts.profile import load_profile

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"


def _profile():
    return load_profile(PROFILE_PATH)


def test_registry_fatal_and_advisory_name_sets_are_disjoint_and_complete() -> None:
    registry = CheckRegistry.default()
    fatal_names = {c.name for c in registry.fatal}
    advisory_names = {c.name for c in registry.advisory}
    all_names = {c.name for c in registry.checks}
    assert not (fatal_names & advisory_names)
    assert fatal_names | advisory_names == all_names
    assert fatal_names
    assert advisory_names


def test_clean_fixture_yields_pass_and_nonempty_check_tuple(tmp_path) -> None:
    """"Clean" here means every registered check has enough information to
    genuinely evaluate and pass — not just the structural checks: distinct
    (non-duplicate) camera frames on EVERY camera topic (WR-01: the fixed
    `videos.frozen_duplicate_frames` check now inspects every topic, not
    only the master's, so every camera needs distinct frames for this fixture
    to be genuinely clean), and generous declared velocity/acceleration
    limits on every state field, since D-08 makes an undeclared limit
    report as NOT passing (visibly unevaluated), which the shipped Galaxea
    profile (no declared limits anywhere) would otherwise always trigger."""
    profile = _profile()
    widened_state_topics = {
        name: f.model_copy(update={"max_velocity": 1000.0, "max_acceleration": 1000.0})
        for name, f in profile.state_topics.items()
    }
    generous_profile = profile.model_copy(update={"state_topics": widened_state_topics})

    fixture_path = tmp_path / "clean.mcap"

    def _distinct_frames(base: tuple[int, int, int]) -> list:
        return fixture_mod.camera_messages(
            count=5,
            jpeg_bytes=[
                fixture_mod._make_tiny_jpeg((base[0] + i, base[1] + i, base[2] + i))
                for i in range(5)
            ],
        )

    camera_topic_specs = {
        topic: fixture_mod.TopicSpec(
            fixture_mod.SCHEMA_NAME_COMPRESSED_IMAGE,
            fixture_mod.SCHEMA_TEXT_COMPRESSED_IMAGE,
            _distinct_frames(base),
        )
        for topic, base in {
            "/hdas/camera_head/left_raw/image_raw_color/compressed": (10, 20, 30),
            "/hdas/camera_head/right_raw/image_raw_color/compressed": (40, 50, 60),
            "/hdas/camera_wrist_left/color/image_raw/compressed": (70, 80, 90),
            "/hdas/camera_wrist_right/color/image_raw/compressed": (100, 110, 120),
        }.items()
    }
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=5,
        joint_samples=5,
        topics=camera_topic_specs,
    )

    episode = GalaxeaInputAdapter().parse(str(fixture_path), generous_profile)
    checks = episode.header.global_meta["checks"]
    # +1: MasterClockSynchronizer.synchronize appends the G-01/VERIFY-01
    # cameras.coverage_deficit check after every registry-run check
    # (04-04-PLAN.md) -- every camera here gets an equal frame count, so it
    # passes with zero deficit, same as the registry's own checks.
    assert len(checks) == len(CheckRegistry.default().checks) + 1
    assert all(isinstance(c, CheckResult) for c in checks)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert episode.header.global_meta["disposition"] == Disposition.PASS


def test_corrupted_chunk_crc_is_rejected_by_named_fatal_check(tmp_path) -> None:
    fixture_path = tmp_path / "corrupt.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)
    fixture_mod.corrupt_chunk_crc(str(fixture_path))

    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert "metadata.mcap_crc" in str(excinfo.value)


def test_non_monotonic_timestamp_sequence_is_rejected_by_named_fatal_check(tmp_path) -> None:
    fixture_path = tmp_path / "non_monotonic.mcap"
    # Deliberately decreasing timestamps on arm_left's feedback topic.
    messages = fixture_mod.joint_state_messages(
        count=4, joint_count=7, name="left_arm", start_ns=0, step_ns=40_000_000
    )
    scrambled = [messages[0], messages[2], messages[1], messages[3]]
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=5,
        joint_samples=5,
        topics={
            "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE, fixture_mod.SCHEMA_TEXT_JOINT_STATE, scrambled
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert "temporal.monotonic_timestamps" in str(excinfo.value)


def test_zero_frame_master_camera_is_rejected_by_named_fatal_check(tmp_path) -> None:
    fixture_path = tmp_path / "no_master_frames.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=5,
        joint_samples=5,
        topics={
            "/hdas/camera_head/left_raw/image_raw_color/compressed": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_COMPRESSED_IMAGE,
                fixture_mod.SCHEMA_TEXT_COMPRESSED_IMAGE,
                [],
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert "videos.master_camera_frames" in str(excinfo.value)


def test_non_ros2msg_schema_encoding_is_rejected_by_named_fatal_check(tmp_path) -> None:
    fixture_path = tmp_path / "bad_encoding.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=5,
        joint_samples=5,
        topics={
            "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                "sensor_msgs/msg/JointState",
                fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                [(0, {"position": [0.0] * 7})],
                encoding="ros2idl",
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert "metadata.schema_decodability" in str(excinfo.value)


def test_repeated_camera_payload_yields_advisory_duplicate_check_and_pass_with_flags(
    tmp_path,
) -> None:
    fixture_path = tmp_path / "frozen.mcap"
    repeated = fixture_mod._TINY_JPEG_BYTES
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=5,
        joint_samples=5,
        topics={
            "/hdas/camera_head/left_raw/image_raw_color/compressed": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_COMPRESSED_IMAGE,
                fixture_mod.SCHEMA_TEXT_COMPRESSED_IMAGE,
                fixture_mod.camera_messages(count=5, jpeg_bytes=repeated),
            )
        },
    )
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    checks = {c.check_name: c for c in episode.header.global_meta["checks"]}
    duplicate_check = checks["videos.frozen_duplicate_frames"]
    master_topic = "/hdas/camera_head/left_raw/image_raw_color/compressed"
    assert duplicate_check.evidence["per_topic"][master_topic]["duplicate_count"] == 4
    assert duplicate_check.evidence["blind_spot"] == "re_encoded_static_scene_not_detected"
    assert episode.header.global_meta["disposition"] == Disposition.PASS_WITH_FLAGS


def test_single_frame_camera_yields_unevaluated_duplicate_check_not_a_pass(tmp_path) -> None:
    """A single master-clock camera frame is enough to exercise this
    advisory check's own "unevaluated" marker (it only counts digest pairs
    from the index pass), but 02-04's SYNC-01 rate-detection abort would
    otherwise reject a 1-frame master clock when no rate is declared — so
    this fixture's profile pins `target_fps` explicitly, sidestepping
    detection entirely without touching what this test actually measures."""
    fixture_path = tmp_path / "single_frame.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=1, joint_samples=3)
    profile = _profile()
    pinned_rate_profile = profile.model_copy(
        update={"master_clock": profile.master_clock.model_copy(update={"target_fps": 25.0})}
    )
    episode = GalaxeaInputAdapter().parse(str(fixture_path), pinned_rate_profile)
    checks = {c.check_name: c for c in episode.header.global_meta["checks"]}
    duplicate_check = checks["videos.frozen_duplicate_frames"]
    master_topic = "/hdas/camera_head/left_raw/image_raw_color/compressed"
    assert duplicate_check.evidence["per_topic"][master_topic]["pairs_compared"] == 0
    assert duplicate_check.evidence["per_topic"][master_topic]["unevaluated"] is True
    assert duplicate_check.passed is False


def test_secondary_camera_frozen_frames_now_fails_ingest_check() -> None:
    """WR-01 regression: before the fix, `_check_frozen_duplicate_frames`
    read only `ctx.master_camera_topic`'s digests, so a stalled
    secondary-camera (wrist) driver went undetected — this exact context
    would have reported `passed=True` against the pre-fix implementation."""
    master_topic = "/hdas/camera_head/left_raw/image_raw_color/compressed"
    wrist_topic = "/hdas/camera_wrist_left/color/image_raw/compressed"
    ctx = AdvisoryCheckContext(
        camera_payload_digests={
            master_topic: ["a", "b", "c"],
            wrist_topic: ["x", "x", "x"],
        },
        master_camera_topic=master_topic,
        joint_series=(),
        cross_stream_timestamps={},
    )
    result = _check_frozen_duplicate_frames(ctx)
    assert result.passed is False
    assert result.evidence["per_topic"][wrist_topic]["duplicate_count"] == 2
    assert result.evidence["per_topic"][master_topic]["duplicate_count"] == 0


def test_single_digest_topic_yields_per_topic_unevaluated_not_whole_check() -> None:
    """The unevaluated marker moved from the whole check's evidence to a
    per-topic entry — the fixed check still returns exactly one
    `CheckResult` and still carries the static-scene blind-spot marker."""
    ctx = AdvisoryCheckContext(
        camera_payload_digests={"/topic/a": ["only-one"]},
        master_camera_topic="/topic/a",
        joint_series=(),
        cross_stream_timestamps={},
    )
    result = _check_frozen_duplicate_frames(ctx)
    assert result.evidence["per_topic"]["/topic/a"]["unevaluated"] is True
    assert "unevaluated" not in result.evidence
    assert result.evidence["blind_spot"] == "re_encoded_static_scene_not_detected"
    assert result.passed is False


def test_velocity_limit_exactly_at_bound_does_not_flag_one_beyond_does(tmp_path) -> None:
    profile = _profile()
    arm_left = profile.state_topics["arm_left"]
    # step_ns=40ms -> dt=0.04s. velocity = delta_position / dt.
    # Pick a delta that yields velocity == limit exactly, and a delta that
    # yields one increment beyond it, in two separate fixtures.
    limit = 2.0
    dt_s = 0.04
    delta_at_limit = limit * dt_s
    delta_beyond = (limit + 0.5) * dt_s

    def _fixture_with_delta(path, delta: float):
        positions = [[0.0] * 7, [delta] * 7, [0.0] * 7]
        messages = list(
            zip(
                [0, 40_000_000, 80_000_000],
                [
                    {
                        "header": fixture_mod._header_value(t),
                        "name": ["left_arm"],
                        "position": p,
                        "velocity": [],
                        "effort": [],
                    }
                    for t, p in zip([0, 40_000_000, 80_000_000], positions)
                ],
            )
        )
        fixture_mod.build_fixture_mcap(
            str(path),
            camera_frames=3,
            joint_samples=3,
            topics={
                "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                    fixture_mod.SCHEMA_NAME_JOINT_STATE,
                    fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                    messages,
                )
            },
        )

    widened_arm_left = arm_left.model_copy(update={"max_velocity": limit})
    limited_profile = profile.model_copy(
        update={"state_topics": {**profile.state_topics, "arm_left": widened_arm_left}}
    )

    at_limit_path = tmp_path / "at_limit.mcap"
    _fixture_with_delta(at_limit_path, delta_at_limit)
    episode_at_limit = GalaxeaInputAdapter().parse(str(at_limit_path), limited_profile)
    check_at_limit = {
        c.check_name: c for c in episode_at_limit.header.global_meta["checks"]
    }["kinematics.velocity_acceleration_bounds"]
    field_evidence_at_limit = check_at_limit.evidence["per_field"]["/hdas/feedback_arm_left"]
    assert "resolved_max_velocity_limit" in field_evidence_at_limit
    assert field_evidence_at_limit["resolved_max_velocity_limit"] == limit
    # max_acceleration was deliberately left undeclared on this profile
    # copy — only max_velocity should be absent from no_limit_declared.
    assert "max_velocity" not in field_evidence_at_limit.get("no_limit_declared", [])

    beyond_path = tmp_path / "beyond.mcap"
    _fixture_with_delta(beyond_path, delta_beyond)
    episode_beyond = GalaxeaInputAdapter().parse(str(beyond_path), limited_profile)
    check_beyond = {
        c.check_name: c for c in episode_beyond.header.global_meta["checks"]
    }["kinematics.velocity_acceleration_bounds"]
    assert episode_beyond.header.global_meta["disposition"] == Disposition.PASS_WITH_FLAGS
    assert check_beyond.passed is False


def test_no_declared_limit_carries_no_limit_marker_and_measured_extremes(tmp_path) -> None:
    fixture_path = tmp_path / "no_limit.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    check = {c.check_name: c for c in episode.header.global_meta["checks"]}[
        "kinematics.velocity_acceleration_bounds"
    ]
    field_evidence = check.evidence["per_field"]["/hdas/feedback_arm_left"]
    assert "max_velocity" in field_evidence["no_limit_declared"]
    assert "measured_max_velocity" in field_evidence
    assert check.passed is False


def test_zero_timestamp_delta_counted_not_infinite(tmp_path) -> None:
    fixture_path = tmp_path / "zero_delta.mcap"
    same_ts_messages = [
        (
            0,
            {
                "header": fixture_mod._header_value(0),
                "name": ["left_arm"],
                "position": [0.0] * 7,
                "velocity": [],
                "effort": [],
            },
        ),
        (
            0,
            {
                "header": fixture_mod._header_value(0),
                "name": ["left_arm"],
                "position": [1.0] * 7,
                "velocity": [],
                "effort": [],
            },
        ),
        (
            40_000_000,
            {
                "header": fixture_mod._header_value(40_000_000),
                "name": ["left_arm"],
                "position": [1.0] * 7,
                "velocity": [],
                "effort": [],
            },
        ),
    ]
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE,
                fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                same_ts_messages,
            )
        },
    )
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    check = {c.check_name: c for c in episode.header.global_meta["checks"]}[
        "kinematics.velocity_acceleration_bounds"
    ]
    field_evidence = check.evidence["per_field"]["/hdas/feedback_arm_left"]
    assert field_evidence["zero_delta_sample_count"] == 1
    assert np.isfinite(field_evidence["measured_max_velocity"])


def test_skew_reported_in_ns_with_rounding_mode_and_single_stream_marker(tmp_path) -> None:
    fixture_path = tmp_path / "skew.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    check = {c.check_name: c for c in episode.header.global_meta["checks"]}[
        "temporal.cross_stream_skew"
    ]
    assert isinstance(check.evidence["max_skew_ns"], int)
    assert isinstance(check.evidence["mean_skew_ns"], int)
    assert check.evidence["rounding_mode"] == "floor"


def test_check_sequence_identical_across_two_runs(tmp_path) -> None:
    fixture_path = tmp_path / "repeatable.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)
    profile = _profile()
    episode_1 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    names_1 = [c.check_name for c in episode_1.header.global_meta["checks"]]
    list(episode_1.frames)
    episode_2 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    names_2 = [c.check_name for c in episode_2.header.global_meta["checks"]]
    list(episode_2.frames)
    assert names_1 == names_2
