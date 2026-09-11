"""Task 2: full-stream expansion of the Galaxea adapter — state assembly,
ordered action-source resolution, structural aborts, and the stream
inventory. Every test builds its own fixture except the real-sample test,
which skips (never fails) when `data/samples/` is absent.
"""

import glob

import numpy as np
import pytest

import data.tests.mcap_fixture as fixture_mod
from data.adapters.galaxea import GalaxeaAdapterError, GalaxeaInputAdapter
from data.contracts.profile import load_profile

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"

_STATE_VECTOR_WIDTH = 20
"""Measured, not asserted in adapter code (DERISK-FINDINGS.md): 7+7+4+1+1
over state_vector_order's five keys. Pinned here, in the test — a grep of
data/adapters/ must find no `20`/`26`/`19` literal vector width."""


def _profile():
    return load_profile(PROFILE_PATH)


def _first_real_file_with_full_camera_coverage(profile) -> str | None:
    """The two real-sample tests below exercise general real-MCAP decoding
    (state-vector width, stream inventory, multi-camera MP4 output) and
    predate G-01/VERIFY-01's camera-coverage enforcement (04-04-PLAN.md).
    At least one of the three real recordings under `data/samples/`
    (gitignored, locally measured) genuinely has a secondary camera whose
    recording starts ~170ms after the master's — a real, now-correctly-
    detected synchronization defect, not a bug in these tests. Picks the
    first real file whose coverage check would not truncate/reject it, so
    these two tests keep exercising the general decoding path they were
    written for rather than becoming entangled with a specific real
    recording's own defect (which has its own dedicated coverage in
    `data/tests/test_sync_camera_coverage.py`)."""
    for path in sorted(glob.glob("data/samples/vr-teleop-20260904/*/*.mcap")):
        try:
            episode = GalaxeaInputAdapter().parse(path, profile)
        except GalaxeaAdapterError:
            continue
        if (episode.header.frame_count or 0) > 0:
            return path
    return None


def test_full_state_vector_assembles_to_measured_width(tmp_path) -> None:
    profile = _profile()
    fixture_path = tmp_path / "full.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=4, joint_samples=4)

    episode = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    frame = next(iter(episode.frames))

    expected_width = sum(
        len(f.joint_names)
        for f in profile.state_topics.values()
        if f.include_in_observation_state
    )
    assert expected_width == _STATE_VECTOR_WIDTH

    total_width = 0
    for state_key in profile.state_vector_order:
        assert state_key in frame.state
        assert frame.state[state_key].dtype == np.float32
        total_width += frame.state[state_key].shape[0]
    assert total_width == _STATE_VECTOR_WIDTH


def test_widening_one_field_changes_emitted_width_with_no_adapter_edit(tmp_path) -> None:
    profile = _profile()
    torso = profile.state_topics["torso"]
    widened_torso = torso.model_copy(update={"joint_names": [*torso.joint_names, "torso_j5"]})
    widened_profile = profile.model_copy(
        update={"state_topics": {**profile.state_topics, "torso": widened_torso}}
    )

    fixture_path = tmp_path / "widened.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_torso": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE,
                fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                fixture_mod.joint_state_messages(count=3, joint_count=5, name="torso"),
            )
        },
    )

    episode = GalaxeaInputAdapter().parse(str(fixture_path), widened_profile)
    frame = next(iter(episode.frames))
    total_width = sum(
        frame.state[key].shape[0] for key in widened_profile.state_vector_order
    )
    assert total_width == _STATE_VECTOR_WIDTH + 1


def test_declared_measured_length_mismatch_aborts(tmp_path) -> None:
    fixture_path = tmp_path / "mismatch.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE,
                fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                fixture_mod.joint_state_messages(count=3, joint_count=6, name="left_arm"),
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    message = str(excinfo.value)
    assert "/hdas/feedback_arm_left" in message
    assert "7" in message
    assert "6" in message


def test_missing_required_topics_abort_before_any_decode(tmp_path, monkeypatch) -> None:
    fixture_path = tmp_path / "missing.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_arm_left": None,
            "/hdas/feedback_torso": None,
            "/hdas/feedback_gripper_right": None,
        },
    )

    calls: list[str] = []

    import data.adapters.galaxea as galaxea_mod

    def _spy(path: str, topics: list[str]):
        calls.append(path)
        raise AssertionError("decode must not be reached when a required topic is missing")

    monkeypatch.setattr(galaxea_mod, "iter_decoded_messages", _spy)

    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    message = str(excinfo.value)
    assert calls == []
    assert "/hdas/feedback_arm_left" in message
    assert "/hdas/feedback_gripper_right" in message
    assert "/hdas/feedback_torso" in message
    # sorted order
    left_pos = message.index("/hdas/feedback_arm_left")
    right_pos = message.index("/hdas/feedback_gripper_right")
    torso_pos = message.index("/hdas/feedback_torso")
    assert left_pos < right_pos < torso_pos


def test_required_topic_present_zero_messages_has_distinct_wording(tmp_path) -> None:
    fixture_path = tmp_path / "empty_required.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_arm_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE, fixture_mod.SCHEMA_TEXT_JOINT_STATE, []
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    present_zero_message = str(excinfo.value)

    fixture_path_2 = tmp_path / "absent_required.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path_2),
        camera_frames=3,
        joint_samples=3,
        topics={"/hdas/feedback_arm_left": None},
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo2:
        GalaxeaInputAdapter().parse(str(fixture_path_2), _profile())
    absent_message = str(excinfo2.value)

    assert present_zero_message != absent_message
    assert "zero messages" in present_zero_message
    assert "channel list" in absent_message


def test_optional_chassis_zero_messages_does_not_abort_and_is_not_zero_filled(tmp_path) -> None:
    fixture_path = tmp_path / "no_chassis.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/hdas/feedback_chassis": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE, fixture_mod.SCHEMA_TEXT_JOINT_STATE, []
            )
        },
    )
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    frame = next(iter(episode.frames))
    assert "chassis" not in frame.state


def test_arm_left_action_resolves_to_motor_control_when_target_empty(tmp_path) -> None:
    fixture_path = tmp_path / "fallback.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=3, joint_samples=3)
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert episode.header.global_meta["resolved_action_source"]["left_arm"] == (
        "/motion_control/control_arm_left"
    )


def test_arm_left_action_resolves_to_first_source_when_both_populated(tmp_path) -> None:
    fixture_path = tmp_path / "both_populated.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/motion_target/target_joint_state_arm_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE,
                fixture_mod.SCHEMA_TEXT_JOINT_STATE,
                fixture_mod.joint_state_messages(count=3, joint_count=7, name="left_arm_target"),
            )
        },
    )
    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert episode.header.global_meta["resolved_action_source"]["left_arm"] == (
        "/motion_target/target_joint_state_arm_left"
    )


def test_action_field_all_sources_empty_aborts_when_required(tmp_path) -> None:
    fixture_path = tmp_path / "no_gripper_action.mcap"
    fixture_mod.build_fixture_mcap(
        str(fixture_path),
        camera_frames=3,
        joint_samples=3,
        topics={
            "/motion_target/target_position_gripper_left": fixture_mod.TopicSpec(
                fixture_mod.SCHEMA_NAME_JOINT_STATE, fixture_mod.SCHEMA_TEXT_JOINT_STATE, []
            )
        },
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    assert "left_gripper" in str(excinfo.value)


def test_stream_inventory_covers_every_channel_sorted_with_unemitted_and_unset_hz(
    tmp_path,
) -> None:
    fixture_path = tmp_path / "inventory.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=3, joint_samples=3)

    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    inventory = episode.header.global_meta["stream_inventory"]

    fixture_topics = set(fixture_mod.default_topics(camera_frames=3, joint_samples=3))
    assert {row.topic for row in inventory} == fixture_topics
    assert [row.topic for row in inventory] == sorted(row.topic for row in inventory)

    by_topic = {row.topic: row for row in inventory}
    assert by_topic[fixture_mod.TOPIC_TASK_TAKEOVER_STATUS].emitted is False
    assert by_topic[fixture_mod.TOPIC_VR_POSE].emitted is False
    assert by_topic["/motion_target/target_joint_state_arm_left"].message_count == 0
    assert by_topic["/motion_target/target_joint_state_arm_left"].measured_hz is None
    assert by_topic["/hdas/camera_head/right_raw/image_raw_color/compressed"].emitted is False
    assert by_topic["/hdas/camera_wrist_left/color/image_raw/compressed"].emitted is True


def test_global_meta_records_joint_name_ordering_and_profile_hash_deterministically(
    tmp_path,
) -> None:
    fixture_path = tmp_path / "ordering.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=3, joint_samples=3)
    profile = _profile()

    episode_1 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    meta_1 = episode_1.header.global_meta
    list(episode_1.frames)  # exhaust before re-parsing

    episode_2 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    meta_2 = episode_2.header.global_meta
    list(episode_2.frames)

    assert meta_1["joint_name_ordering"] == meta_2["joint_name_ordering"]
    assert meta_1["profile_hash"] == meta_2["profile_hash"]
    assert meta_1["joint_name_ordering"]["arm_left"] == tuple(
        profile.state_topics["arm_left"].joint_names
    )
    assert meta_1["joint_name_ordering"]["left_arm"] == tuple(
        profile.action_topics["left_arm"].joint_names
    )


def test_real_sample_recording_if_present() -> None:
    real_files = sorted(glob.glob("data/samples/vr-teleop-20260904/*/*.mcap"))
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    profile = _profile()
    clean_file = _first_real_file_with_full_camera_coverage(profile)
    if clean_file is None:
        pytest.skip("every real sample file has a camera-coverage defect on this checkout")
    episode = GalaxeaInputAdapter().parse(clean_file, profile)
    frame = next(iter(episode.frames))
    total_width = sum(frame.state[key].shape[0] for key in profile.state_vector_order)
    assert total_width == _STATE_VECTOR_WIDTH

    topics = {row.topic for row in episode.header.global_meta["stream_inventory"]}
    assert any("camera_info" in t or "CameraInfo" in t for t in topics) or any(
        "tf" in t for t in topics
    ) or "/sysmgr/task_takeover_status" in topics


def test_secondary_cameras_reach_canonical_frames_end_to_end(tmp_path) -> None:
    """Activates SYNC-01..05's secondary-camera nearest-match capability
    through the REAL adapter, not just against hand-built `raw_streams`
    (02-04's documented gap). `default_topics`' camera messages all share
    the same `start_ns=0, step_ns=40_000_000` timing by default, so every
    tick is well inside the profile's declared 40ms gap bound — every
    frame should carry all three emitted cameras."""
    profile = _profile()
    fixture_path = tmp_path / "multi_camera.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)

    episode = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    frames = list(episode.frames)
    assert frames

    expected_keys = {
        "observation.images.head",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    }
    for frame in frames:
        assert expected_keys <= set(frame.images.keys())


def test_real_sample_secondary_cameras_reach_writer_as_multiple_mp4s_if_present(
    tmp_path,
) -> None:
    """The end-to-end proof the 02-04 handoff asked for: a real multi-camera
    recording reaches `LeRobotV21Writer` as more than one MP4 — skips (never
    fails) when `data/samples/` is absent, matching this module's existing
    real-sample convention."""
    real_files = sorted(glob.glob("data/samples/vr-teleop-20260904/*/*.mcap"))
    if not real_files:
        pytest.skip("data/samples/ is absent on this checkout — gitignored, not a fresh-clone artifact")

    from data.writers.lerobot_v21 import LeRobotV21Writer

    profile = _profile()
    clean_file = _first_real_file_with_full_camera_coverage(profile)
    if clean_file is None:
        pytest.skip("every real sample file has a camera-coverage defect on this checkout")
    episode = GalaxeaInputAdapter().parse(clean_file, profile)

    writer = LeRobotV21Writer(
        tmp_path, profile, stdin_isatty=lambda: True, prompt=lambda: "fixture task"
    )
    result = writer.write_episode(episode)
    writer.finalize()

    mp4_keys = {k for k in result.output_paths if k.startswith("observation.images.")}
    assert len(mp4_keys) > 1
    assert {"observation.images.wrist_left", "observation.images.wrist_right"} <= mp4_keys
