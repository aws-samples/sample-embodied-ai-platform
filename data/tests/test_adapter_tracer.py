"""End-to-end tracer test: one synthetic Galaxea MCAP -> one CanonicalEpisode,
through the CRC-validating index pass, the streaming decode pass, and the
master-clock grid — plus the INGEST-01 three-custom-package decode proof.
"""

import importlib.util

import numpy as np
import pytest

from data.adapters.galaxea import (
    CompressedImageRef,
    GalaxeaInputAdapter,
    iter_decoded_messages,
)
from data.contracts.containers import CanonicalEpisode, ImageRef
from data.contracts.profile import EmbodimentProfile, load_profile
from data.contracts.protocols import BaseInputAdapter, BaseSynchronizer
from data.sync.master_clock import MasterClockSynchronizer
from data.tests.mcap_fixture import (
    SCHEMA_NAME_MOTOR_CONTROL,
    SCHEMA_NAME_TASK_TAKEOVER_STATUS,
    SCHEMA_NAME_VR_POSE,
    TOPIC_TASK_TAKEOVER_STATUS,
    TOPIC_VR_POSE,
    build_fixture_mcap,
)

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"


def _profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def test_tracer_end_to_end(tmp_path) -> None:
    fixture_path = tmp_path / "tracer.mcap"
    build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)

    profile = _profile()
    episode = GalaxeaInputAdapter().parse(str(fixture_path), profile)

    frames = list(episode.frames)
    assert len(frames) == 5

    first = frames[0]
    expected_left_arm = np.array([1.0 + 0 * 0.1 + j * 0.01 for j in range(7)], dtype=np.float32)
    assert first.state["left_arm"].dtype == np.float32
    assert first.state["left_arm"].shape == (7,)
    np.testing.assert_allclose(first.state["left_arm"], expected_left_arm, atol=1e-5)

    master_video_key = "observation.images.head"
    assert master_video_key in first.images
    image_ref = first.images[master_video_key]
    pixels = image_ref.decode()
    assert pixels.ndim == 3
    assert pixels.shape[2] == 3
    assert pixels.dtype == np.uint8

    assert episode.header.fps == pytest.approx(25.0, rel=1e-6)

    # Single-pass guarantee: iterating episode.frames a second time yields nothing.
    assert list(episode.frames) == []


def test_camera_buffers_hold_bytes_never_arrays(tmp_path) -> None:
    """INGEST-05's structural proof: the per-topic camera buffer inside the
    decode pass holds compressed `bytes`, never a decoded pixel array — no
    structure anywhere in the index or decode pass holds a whole episode's
    decoded frames. Checked at both the `CompressedImageRef.data` field
    (never `np.ndarray`) and the still-undecoded state before any caller
    calls `.decode()`."""
    fixture_path = tmp_path / "bytes_not_arrays.mcap"
    build_fixture_mcap(str(fixture_path), camera_frames=3, joint_samples=3)

    episode = GalaxeaInputAdapter().parse(str(fixture_path), _profile())
    for frame in episode.frames:
        for image_ref in frame.images.values():
            assert isinstance(image_ref, CompressedImageRef)
            assert isinstance(image_ref.data, bytes)
            assert not isinstance(image_ref.data, np.ndarray)


def test_parse_is_read_only_and_idempotent(tmp_path) -> None:
    """INGEST-01: the adapter opens every MCAP read-only and writes no
    file, so an interrupted or concurrently-run parse() leaves the source
    recording and the working tree byte-identical. Two parse() calls over
    the same fixture MCAP produce identical decoded first-frame state, and
    the fixture's own bytes are untouched by either call."""
    fixture_path = tmp_path / "idempotent.mcap"
    build_fixture_mcap(str(fixture_path), camera_frames=4, joint_samples=4)
    source_bytes_before = fixture_path.read_bytes()

    profile = _profile()
    episode_1 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    first_1 = next(iter(episode_1.frames))

    episode_2 = GalaxeaInputAdapter().parse(str(fixture_path), profile)
    first_2 = next(iter(episode_2.frames))

    np.testing.assert_array_equal(first_1.state["left_arm"], first_2.state["left_arm"])
    assert fixture_path.read_bytes() == source_bytes_before


def test_ingest_01_three_package_decode_proof(tmp_path) -> None:
    """INGEST-01's hardest claim: all three custom message packages decode
    from embedded ros2msg schema text alone, with no ROS2 install, and this
    proof runs entirely against the synthetic fixture — the real recordings
    can never exercise it (`/sysmgr/task_takeover_status` measures 0
    messages in all three real samples, DERISK-02)."""
    assert importlib.util.find_spec("rclpy") is None

    fixture_path = tmp_path / "three_package.mcap"
    build_fixture_mcap(str(fixture_path), camera_frames=1, joint_samples=1)

    motor_control_topic = "/motion_control/control_arm_left"
    seen: dict[str, object] = {}
    for topic, _ts, msg in iter_decoded_messages(
        str(fixture_path), [motor_control_topic, TOPIC_VR_POSE, TOPIC_TASK_TAKEOVER_STATUS]
    ):
        if topic == motor_control_topic and "hdas_msg" not in seen:
            seen["hdas_msg"] = list(msg.p_des)
        elif topic == TOPIC_VR_POSE and "teleoperation_msg_ros2" not in seen:
            seen["teleoperation_msg_ros2"] = msg.control_mode
        elif topic == TOPIC_TASK_TAKEOVER_STATUS and "system_manager_msg" not in seen:
            seen["system_manager_msg"] = msg.task_id

    assert seen["hdas_msg"] == pytest.approx(
        [2.0 + j * 0.01 for j in range(7)]
    ), f"named field p_des on {SCHEMA_NAME_MOTOR_CONTROL} did not round-trip exactly"
    assert seen["teleoperation_msg_ros2"] == 3, (
        f"named field control_mode on {SCHEMA_NAME_VR_POSE} did not round-trip exactly"
    )
    assert seen["system_manager_msg"] == "fixture-task-0", (
        f"named field task_id on {SCHEMA_NAME_TASK_TAKEOVER_STATUS} did not round-trip exactly"
    )


# --- CONT-02/03/04 protocol conformance: no-inheritance stubs, mypy-checked
# via the annotated assignments below, matching test_contracts.py's pattern.


def test_stub_implementers_have_no_inheritance() -> None:
    for cls in (GalaxeaInputAdapter, MasterClockSynchronizer, CompressedImageRef):
        assert cls.__mro__ == (cls, object)


def _protocol_conformance_annotations() -> None:
    """Never called — exists purely so mypy checks these annotated
    assignments prove structural conformance without inheritance."""
    adapter: BaseInputAdapter = GalaxeaInputAdapter()
    synchronizer: BaseSynchronizer = MasterClockSynchronizer()
    image_ref: ImageRef = CompressedImageRef(b"", 0)
    del adapter, synchronizer, image_ref


def _episode_type_annotation_uses_canonical_episode(
    episode: CanonicalEpisode,
) -> CanonicalEpisode:
    """Exists purely so mypy checks GalaxeaInputAdapter.parse's return type
    is exactly CanonicalEpisode (never re-imports the removed
    TrajectoryEpisode name)."""
    return episode
