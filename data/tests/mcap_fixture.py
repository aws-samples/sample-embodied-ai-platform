"""Synthetic MCAP fixture builder for the Track A test suite.

A tracked test *helper*, not a `test_*` module — pytest does not collect it, and
it is importable as `data.tests.mcap_fixture` via `pytest.ini`'s
`pythonpath = .`.

Every schema-text constant below is authored to decode through
`mcap_ros2`'s runtime `ros2msg` decoder with **zero ROS2 install** — this
module is what makes every adapter test (including INGEST-01's hardest claim,
the three-custom-package decode proof) runnable on a fresh clone where
`data/samples/` is absent. `/sysmgr/task_takeover_status` carries 0 messages in
all three real sample recordings (DERISK-02), so the real samples can *never*
prove that decoder path — only this fixture can.

Provenance of the two extra custom-package schemas (`teleoperation_msg_ros2/msg/VrPose`,
`system_manager_msg/msg/TaskTakeoverStatus`): extracted **verbatim** from
`data/samples/vr-teleop-20260904/R2026A63W0A9900005_20260904065700769_RAW/…mcap`'s
own `Schema` records via a one-off `mcap.stream_reader.StreamReader` walk
(`schema.data.decode()`), not authored from scratch — the real recordings were
present on the machine that wrote this file. See the constants below for the
exact real topic names those schemas travel on (`/vr_pose`,
`/sysmgr/task_takeover_status`). `hdas_msg/msg/MotorControl`'s schema text is
intentionally NOT verbatim: the real recording's own `p_des` field is
`float32[]`, but this fixture declares `float64[]` per this plan's own Task 1
action text — a deliberate, documented divergence, not an oversight.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import simplejpeg
from mcap.writer import CompressionType
from mcap_ros2.writer import Writer as Ros2Writer

_DEP_SEP = "=" * 80

# ---------------------------------------------------------------------------
# ros2msg schema text — one module-level constant per message type.
# ---------------------------------------------------------------------------

SCHEMA_NAME_HEADER = "std_msgs/msg/Header"
SCHEMA_TEXT_HEADER = "builtin_interfaces/Time stamp\nstring frame_id\n"
"""Dependent-only: never registered as a topic's own top-level schema, only
concatenated into every schema below that carries a `header` field — exactly
how the real recordings' own schema records repeat it per top-level type."""

SCHEMA_NAME_TIME = "builtin_interfaces/msg/Time"
SCHEMA_TEXT_TIME = "int32 sec\nuint32 nanosec\n"
"""Also dependent-only. `mcap_ros2._dynamic.generate_dynamic` pre-registers
`builtin_interfaces/Time` unconditionally, so no schema below strictly needs
to redeclare it — but the two verbatim-extracted schemas (VrPose,
TaskTakeoverStatus) redeclare it anyway, because that is what their real
schema records do, and the closer this fixture's bytes are to the real
recordings' bytes, the more the decode proof built on them is worth."""


def _with_header(body: str) -> str:
    """Appends the standard `std_msgs/Header` dependent-type block."""
    return f"{body}\n{_DEP_SEP}\nMSG: std_msgs/Header\n{SCHEMA_TEXT_HEADER}"


SCHEMA_NAME_JOINT_STATE = "sensor_msgs/msg/JointState"
SCHEMA_TEXT_JOINT_STATE = _with_header(
    "std_msgs/Header header\n"
    "string[] name\n"
    "float64[] position\n"
    "float64[] velocity\n"
    "float64[] effort\n"
)

SCHEMA_NAME_COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
SCHEMA_TEXT_COMPRESSED_IMAGE = _with_header(
    "std_msgs/Header header\nstring format\nuint8[] data\n"
)

SCHEMA_NAME_MOTOR_CONTROL = "hdas_msg/msg/MotorControl"
SCHEMA_TEXT_MOTOR_CONTROL = _with_header(
    "std_msgs/Header header\nstring name\nfloat64[] p_des\n"
)
"""INGEST-01 names this package explicitly. Minimal, NOT verbatim — see module
docstring: the real recording's own `p_des` is `float32[]`; this plan's Task 1
action text specifies `float64[] p_des`, so that instruction wins here."""

# --- Verbatim extractions (real recordings were present on this machine) ---
# Source: R2026A63W0A9900005_20260904065700769_RAW.mcap, Schema id 1
# (teleoperation_msg_ros2/msg/VrPose) and Schema id 15
# (system_manager_msg/msg/TaskTakeoverStatus), read via a one-off
# `mcap.stream_reader.StreamReader(f, validate_crcs=False).records` walk
# collecting `schema.data.decode()` for those two schema names. Reproduced
# byte-for-byte below (comments included) — INGEST-01 names both packages
# explicitly, and neither has a single message in any of the 3 real samples
# (`/sysmgr/task_takeover_status` measures 0 messages in all three —
# DERISK-02), so this fixture is the *only* place the decode claim is ever
# provable.

SCHEMA_NAME_VR_POSE = "teleoperation_msg_ros2/msg/VrPose"
TOPIC_VR_POSE = "/vr_pose"
"""The real recordings' own topic name for this schema — confirmed by direct
Channel-record enumeration against the real sample MCAP, not guessed."""

SCHEMA_TEXT_VR_POSE = """# VrPose
std_msgs/Header header
bool power
float32[] head_position
float32[] head_quaternion
float32[] left_position
float32[] left_quaternion
bool left_gripper_close
float32[] right_position
float32[] right_quaternion
bool right_gripper_close
float32[] left_j4j6_cmd
float32[] right_j4j6_cmd
bool left_disconnect_pressed
bool right_disconnect_pressed
int8 control_mode
int8 manipulation_mode
bool waist_right
bool waist_left
bool tilt_forward
bool tilt_backward
bool torso_up
bool torso_down
bool torso_forward
bool torso_backward
float32 x_vel
float32 y_vel
float32 angular_vel
bool breakpoint
bool exception
bool need_reset_to_init
bool engage_by_operator
bool disengage_by_operator
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id

================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
"""

SCHEMA_NAME_TASK_TAKEOVER_STATUS = "system_manager_msg/msg/TaskTakeoverStatus"
TOPIC_TASK_TAKEOVER_STATUS = "/sysmgr/task_takeover_status"
"""The real recordings' own topic name — confirmed by direct Channel-record
enumeration; also the exact name INGEST-01's plan text names by hand and the
one PROV-02 calls out as an unmapped-but-inventoried stream."""

SCHEMA_TEXT_TASK_TAKEOVER_STATUS = """std_msgs/Header header
string takeover
string task_id
string task_result
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.

# Two-integer timestamp that is expressed as seconds and nanoseconds.
builtin_interfaces/Time stamp

# Transform frame with which this data is associated.
string frame_id

================================================================================
MSG: builtin_interfaces/Time
# This message communicates ROS Time defined here:
# https://design.ros2.org/articles/clock_and_time.html

# The seconds component, valid over all int32 values.
int32 sec

# The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
# e.g.
# The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
# The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
uint32 nanosec
"""


# ---------------------------------------------------------------------------
# Tiny real-JPEG camera payload, encoded once and reused.
# ---------------------------------------------------------------------------


def _make_tiny_jpeg(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Encodes a tiny 4x4 solid-color RGB image to real JPEG bytes via
    `simplejpeg` (the seam `data/decoders.py` already uses to decode), so
    `CompressedImageRef.decode()` in the tracer test produces real pixels,
    not a stub."""
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :] = color
    return simplejpeg.encode_jpeg(pixels, colorspace="rgb")


_TINY_JPEG_BYTES = _make_tiny_jpeg()


# ---------------------------------------------------------------------------
# Per-topic message specs and the builder itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicSpec:
    """One topic's channel + messages for `build_fixture_mcap`.

    `messages` is a list of `(log_time_ns, value_dict)` pairs. `value_dict`
    is handed straight to `mcap_ros2.writer.Writer.write_message` as the
    message object — `mcap_ros2`'s own `_get_property` falls back to
    `obj[name]` for anything without a matching attribute, so a plain nested
    dict works with no `SimpleNamespace` wrapper needed.
    """

    schema_name: str
    schema_text: str
    messages: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    encoding: str = "ros2msg"
    """Deliberately overridable: Task 3's schema-decodability fatal check
    test writes a channel whose schema encoding is NOT `ros2msg` (e.g.
    `ros2idl`) to prove the check rejects it by name rather than
    reinterpreting it as another valid type."""


def _header_value(log_time_ns: int, frame_id: str = "") -> dict[str, Any]:
    return {
        "stamp": {"sec": log_time_ns // 1_000_000_000, "nanosec": log_time_ns % 1_000_000_000},
        "frame_id": frame_id,
    }


def joint_state_messages(
    *,
    count: int,
    positions: list[list[float]] | None = None,
    start_ns: int = 0,
    step_ns: int = 40_000_000,
    joint_count: int = 7,
    name: str | None = None,
    header_stamp_offset_ns: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    """Builds `count` `sensor_msgs/msg/JointState` messages.

    `positions` overrides the default deterministic per-sample position
    array (`positions[i][j] = 1.0 + i * 0.1 + j * 0.01`) — pass an explicit
    list to pin exact numbers a test asserts against, or to write a length
    that deliberately mismatches a profile's declared `joint_names`.

    `header_stamp_offset_ns` (default 0, additive — every existing call site
    is unaffected) offsets `header.stamp` from the MCAP record's own
    `log_time` (the `ts` this function passes to `write_message`) by exactly
    this many nanoseconds: `header.stamp` encodes `ts +
    header_stamp_offset_ns`, never `ts` itself. This is the independent
    two-clock control plan 03-11's clock-skew fixture needs — a message's
    `header.stamp` (publish time) and the MCAP record `log_time` (recorder
    write time) are otherwise always numerically identical in every fixture
    this module builds.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    for i in range(count):
        ts = start_ns + i * step_ns
        pos = (
            positions[i]
            if positions is not None
            else [1.0 + i * 0.1 + j * 0.01 for j in range(joint_count)]
        )
        out.append(
            (
                ts,
                {
                    "header": _header_value(ts + header_stamp_offset_ns),
                    "name": [name] if name else [],
                    "position": pos,
                    "velocity": [],
                    "effort": [],
                },
            )
        )
    return out


def motor_control_messages(
    *,
    count: int,
    p_des: list[list[float]] | None = None,
    start_ns: int = 0,
    step_ns: int = 40_000_000,
    joint_count: int = 7,
    header_stamp_offset_ns: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    """Builds `count` `hdas_msg/msg/MotorControl` messages.

    `header_stamp_offset_ns` — see `joint_state_messages`'s docstring; same
    additive, default-0 two-clock control.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    for i in range(count):
        ts = start_ns + i * step_ns
        values = (
            p_des[i]
            if p_des is not None
            else [2.0 + i * 0.1 + j * 0.01 for j in range(joint_count)]
        )
        out.append(
            (
                ts,
                {
                    "header": _header_value(ts + header_stamp_offset_ns),
                    "name": "",
                    "p_des": values,
                },
            )
        )
    return out


def camera_messages(
    *,
    count: int,
    jpeg_bytes: bytes | list[bytes] | None = None,
    start_ns: int = 0,
    step_ns: int = 40_000_000,
    fmt: str = "jpeg",
) -> list[tuple[int, dict[str, Any]]]:
    """Builds `count` `sensor_msgs/msg/CompressedImage` messages.

    `jpeg_bytes` as a single `bytes` value repeats the SAME payload for
    every message — the INTEG-03 frozen/duplicate-frame fixture. A list
    supplies one payload per message.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    for i in range(count):
        ts = start_ns + i * step_ns
        if isinstance(jpeg_bytes, list):
            payload = jpeg_bytes[i]
        elif jpeg_bytes is not None:
            payload = jpeg_bytes
        else:
            payload = _TINY_JPEG_BYTES
        out.append((ts, {"header": _header_value(ts), "format": fmt, "data": payload}))
    return out


def vr_pose_messages(
    *, count: int, control_mode: int = 3, start_ns: int = 0, step_ns: int = 40_000_000
) -> list[tuple[int, dict[str, Any]]]:
    """Builds `count` `teleoperation_msg_ros2/msg/VrPose` messages. Asserted
    field for the INGEST-01 proof is `control_mode` (int8, trivially exact)."""
    out: list[tuple[int, dict[str, Any]]] = []
    for i in range(count):
        ts = start_ns + i * step_ns
        out.append(
            (
                ts,
                {
                    "header": _header_value(ts),
                    "power": True,
                    "head_position": [0.0, 0.0, 0.0],
                    "head_quaternion": [0.0, 0.0, 0.0, 1.0],
                    "left_position": [0.0, 0.0, 0.0],
                    "left_quaternion": [0.0, 0.0, 0.0, 1.0],
                    "left_gripper_close": False,
                    "right_position": [0.0, 0.0, 0.0],
                    "right_quaternion": [0.0, 0.0, 0.0, 1.0],
                    "right_gripper_close": False,
                    "left_j4j6_cmd": [],
                    "right_j4j6_cmd": [],
                    "left_disconnect_pressed": False,
                    "right_disconnect_pressed": False,
                    "control_mode": control_mode,
                    "manipulation_mode": 0,
                    "waist_right": False,
                    "waist_left": False,
                    "tilt_forward": False,
                    "tilt_backward": False,
                    "torso_up": False,
                    "torso_down": False,
                    "torso_forward": False,
                    "torso_backward": False,
                    "x_vel": 0.0,
                    "y_vel": 0.0,
                    "angular_vel": 0.0,
                    "breakpoint": False,
                    "exception": False,
                    "need_reset_to_init": False,
                    "engage_by_operator": False,
                    "disengage_by_operator": False,
                },
            )
        )
    return out


def task_takeover_status_messages(
    *, count: int, task_id: str = "fixture-task-0", start_ns: int = 0, step_ns: int = 40_000_000
) -> list[tuple[int, dict[str, Any]]]:
    """Builds `count` `system_manager_msg/msg/TaskTakeoverStatus` messages.
    Asserted field for the INGEST-01 proof is `task_id` (string, trivially
    exact)."""
    out: list[tuple[int, dict[str, Any]]] = []
    for i in range(count):
        ts = start_ns + i * step_ns
        out.append(
            (
                ts,
                {
                    "header": _header_value(ts),
                    "takeover": "false",
                    "task_id": task_id,
                    "task_result": "unknown",
                },
            )
        )
    return out


def varying_joint_positions(
    *, count: int, joint_count: int, seed: int, base: float
) -> list[list[float]]:
    """A seeded random walk in exactly the `positions=`/`p_des=` shape
    `joint_state_messages`/`motor_control_messages` already accept: shape
    `(count, joint_count)`.

    Magnitudes here are load-bearing, not arbitrary. `_carries_variance`
    (`data/validation/correlation.py`) requires a channel's centered norm to
    exceed `_VARIANCE_NOISE_FLOOR * max(|raw|max, 1.0) * sqrt(n)`. This
    walk's per-step increments are `uniform(-0.2, 0.2)` against a `base`
    magnitude of order 1-5, so the true per-timestep delta clears that
    `1e-4`-relative floor by roughly three orders of magnitude. This
    module's OTHER default motion model — `joint_state_messages`'s own
    linear ramp `1.0 + i*0.1 + j*0.01` — has a per-timestep delta that is a
    CONSTANT (the ramp's own slope), which after the real synchronizer's
    float32 round-trip clears that same floor by none: it is exactly the
    degenerate, unscoreable signal `check_action_state_lag`'s
    `_VARIANCE_NOISE_FLOOR` docstring measures. That contrast — not an
    arbitrary choice of numbers — is why this function exists as a second,
    genuinely-varying motion model alongside the ramp.
    """
    rng = np.random.default_rng(seed)
    increments = rng.uniform(-0.2, 0.2, size=(count, joint_count))
    walk = np.cumsum(increments, axis=0)
    offsets = np.array([base + j * 0.05 for j in range(joint_count)])
    return (walk + offsets).tolist()


def achieved_delta_positions(positions: list[list[float]]) -> list[list[float]]:
    """The per-sample achieved delta of a position series: `d[i] =
    positions[i+1] - positions[i]`, with the final row all zeros.

    The final row's value is irrelevant by construction:
    `check_action_state_lag` compares `np.diff(state)` against
    `action[:-1]` — the action series' own last row is trimmed away before
    any comparison, so whatever this function puts there can never affect
    the verdict.
    """
    arr = np.asarray(positions, dtype=np.float64)
    delta = np.zeros_like(arr)
    if arr.shape[0] > 1:
        delta[:-1] = np.diff(arr, axis=0)
    return delta.tolist()


def varying_motion_joint_topics(
    *, joint_samples: int, seed: int = 7
) -> dict[str, TopicSpec]:
    """The joint-topic-only override map for a genuinely varying (non-ramp)
    motion model, routed through the real adapter/synchronizer/writer path.

    Returns exactly 12 entries — the six `/hdas/feedback_*` state topics
    plus the six action-source topics `default_topics` also declares — and
    no camera topics, no `/vr_pose`, no
    `/sysmgr/task_takeover_status`, so a caller can merge this map into
    `build_fixture_mcap`'s `topics` argument alongside independent camera
    overrides without collision.

    State side: each of the six state topics gets its own independent
    random walk (`varying_joint_positions` with `seed=seed+i`,
    `base=1.0 + i*0.3`, in the fixed order `arm_left`, `arm_right`,
    `torso`, `gripper_left`, `gripper_right`, `chassis`). `torso` and
    `chassis` vary too even though neither is in the shipped profile's
    `action_primary` — so nothing about this fixture's non-degeneracy
    depends on which keys happen to be action-aligned.

    Action side: each populated action topic carries
    `achieved_delta_positions` of its PAIRED state topic's own series —
    the literal VALID-03 pairing (`action[t] == state[t+1] - state[t]`).
    `/motion_target/target_joint_state_arm_left` stays a `TopicSpec` with
    an EMPTY message list: all three real recordings measure zero messages
    on it (DERISK-02), and `default_topics` already encodes that reality —
    preserving it here means the left arm still resolves through its
    `/motion_control/control_arm_left` fallback, exactly as in production.
    `/motion_target/target_joint_state_arm_right` is populated (mirroring
    the real recordings, where that topic IS populated for the right arm)
    and is what the adapter's ordered-fallback resolution actually picks
    for `right_arm` — `/motion_control/control_arm_right` is also
    populated, from the same series, so the coupling holds whichever
    source resolves.
    """
    channel_defs: list[tuple[str, int, str, int]] = [
        ("/hdas/feedback_arm_left", 7, "left_arm", 0),
        ("/hdas/feedback_arm_right", 7, "right_arm", 1),
        ("/hdas/feedback_torso", 4, "torso", 2),
        ("/hdas/feedback_gripper_left", 1, "left_gripper", 3),
        ("/hdas/feedback_gripper_right", 1, "right_gripper", 4),
        ("/hdas/feedback_chassis", 3, "chassis", 5),
    ]

    specs: dict[str, TopicSpec] = {}
    positions_by_key: dict[str, list[list[float]]] = {}
    for topic, joint_count, key, i in channel_defs:
        positions = varying_joint_positions(
            count=joint_samples, joint_count=joint_count, seed=seed + i, base=1.0 + i * 0.3
        )
        positions_by_key[key] = positions
        specs[topic] = TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(
                count=joint_samples, positions=positions, joint_count=joint_count, name=key
            ),
        )

    left_arm_delta = achieved_delta_positions(positions_by_key["left_arm"])
    right_arm_delta = achieved_delta_positions(positions_by_key["right_arm"])
    left_gripper_delta = achieved_delta_positions(positions_by_key["left_gripper"])
    right_gripper_delta = achieved_delta_positions(positions_by_key["right_gripper"])

    specs["/motion_target/target_joint_state_arm_left"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE, SCHEMA_TEXT_JOINT_STATE, []
    )
    specs["/motion_control/control_arm_left"] = TopicSpec(
        SCHEMA_NAME_MOTOR_CONTROL,
        SCHEMA_TEXT_MOTOR_CONTROL,
        motor_control_messages(count=joint_samples, p_des=left_arm_delta, joint_count=7),
    )
    specs["/motion_target/target_joint_state_arm_right"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=right_arm_delta,
            joint_count=7,
            name="right_arm_target",
        ),
    )
    specs["/motion_control/control_arm_right"] = TopicSpec(
        SCHEMA_NAME_MOTOR_CONTROL,
        SCHEMA_TEXT_MOTOR_CONTROL,
        motor_control_messages(count=joint_samples, p_des=right_arm_delta, joint_count=7),
    )
    specs["/motion_target/target_position_gripper_left"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=left_gripper_delta,
            joint_count=1,
            name="left_gripper_target",
        ),
    )
    specs["/motion_target/target_position_gripper_right"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=right_gripper_delta,
            joint_count=1,
            name="right_gripper_target",
        ),
    )

    return specs


def leading_target_joint_topics(
    *, joint_samples: int, seed: int = 7
) -> dict[str, TopicSpec]:
    """The joint-topic-only override map for an ABSOLUTE-POSITION-TARGET
    action model (VALID-03 Option B, plan 03-11): each populated action
    topic carries its paired state topic's OWN position series exactly —
    `action[t] == state[t]` — a genuinely varying (random-walk),
    non-degenerate, absolute-position-tracking pairing. This is the
    LITERAL Option B pairing `check_action_state_lag` now applies
    (`data/validation/correlation.py::tracking_error_lag` compares action
    against RAW state, never against state's delta), as opposed to
    `varying_motion_joint_topics`'s achieved-DELTA pairing
    (`action[t] == state[t+1] - state[t]`) that plan 03-10 built for the
    now-superseded delta-vs-delta discrimination proof — that function is
    left completely unchanged so its own already-committed tests keep
    their exact meaning.

    Same 12-topic shape as `varying_motion_joint_topics` (six state
    topics, six action-source topics, no camera/vr_pose/task-takeover
    topics), same per-topic random-walk state model
    (`varying_joint_positions`, identical seeds/bases) — only the ACTION
    series' construction differs, so the two functions are directly
    comparable.
    """
    channel_defs: list[tuple[str, int, str, int]] = [
        ("/hdas/feedback_arm_left", 7, "left_arm", 0),
        ("/hdas/feedback_arm_right", 7, "right_arm", 1),
        ("/hdas/feedback_torso", 4, "torso", 2),
        ("/hdas/feedback_gripper_left", 1, "left_gripper", 3),
        ("/hdas/feedback_gripper_right", 1, "right_gripper", 4),
        ("/hdas/feedback_chassis", 3, "chassis", 5),
    ]

    specs: dict[str, TopicSpec] = {}
    positions_by_key: dict[str, list[list[float]]] = {}
    for topic, joint_count, key, i in channel_defs:
        positions = varying_joint_positions(
            count=joint_samples, joint_count=joint_count, seed=seed + i, base=1.0 + i * 0.3
        )
        positions_by_key[key] = positions
        specs[topic] = TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(
                count=joint_samples, positions=positions, joint_count=joint_count, name=key
            ),
        )

    left_arm_target = positions_by_key["left_arm"]
    right_arm_target = positions_by_key["right_arm"]
    left_gripper_target = positions_by_key["left_gripper"]
    right_gripper_target = positions_by_key["right_gripper"]

    specs["/motion_target/target_joint_state_arm_left"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE, SCHEMA_TEXT_JOINT_STATE, []
    )
    specs["/motion_control/control_arm_left"] = TopicSpec(
        SCHEMA_NAME_MOTOR_CONTROL,
        SCHEMA_TEXT_MOTOR_CONTROL,
        motor_control_messages(count=joint_samples, p_des=left_arm_target, joint_count=7),
    )
    specs["/motion_target/target_joint_state_arm_right"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=right_arm_target,
            joint_count=7,
            name="right_arm_target",
        ),
    )
    specs["/motion_control/control_arm_right"] = TopicSpec(
        SCHEMA_NAME_MOTOR_CONTROL,
        SCHEMA_TEXT_MOTOR_CONTROL,
        motor_control_messages(count=joint_samples, p_des=right_arm_target, joint_count=7),
    )
    specs["/motion_target/target_position_gripper_left"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=left_gripper_target,
            joint_count=1,
            name="left_gripper_target",
        ),
    )
    specs["/motion_target/target_position_gripper_right"] = TopicSpec(
        SCHEMA_NAME_JOINT_STATE,
        SCHEMA_TEXT_JOINT_STATE,
        joint_state_messages(
            count=joint_samples,
            positions=right_gripper_target,
            joint_count=1,
            name="right_gripper_target",
        ),
    )

    return specs


_CAMERA_TOPIC_NAMES = (
    "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "/hdas/camera_wrist_left/color/image_raw/compressed",
    "/hdas/camera_wrist_right/color/image_raw/compressed",
)
"""The four camera topics this module declares — the master (head_left),
head_right, wrist_left and wrist_right cameras of the shipped Galaxea
profile. Named here so `default_topics`' `camera_frames_by_topic` override
can validate/iterate against a single authoritative list rather than four
separately-typed dict literals drifting apart."""


def default_topics(
    *,
    camera_frames: int,
    joint_samples: int,
    camera_frames_by_topic: dict[str, int] | None = None,
) -> dict[str, TopicSpec]:
    """The full default fixture: every topic the shipped Galaxea profile
    declares required, plus the four camera topics, plus the two extra
    unmapped custom-package channels INGEST-01 names. Every value is
    deterministic so a test can assert exact numbers.

    `arm_left`'s preferred action source
    (`/motion_target/target_joint_state_arm_left`) is written with **zero**
    messages by default, matching the measured reality in all three real
    samples (DERISK-02) — `/motion_control/control_arm_left` is the only
    populated path by default, exactly like the real recordings.

    `camera_frames_by_topic` (default `None`, additive keyword — every
    existing call site is unaffected) overrides ONE OR MORE of the four
    camera topics' own frame count instead of `camera_frames` — e.g.
    `{"/hdas/camera_wrist_left/color/image_raw/compressed": 8}` gives that
    one camera 8 frames while the other three keep `camera_frames`. Each
    overridden topic still uses `camera_messages`' own default `step_ns`/
    `start_ns`, so a shorter camera's stream genuinely ends earlier in
    time rather than being sampled at a different rate (G-01/VERIFY-01,
    04-04-PLAN.md). `None` reproduces today's behaviour byte-for-byte.
    """
    camera_frame_counts = {topic: camera_frames for topic in _CAMERA_TOPIC_NAMES}
    if camera_frames_by_topic:
        camera_frame_counts.update(camera_frames_by_topic)

    return {
        # --- state feedback topics ---
        "/hdas/feedback_arm_left": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=7, name="left_arm"),
        ),
        "/hdas/feedback_arm_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=7, name="right_arm"),
        ),
        "/hdas/feedback_torso": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=4, name="torso"),
        ),
        "/hdas/feedback_gripper_left": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=1, name="left_gripper"),
        ),
        "/hdas/feedback_gripper_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=1, name="right_gripper"),
        ),
        "/hdas/feedback_chassis": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=3, name="chassis"),
        ),
        # --- action source topics (ordered fallback per field) ---
        "/motion_target/target_joint_state_arm_left": TopicSpec(
            SCHEMA_NAME_JOINT_STATE, SCHEMA_TEXT_JOINT_STATE, []
        ),  # measured 0 msgs in all 3 real samples (DERISK-02) — default matches
        "/motion_control/control_arm_left": TopicSpec(
            SCHEMA_NAME_MOTOR_CONTROL,
            SCHEMA_TEXT_MOTOR_CONTROL,
            motor_control_messages(count=joint_samples, joint_count=7),
        ),
        "/motion_target/target_joint_state_arm_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=7, name="right_arm_target"),
        ),
        "/motion_control/control_arm_right": TopicSpec(
            SCHEMA_NAME_MOTOR_CONTROL,
            SCHEMA_TEXT_MOTOR_CONTROL,
            motor_control_messages(count=joint_samples, joint_count=7),
        ),
        "/motion_target/target_position_gripper_left": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=1, name="left_gripper_target"),
        ),
        "/motion_target/target_position_gripper_right": TopicSpec(
            SCHEMA_NAME_JOINT_STATE,
            SCHEMA_TEXT_JOINT_STATE,
            joint_state_messages(count=joint_samples, joint_count=1, name="right_gripper_target"),
        ),
        # --- camera topics ---
        "/hdas/camera_head/left_raw/image_raw_color/compressed": TopicSpec(
            SCHEMA_NAME_COMPRESSED_IMAGE,
            SCHEMA_TEXT_COMPRESSED_IMAGE,
            camera_messages(
                count=camera_frame_counts["/hdas/camera_head/left_raw/image_raw_color/compressed"]
            ),
        ),
        "/hdas/camera_head/right_raw/image_raw_color/compressed": TopicSpec(
            SCHEMA_NAME_COMPRESSED_IMAGE,
            SCHEMA_TEXT_COMPRESSED_IMAGE,
            camera_messages(
                count=camera_frame_counts[
                    "/hdas/camera_head/right_raw/image_raw_color/compressed"
                ]
            ),
        ),
        "/hdas/camera_wrist_left/color/image_raw/compressed": TopicSpec(
            SCHEMA_NAME_COMPRESSED_IMAGE,
            SCHEMA_TEXT_COMPRESSED_IMAGE,
            camera_messages(
                count=camera_frame_counts["/hdas/camera_wrist_left/color/image_raw/compressed"]
            ),
        ),
        "/hdas/camera_wrist_right/color/image_raw/compressed": TopicSpec(
            SCHEMA_NAME_COMPRESSED_IMAGE,
            SCHEMA_TEXT_COMPRESSED_IMAGE,
            camera_messages(
                count=camera_frame_counts["/hdas/camera_wrist_right/color/image_raw/compressed"]
            ),
        ),
        # --- extra unmapped custom-package channels (INGEST-01 / PROV-02) ---
        TOPIC_VR_POSE: TopicSpec(
            SCHEMA_NAME_VR_POSE, SCHEMA_TEXT_VR_POSE, vr_pose_messages(count=1)
        ),
        TOPIC_TASK_TAKEOVER_STATUS: TopicSpec(
            SCHEMA_NAME_TASK_TAKEOVER_STATUS,
            SCHEMA_TEXT_TASK_TAKEOVER_STATUS,
            task_takeover_status_messages(count=1),
        ),
    }


def build_fixture_mcap(
    path: str | Path,
    *,
    camera_frames: int = 5,
    joint_samples: int = 5,
    topics: dict[str, TopicSpec | None] | None = None,
    camera_frames_by_topic: dict[str, int] | None = None,
) -> None:
    """Writes a synthetic MCAP to `path` carrying the same `ros2msg` schema
    text the real Galaxea recordings carry.

    `topics`, when given, overrides `default_topics(camera_frames,
    joint_samples, camera_frames_by_topic)` entry-by-entry: a key mapped to
    a `TopicSpec` replaces that topic's default spec; a key mapped to
    `None` omits the channel entirely (simulating "topic absent from the
    channel list", INGEST-03). Topics not mentioned in `topics` keep their
    default spec — including one whose per-topic count was overridden via
    `camera_frames_by_topic` below.

    `camera_frames_by_topic` (default `None`, additive keyword) is forwarded
    straight to `default_topics` — see that function's own docstring. `None`
    reproduces today's behaviour byte-for-byte.

    Written with `compression=CompressionType.NONE` — a deliberate choice
    (not a real-recording property): it makes `corrupt_chunk_crc`'s
    single-byte flip land in the RAW chunk payload with no zstd-decompression
    step in between, so the CRC mismatch this fixture exists to trigger is
    deterministic rather than incidentally depending on where a corrupted
    zstd frame happens to fail.
    """
    merged = dict(
        default_topics(
            camera_frames=camera_frames,
            joint_samples=joint_samples,
            camera_frames_by_topic=camera_frames_by_topic,
        )
    )
    if topics:
        for topic, spec in topics.items():
            if spec is None:
                merged.pop(topic, None)
            else:
                merged[topic] = spec

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "wb") as f:
        writer = Ros2Writer(f, compression=CompressionType.NONE)
        schema_ids: dict[tuple[str, str, str], int] = {}
        channel_ids: dict[str, int] = {}
        for topic, spec in merged.items():
            if spec.encoding == "ros2msg":
                # The normal path: mcap_ros2.writer.Writer owns schema
                # registration, channel registration, and CDR encoding.
                # `write_message` registers a channel LAZILY on its first
                # call for a topic — so a topic with zero messages would
                # otherwise get no Channel record at all, making "present
                # with zero messages" indistinguishable from "absent from
                # the channel list" (exactly the INGEST-03 distinction
                # Task 2's tests need). Force the channel to exist
                # regardless of message count by registering it directly
                # through the wrapped low-level writer and pre-seeding
                # `write_message`'s own channel cache, so the normal
                # `write_message` path below still works unchanged for any
                # messages this topic does have.
                schema = writer.register_msgdef(spec.schema_name, spec.schema_text)
                channel_id = writer._writer.register_channel(  # noqa: SLF001 - documented above
                    topic=topic, message_encoding="cdr", schema_id=schema.id
                )
                writer._channel_ids[topic] = channel_id  # noqa: SLF001
                for log_time_ns, value in spec.messages:
                    writer.write_message(topic, schema, value, log_time=log_time_ns)
                continue

            # Task 3's schema-decodability fatal check needs a schema whose
            # encoding mcap-ros2-support cannot build a decoder for.
            # mcap_ros2.writer.Writer hardcodes SchemaEncoding.ROS2 in both
            # register_msgdef AND write_message (the latter refuses to
            # encode against a non-ROS2 schema at all) — so this whole
            # branch drops to the wrapped low-level `mcap.writer.Writer`
            # directly and writes raw placeholder bytes. What the payload
            # bytes decode to is irrelevant: the check this exists to prove
            # rejects the schema by its encoding, before any payload is
            # ever touched.
            cache_key = (spec.schema_name, spec.schema_text, spec.encoding)
            schema_id = schema_ids.get(cache_key)
            if schema_id is None:
                schema_id = writer._writer.register_schema(  # noqa: SLF001 - deliberate low-level escape, documented above
                    spec.schema_name, spec.encoding, spec.schema_text.encode()
                )
                schema_ids[cache_key] = schema_id
            channel_id = channel_ids.get(topic)
            if channel_id is None:
                channel_id = writer._writer.register_channel(  # noqa: SLF001
                    topic=topic, message_encoding="cdr", schema_id=schema_id
                )
                channel_ids[topic] = channel_id
            for log_time_ns, _value in spec.messages:
                writer._writer.add_message(  # noqa: SLF001
                    channel_id=channel_id,
                    log_time=log_time_ns,
                    publish_time=log_time_ns,
                    sequence=0,
                    data=b"\x00",
                )
        writer.finish()


# ---------------------------------------------------------------------------
# Chunk-CRC corruption for the INTEG-01 CRC regression test.
# ---------------------------------------------------------------------------

_MAGIC_SIZE = 8
_OPCODE_CHUNK = 0x06


def corrupt_chunk_crc(path: str | Path) -> int:
    """Flips one byte inside the first Chunk record's raw payload, so the
    file's recorded `uncompressed_crc` no longer matches.

    Walks the file's own top-level opcode+length record framing by hand
    (mirroring `mcap.stream_reader.StreamReader`'s traversal exactly, per
    `mcap.records.Chunk.read`'s documented field layout) to find that byte's
    exact offset — never guesses an offset. Returns the byte offset changed.

    Requires the fixture to have been written with `compression=NONE` (the
    `build_fixture_mcap` default): with no compression, the flipped byte
    changes the CRC-checked content directly, with no zstd-decompression
    step that could turn the corruption into an unrelated decompression
    error instead of the intended `CRCValidationError`.
    """
    resolved = Path(path)
    data = bytearray(resolved.read_bytes())

    offset = _MAGIC_SIZE
    chunk_body_start: int | None = None
    while offset < len(data):
        opcode = data[offset]
        (length,) = struct.unpack_from("<Q", data, offset + 1)
        body_start = offset + 9
        if opcode == _OPCODE_CHUNK:
            chunk_body_start = body_start
            break
        offset = body_start + length

    if chunk_body_start is None:
        raise ValueError(f"{resolved}: no Chunk record found to corrupt")

    # Chunk body: start_time(8) end_time(8) uncompressed_size(8)
    # uncompressed_crc(4) compression_len(4) compression(bytes) data_len(8) data(bytes)
    pos = chunk_body_start + 8 + 8 + 8 + 4
    (compression_len,) = struct.unpack_from("<I", data, pos)
    pos += 4 + compression_len
    (data_len,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    if data_len == 0:
        raise ValueError(f"{resolved}: chunk has zero-length data, nothing to corrupt")

    flip_offset = pos
    data[flip_offset] ^= 0xFF
    resolved.write_bytes(bytes(data))
    return flip_offset
