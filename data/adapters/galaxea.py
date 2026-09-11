"""Galaxea R1 Pro MCAP input adapter.

Two-pass design per D-01 (02-CONTEXT.md), split by check cost, not by
concern:

1. A **CRC-validating index pass** (`_run_index_pass`) — a full sequential
   scan with `validate_crcs=True` (never `inspect_mcap.py`'s CRC-validation
   left disabled for preflight speed — see that module's own docstring
   warning), reusing only the enumeration *shape* of
   `inspect_mcap.py`'s `_scan_mcap_records` (`isinstance` dispatch over
   `Schema`/`Channel`/`Header` records), never that function itself. No
   general message deserialization happens here; camera-frame "payload"
   digests hash just the `data` field of a raw, still-CDR-encoded
   `CompressedImage` message — extracted via `_extract_compressed_image_payload`
   using `mcap_ros2`'s low-level `CdrReader` primitive against
   `CompressedImage`'s fixed, known field layout, never the general
   schema-driven `generate_dynamic()` decoder — see that function's
   docstring for why hashing the whole raw message (this module's first
   draft) was wrong: a real camera message's `header.stamp` differs on
   every frame even when the image bytes themselves repeat, which would
   make every digest unique and silently defeat frozen-frame detection.
2. A **streaming decode pass** (`_run_decode_pass`) over the module-level
   `iter_decoded_messages` — this engine's single decode entry point, and
   the one INGEST-01's three-package proof calls directly.

`GalaxeaInputAdapter` satisfies `BaseInputAdapter` (data/contracts/protocols.py)
structurally — this module never imports from, nor inherits, that Protocol.
"""

from dataclasses import dataclass, replace
from typing import Any, Iterator

import numpy as np
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2._cdr import CdrReader
from mcap_ros2.reader import read_ros2_messages

from data.adapters.checks import (
    AdvisoryCheckContext,
    CheckRegistry,
    FatalCheckContext,
    JointSeries,
    compute_payload_digest,
)
from data.contracts.containers import (
    CanonicalEpisode,
    Disposition,
    EpisodeHeader,
    StreamInfo,
)
from data.contracts.profile import EmbodimentProfile, profile_hash
from data.decoders import get_decoder_backend
from data.sync.master_clock import CAMERA_COVERAGE_CHECK_NAME, MasterClockSynchronizer

_ROS2_MSG_ENCODING = "ros2msg"
_COMPRESSED_IMAGE_SCHEMA = "sensor_msgs/msg/CompressedImage"


def _extract_compressed_image_payload(raw: bytes) -> bytes:
    """Extracts just the `data` field from a raw, still-CDR-encoded
    `sensor_msgs/msg/CompressedImage` message.

    Walks exactly the three fields ahead of `data` (`header.stamp.sec`,
    `header.stamp.nanosec`, `header.frame_id`, `format`) using
    `mcap_ros2`'s own low-level `CdrReader` primitive against
    `CompressedImage`'s fixed, known field layout — deliberately NOT the
    general schema-driven `generate_dynamic()` decoder, which builds a
    whole typed message object at real cost. This is the narrow read that
    keeps INTEG-03's frozen-frame check honest: hashing the *image* bytes,
    not a per-message header timestamp that differs even when a stalled
    driver republishes the exact same picture.
    """
    reader = CdrReader(raw)
    reader.int32()  # header.stamp.sec
    reader.uint32()  # header.stamp.nanosec
    reader.string()  # header.frame_id
    reader.string()  # format
    length = reader.sequence_length()
    return reader.uint8_array(length)


class GalaxeaAdapterError(Exception):
    """Raised for any structural failure reading or decoding a Galaxea
    MCAP — a truncated/corrupt file, a schema mcap-ros2-support cannot
    build a decoder for, a missing required topic, or a declared/measured
    joint-count mismatch. Every raise site names the file and/or topic
    involved; nothing here lets a bare library traceback escape to the
    caller.

    When raised because one or more **fatal** checks failed (never for any
    other structural failure — a file that could not even be opened, a
    missing required topic, a joint-count mismatch), this error carries
    `rejected_header`/`source_uri` so a caller can hand them straight to
    `ProvenanceRecorder.record_rejected(rejected_header, source_uri,
    episode_index)` (data/provenance/recorder.py) — the only durable
    evidence a fatally-aborted episode was ever seen (PROV-04). This is the
    "hook" that exists in Phase 2 even though the batch loop supplying
    `episode_index` is Phase 4's CLI; both attributes are `None` on every
    other raise site in this module.
    """

    def __init__(
        self,
        message: str,
        *,
        rejected_header: EpisodeHeader | None = None,
        source_uri: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rejected_header = rejected_header
        self.source_uri = source_uri


@dataclass(frozen=True, slots=True)
class CompressedImageRef:
    """One camera frame's compressed JPEG bytes, decoded only on demand.

    Satisfies `ImageRef` (data/contracts/containers.py) structurally — no
    import from, no inheritance of, that module. The compressed bytes travel
    with the frame; pixels exist only for as long as one caller holds the
    `decode()` result (INGEST-05).
    """

    data: bytes
    timestamp_ns: int

    def decode(self) -> np.ndarray:
        """Decodes the held JPEG bytes to an HxWx3 uint8 RGB array via the
        configured decoder backend (`data/decoders.py`) — never `simplejpeg`
        directly; that indirection is the whole point of the seam."""
        return get_decoder_backend().decode(self.data)


def iter_decoded_messages(path: str, topics: list[str]) -> Iterator[tuple[str, int, Any]]:
    """The engine's single module-level decode entry point.

    Wraps `mcap_ros2.reader.read_ros2_messages(path, topics=topics,
    log_time_order=True)` — a generator — in the house deliberately-broad
    exception boundary. This is the ONE call site every consumer (the
    adapter's own decode pass, and INGEST-01's three-package proof) shares;
    keeping it module-level rather than inlined in a private method is what
    makes the three-package decode claim provable against engine code
    instead of against the library directly, with no need to route through
    `parse()` (which only ever requests profile-declared topics).

    Yields `(topic, log_time_ns, decoded_ros2_message)`.
    """
    try:
        for msg in read_ros2_messages(path, topics=topics, log_time_order=True):
            yield msg.channel.topic, msg.log_time_ns, msg.ros_msg
    except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
        raise GalaxeaAdapterError(
            f"failed to decode messages from {path!r} (topics={topics}): "
            f"{type(e).__name__}: {e}"
        ) from e


@dataclass
class _IndexResult:
    """Everything the CRC-validating index pass learns about one MCAP file
    with zero payload deserialization."""

    channels_by_topic: dict[str, Channel]
    schema_by_topic: dict[str, Schema]
    schemas_by_id: dict[int, Schema]
    message_counts: dict[str, int]
    first_log_time_ns: dict[str, int]
    last_log_time_ns: dict[str, int]
    camera_payload_digests: dict[str, list[str]]
    non_monotonic_topics: frozenset[str]
    scan_error: str | None


def _run_index_pass(path: str) -> _IndexResult:
    """Full sequential scan of `path` with MCAP chunk/data CRC validation
    **enabled** (`validate_crcs=True`) — the one setting `inspect_mcap.py`'s
    otherwise-reusable `_scan_mcap_records` deliberately leaves disabled for
    preflight speed, and which this pass must NOT inherit: copying that
    default would make INTEG-01's "failed MCAP
    chunk/data CRC is fatal" check structurally unenforceable
    (02-RESEARCH.md Pitfall 1).

    No message payload is ever deserialized here. "Camera payload digest"
    is a deliberate simplification: it hashes the raw, still-CDR-encoded
    `Message.data` bytes for every topic whose schema is
    `sensor_msgs/msg/CompressedImage`, not a decoded image payload sub-field
    — extracting just the `data` field from raw CDR bytes without the
    general dynamic decoder would itself be a hand-rolled deserializer, and
    the header/format-string prefix ahead of the actual JPEG bytes is a
    vanishingly small fraction of a compressed-image message's total size,
    so hashing the whole raw message is a faithful proxy for "did this
    camera's stalled driver republish an identical buffer" (INTEG-03/D-02)
    at truly zero decode cost.

    Any error hit WHILE scanning (a CRC mismatch, a truncated chunk, or any
    other `StreamReader` failure) is captured into `scan_error` rather than
    raised — `checks.py`'s `metadata.mcap_crc` fatal check is what turns
    that into a named, evidence-carrying abort, so the check registry stays
    the one inspectable place a corrupt file is reported from (INTEG-01).
    Only a failure to even open the file (unrelated to CRC/parse integrity)
    raises here directly — there is no "check" for a file that never opened.
    """
    channels_by_id: dict[int, Channel] = {}
    schemas_by_id: dict[int, Schema] = {}
    channels_by_topic: dict[str, Channel] = {}
    message_counts: dict[str, int] = {}
    first_log_time_ns: dict[str, int] = {}
    last_log_time_ns: dict[str, int] = {}
    camera_payload_digests: dict[str, list[str]] = {}
    non_monotonic_topics: set[str] = set()
    scan_error: str | None = None

    try:
        with open(path, "rb") as f:
            try:
                reader = StreamReader(f, skip_magic=False, validate_crcs=True)
                for record in reader.records:
                    if isinstance(record, Schema):
                        schemas_by_id[record.id] = record
                    elif isinstance(record, Channel):
                        channels_by_id[record.id] = record
                        channels_by_topic[record.topic] = record
                    elif isinstance(record, Message):
                        channel = channels_by_id.get(record.channel_id)
                        if channel is None:
                            continue
                        topic = channel.topic
                        message_counts[topic] = message_counts.get(topic, 0) + 1
                        if topic not in first_log_time_ns:
                            first_log_time_ns[topic] = record.log_time
                        prev = last_log_time_ns.get(topic)
                        if prev is not None and record.log_time < prev:
                            non_monotonic_topics.add(topic)
                        last_log_time_ns[topic] = record.log_time
                        schema = schemas_by_id.get(channel.schema_id)
                        if schema is not None and schema.name == _COMPRESSED_IMAGE_SCHEMA:
                            try:
                                payload = _extract_compressed_image_payload(record.data)
                            except Exception:  # noqa: BLE001 - narrow-parse fallback
                                # A message that does not match
                                # CompressedImage's expected fixed layout
                                # (corrupt or unexpected) degrades to
                                # hashing the whole raw message rather than
                                # aborting the index pass over an
                                # advisory-only signal.
                                payload = record.data
                            camera_payload_digests.setdefault(topic, []).append(
                                compute_payload_digest(payload)
                            )
            except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
                scan_error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
        raise GalaxeaAdapterError(
            f"failed to open MCAP {path!r}: {type(e).__name__}: {e}"
        ) from e

    schema_by_topic: dict[str, Schema] = {}
    for topic, channel in channels_by_topic.items():
        schema = schemas_by_id.get(channel.schema_id)
        if schema is not None:
            schema_by_topic[topic] = schema

    return _IndexResult(
        channels_by_topic=channels_by_topic,
        schema_by_topic=schema_by_topic,
        schemas_by_id=schemas_by_id,
        message_counts=message_counts,
        first_log_time_ns=first_log_time_ns,
        last_log_time_ns=last_log_time_ns,
        camera_payload_digests=camera_payload_digests,
        non_monotonic_topics=frozenset(non_monotonic_topics),
        scan_error=scan_error,
    )


def _find_master_clock_camera(profile: EmbodimentProfile) -> tuple[str, Any]:
    """Returns `(name, CameraSpec)` for the profile's single
    `role: master_clock` camera. `EmbodimentProfile._validate_master_clock_camera`
    already guarantees exactly one exists — this never has to handle zero or
    multiple."""
    for name, cam in profile.cameras.items():
        if cam.role == "master_clock":
            return name, cam
    raise GalaxeaAdapterError(
        "profile declares no role: master_clock camera — should be unreachable, "
        "EmbodimentProfile validation guarantees exactly one"
    )  # pragma: no cover - guarded by pydantic validation before this runs


def _check_declared_length(topic: str, joint_names: list[str], measured: int) -> None:
    """INGEST-02's abort: the profile's declared `joint_names` length is the
    only per-joint ordering authority, and it must match the measured
    length of the topic's own configured value field exactly. Names the
    topic, the declared length, and the measured length — never a bare
    shape-mismatch traceback."""
    declared = len(joint_names)
    if declared != measured:
        raise GalaxeaAdapterError(
            f"topic {topic!r}: declared joint_names length {declared} does not "
            f"match measured value-field length {measured}"
        )


def _plan_decode_requests(
    profile: EmbodimentProfile, index: _IndexResult
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Resolves, from index-pass facts ALONE (no decode call made), which
    state/action topics `parse()` will actually decode — and raises
    `GalaxeaAdapterError` before any decode if a structural requirement
    (INGEST-02/03) is violated.

    Returns `(state_fields_to_decode, action_resolutions)`:
    - `state_fields_to_decode`: `{field_name: StateField}` for every state
      topic that is required, or optional-but-present-with-messages. An
      optional field with zero or absent messages is omitted — never
      decoded, never zero-filled into the emitted state.
    - `action_resolutions`: `{action_key: (ActionField, winning ActionSource)}`
      for every action field whose ordered `sources` produced a winner —
      the first source (in declared order) with `message_count > 0`. A
      non-required field with no winning source is omitted.

    On any required-topic violation, one `GalaxeaAdapterError` names every
    offending topic in sorted order, distinguishing "absent from the
    channel list" from "present but with zero messages" by wording.
    """
    missing_absent: list[str] = []
    missing_empty: list[str] = []

    state_fields_to_decode: dict[str, Any] = {}
    for field_name, state_field in profile.state_topics.items():
        topic = state_field.topic
        present = topic in index.channels_by_topic
        count = index.message_counts.get(topic, 0)
        if state_field.required:
            if not present:
                missing_absent.append(topic)
                continue
            if count == 0:
                missing_empty.append(topic)
                continue
            state_fields_to_decode[field_name] = state_field
        elif present and count > 0:
            state_fields_to_decode[field_name] = state_field
        # else: optional, absent-or-empty -> skip silently, never abort

    action_resolutions: dict[str, tuple[Any, Any]] = {}
    for action_key, action_field in profile.action_topics.items():
        winner = None
        for source in action_field.sources:
            if index.message_counts.get(source.topic, 0) > 0:
                winner = source
                break
        if winner is not None:
            action_resolutions[action_key] = (action_field, winner)
        elif action_field.required:
            source_topics = ", ".join(s.topic for s in action_field.sources)
            missing_empty.append(
                f"{action_key} action field (no messages on any of its ordered "
                f"sources: {source_topics})"
            )
        # else: optional with no winning source -> skip silently

    if missing_absent or missing_empty:
        parts = []
        if missing_absent:
            parts.append(
                "absent from the MCAP channel list: " + ", ".join(sorted(missing_absent))
            )
        if missing_empty:
            parts.append(
                "present but with zero messages: " + ", ".join(sorted(missing_empty))
            )
        raise GalaxeaAdapterError(
            "required topic(s) failed the presence/non-emptiness check before any "
            "payload decode — " + "; ".join(parts)
        )

    return state_fields_to_decode, action_resolutions


def _secondary_cameras_to_decode(
    profile: EmbodimentProfile, master_name: str
) -> dict[str, Any]:
    """Every profile-declared camera beyond the master-clock camera that is
    actually **emitted** — deciding to decode a camera whose `emit` is
    false would cost real bandwidth (a whole compressed-image buffer per
    frame) for zero downstream benefit, since the writer never looks up an
    unemitted `video_key` and `MasterClockSynchronizer` never reports one
    either. This activates SYNC-01..05's secondary-camera nearest-match
    capability (02-04) end to end through the real adapter — until this
    function existed, `_run_decode_pass` only ever requested the master
    camera's topic, so `wrist_left`/`wrist_right` were unreachable
    (02-04-SUMMARY.md's own documented gap)."""
    return {
        name: cam
        for name, cam in profile.cameras.items()
        if name != master_name and cam.emit
    }


def _run_decode_pass(
    path: str,
    *,
    master_name: str,
    master_cam: Any,
    secondary_cameras: dict[str, Any],
    state_fields: dict[str, Any],
    action_resolutions: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    """The streaming decode pass over every topic `_plan_decode_requests`
    resolved (plus every `secondary_cameras` topic): the master-clock
    camera, every emitted secondary camera, every requested state topic,
    and every action field's winning source topic.

    Returns the `raw_streams` shape `MasterClockSynchronizer.synchronize`
    consumes:
    ```
    {
        "cameras": {name: {"topic", "role", "video_key", "frames": [(ts, CompressedImageRef), ...]}},
        "state": {name: {"topic", "resample", "joint_names", "state_key", "samples": [(ts, np.ndarray), ...]}},
        "action": {action_key: {"topic", "resample", "joint_names", "samples": [(ts, np.ndarray), ...]}},
        "meta": {...},
    }
    ```
    Every non-master `cameras` entry omits `sync_policy`/`max_sync_gap_ms` —
    `MasterClockSynchronizer` falls back to `profile.cameras[<same key>]`'s
    own declared values for those, so this function need not duplicate them
    (02-04-SUMMARY.md's documented fallback contract).

    Buffers only `(log_time_ns, value)` pairs per topic — never a whole
    decoded message object, and never a whole episode's worth of frames
    across all cameras at a time (INGEST-05).
    """
    state_topic_map: dict[str, tuple[str, Any]] = {
        f.topic: (name, f) for name, f in state_fields.items()
    }
    action_topic_map: dict[str, tuple[str, Any, Any]] = {
        source.topic: (action_key, action_field, source)
        for action_key, (action_field, source) in action_resolutions.items()
    }
    camera_topic_map: dict[str, str] = {
        cam.topic: name for name, cam in secondary_cameras.items()
    }
    all_topics = [
        master_cam.topic,
        *camera_topic_map,
        *state_topic_map,
        *action_topic_map,
    ]

    cameras: dict[str, Any] = {
        master_name: {
            "topic": master_cam.topic,
            "role": "master_clock",
            "video_key": master_cam.video_key or master_name,
            "frames": [],
        }
    }
    for name, cam in secondary_cameras.items():
        cameras[name] = {
            "topic": cam.topic,
            "role": "secondary",
            "video_key": cam.video_key or name,
            "frames": [],
        }
    state: dict[str, Any] = {
        name: {
            "topic": f.topic,
            "resample": f.resample,
            "joint_names": f.joint_names,
            "state_key": f.state_key,
            "samples": [],
        }
        for name, f in state_fields.items()
    }
    action: dict[str, Any] = {
        action_key: {
            "topic": source.topic,
            "resample": action_field.resample,
            "joint_names": action_field.joint_names,
            "samples": [],
        }
        for action_key, (action_field, source) in action_resolutions.items()
    }

    checked_lengths: set[str] = set()

    for topic, ts, msg in iter_decoded_messages(path, all_topics):
        if topic == master_cam.topic:
            cameras[master_name]["frames"].append((ts, CompressedImageRef(bytes(msg.data), ts)))
            continue
        if topic in camera_topic_map:
            cam_name = camera_topic_map[topic]
            cameras[cam_name]["frames"].append((ts, CompressedImageRef(bytes(msg.data), ts)))
            continue
        if topic in state_topic_map:
            name, field_spec = state_topic_map[topic]
            values = np.asarray(getattr(msg, field_spec.field), dtype=np.float32)
            if topic not in checked_lengths:
                checked_lengths.add(topic)
                _check_declared_length(topic, field_spec.joint_names, values.shape[0])
            state[name]["samples"].append((ts, values))
            continue
        if topic in action_topic_map:
            action_key, action_field, source = action_topic_map[topic]
            values = np.asarray(getattr(msg, source.field), dtype=np.float32)
            if topic not in checked_lengths:
                checked_lengths.add(topic)
                _check_declared_length(topic, action_field.joint_names, values.shape[0])
            action[action_key]["samples"].append((ts, values))

    return {"cameras": cameras, "state": state, "action": action, "meta": {}}


def _joint_name_ordering(profile: EmbodimentProfile) -> dict[str, tuple[str, ...]]:
    """INGEST-06's reporting half: the per-topic joint-name ordering the
    adapter used, for every declared state and action field — keyed by the
    profile's own field name (the `state_topics`/`extra_state_topics`/
    `action_topics` dict key), including fields that were skipped (zero
    messages, optional) so an aborted or empty episode still records the
    ordering it would have used. No cross-episode comparison happens here
    — that is Phase 4's batch-abort half."""
    ordering: dict[str, tuple[str, ...]] = {}
    for name, state_field in profile.state_topics.items():
        ordering[name] = tuple(state_field.joint_names)
    for name, extra_field in profile.extra_state_topics.items():
        ordering[name] = tuple(extra_field.joint_names)
    for name, action_field in profile.action_topics.items():
        ordering[name] = tuple(action_field.joint_names)
    return ordering


def _emitted_topics(
    profile: EmbodimentProfile, action_resolutions: dict[str, tuple[Any, Any]]
) -> set[str]:
    """Topics whose values flow into the dataset the profile declares —
    driven by profile intent (`include_in_observation_state`, camera
    `emit`, and whether a resolved action key is in `action_primary`), not
    by which topics this adapter's current code path happens to buffer."""
    emitted: set[str] = set()
    for f in profile.state_topics.values():
        if f.include_in_observation_state:
            emitted.add(f.topic)
    for f in profile.extra_state_topics.values():
        if f.include_in_observation_state:
            emitted.add(f.topic)
    for action_key, (_field, source) in action_resolutions.items():
        if action_key in profile.action_primary:
            emitted.add(source.topic)
    for cam in profile.cameras.values():
        if cam.emit:
            emitted.add(cam.topic)
    return emitted


def _build_stream_inventory(
    index: _IndexResult,
    profile: EmbodimentProfile,
    action_resolutions: dict[str, tuple[Any, Any]],
) -> tuple[StreamInfo, ...]:
    """PROV-02's per-stream inventory: one `StreamInfo` row per channel the
    file contains — including channels the profile never maps (the
    camera-intrinsics/transform-tree/operator-takeover streams the Phase 1
    probe revealed, and this plan's two extra custom-package channels) —
    sorted by topic, with `measured_hz` left unset for a stream with fewer
    than two messages rather than reading as `0.0`."""
    emitted = _emitted_topics(profile, action_resolutions)
    rows: list[StreamInfo] = []
    for topic in sorted(index.channels_by_topic):
        schema = index.schema_by_topic.get(topic)
        count = index.message_counts.get(topic, 0)
        first = index.first_log_time_ns.get(topic)
        last = index.last_log_time_ns.get(topic)
        measured_hz: float | None = None
        if count >= 2 and first is not None and last is not None and last > first:
            measured_hz = (count - 1) / ((last - first) / 1e9)
        rows.append(
            StreamInfo(
                topic=topic,
                schema_name=schema.name if schema is not None else "",
                message_count=count,
                measured_hz=measured_hz,
                emitted=topic in emitted,
            )
        )
    return tuple(rows)


def _build_joint_series(
    raw_streams: dict[str, Any], profile: EmbodimentProfile
) -> tuple[JointSeries, ...]:
    """Builds the raw (pre-resample) per-field series `checks.py`'s
    advisory checks measure on — never the resampled grid (D-01's
    placement of advisory checks ahead of the synchronizer)."""
    series: list[JointSeries] = []
    for name, entry in raw_streams.get("state", {}).items():
        samples = sorted(entry["samples"], key=lambda pair: pair[0])
        if not samples:
            continue
        ts = np.array([t for t, _ in samples], dtype=np.int64)
        vals = np.stack([v for _, v in samples]).astype(np.float32)
        state_field = profile.state_topics.get(name)
        max_v = state_field.max_velocity if state_field is not None else None
        max_a = state_field.max_acceleration if state_field is not None else None
        series.append(JointSeries(entry["topic"], ts, vals, max_v, max_a))
    for entry in raw_streams.get("action", {}).values():
        samples = sorted(entry["samples"], key=lambda pair: pair[0])
        if not samples:
            continue
        ts = np.array([t for t, _ in samples], dtype=np.int64)
        vals = np.stack([v for _, v in samples]).astype(np.float32)
        # ActionField declares no max_velocity/max_acceleration (D-07: hard
        # physics limits live only on StateField) — always None here.
        series.append(JointSeries(entry["topic"], ts, vals, None, None))
    return tuple(series)


def _build_cross_stream_timestamps(raw_streams: dict[str, Any]) -> dict[str, np.ndarray]:
    """INTEG-02's skew measurement input: every decoded stream's own sorted
    timestamp array, keyed by topic."""
    out: dict[str, np.ndarray] = {}
    for cam in raw_streams.get("cameras", {}).values():
        ts = sorted(t for t, _ in cam["frames"])
        out[cam["topic"]] = np.array(ts, dtype=np.int64)
    for entry in raw_streams.get("state", {}).values():
        ts = sorted(t for t, _ in entry["samples"])
        if ts:
            out[entry["topic"]] = np.array(ts, dtype=np.int64)
    for entry in raw_streams.get("action", {}).values():
        ts = sorted(t for t, _ in entry["samples"])
        if ts:
            out[entry["topic"]] = np.array(ts, dtype=np.int64)
    return out


def _enforce_camera_coverage(
    episode: CanonicalEpisode, path: str, profile: EmbodimentProfile
) -> CanonicalEpisode:
    """G-01/VERIFY-01 (04-04-PLAN.md): reads the already-computed
    `CAMERA_COVERAGE_CHECK_NAME` verdict out of
    `episode.header.global_meta["checks"]` (produced by
    `data.sync.master_clock.synchronize` via `coverage_check_result`) and
    branches on it. Never recomputes the coverage itself — a second
    implementation of that arithmetic here would risk reproducing the
    synchronizer's own bugs and then certifying them, the exact hazard
    `nearest_camera_matches`' own docstring warns about for validators.

    - Check failed (deficit beyond the declared tolerance): raises
      `GalaxeaAdapterError` carrying a `rejected_header` built the same way
      the fatal-check abort in `parse()` builds one — `episode_id`/
      `source_uri` set to `path`, `fps` `0.0`, `frame_count` `0`, and
      `global_meta` reusing the episode's own already-built
      `stream_inventory` (never rebuilt) plus the full `checks` tuple
      (including the failing coverage entry) and
      `disposition=Disposition.REJECT`. This routes the refusal through
      the exact `Disposition`/`CheckResult` machinery `run_convert`
      already consumes.
    - Check passed with a within-tolerance truncation: downgrades only —
      `PASS` becomes `PASS_WITH_FLAGS`; any disposition already worse than
      `PASS_WITH_FLAGS` is left untouched. Returns a new `CanonicalEpisode`
      wrapping the same single-pass `frames` iterator, the same
      header-rebuild pattern `data/cli.py` and
      `build_varying_baseline_dataset` already use for `task_prompt`.
    - Check passed with no truncation: returns `episode` untouched — the
      no-gap path stays bit-for-bit what it is today.
    """
    checks = tuple(episode.header.global_meta.get("checks", ()))
    coverage_result = next(
        (c for c in checks if c.check_name == CAMERA_COVERAGE_CHECK_NAME), None
    )
    if coverage_result is None:
        return episode

    if not coverage_result.passed:
        evidence = coverage_result.evidence
        shortfall = sorted(
            evidence.get("shortfall_cameras", ()), key=lambda c: c["camera_name"]
        )
        reasons = "; ".join(
            f"{c['camera_name']} ({c['hole_kind']}, covered {c['covered_prefix_ticks']} ticks)"
            for c in shortfall
        )
        message = (
            f"{CAMERA_COVERAGE_CHECK_NAME} failed: deficit_ticks="
            f"{evidence.get('deficit_ticks')} exceeds tolerance_ticks="
            f"{evidence.get('tolerance_ticks')} -- "
            f"{reasons or 'no shortfall camera recorded'}"
        )
        rejected_header = EpisodeHeader(
            episode_id=path,
            fps=0.0,
            robot_type=profile.robot_type,
            task_prompt=None,
            frame_count=0,
            global_meta={
                "profile_hash": profile_hash(profile),
                "episode_id": path,
                "stream_inventory": episode.header.global_meta.get("stream_inventory", ()),
                "checks": checks,
                "disposition": Disposition.REJECT,
            },
        )
        raise GalaxeaAdapterError(message, rejected_header=rejected_header, source_uri=path)

    if not coverage_result.evidence.get("truncated"):
        return episode

    current_disposition = episode.header.global_meta.get("disposition")
    new_disposition = (
        Disposition.PASS_WITH_FLAGS
        if current_disposition == Disposition.PASS
        else current_disposition
    )
    new_header = replace(
        episode.header,
        global_meta={**episode.header.global_meta, "disposition": new_disposition},
    )
    return CanonicalEpisode(header=new_header, frames=episode.frames)


class GalaxeaInputAdapter:
    """`BaseInputAdapter` (data/contracts/protocols.py) by structural
    typing. Reads one Galaxea MCAP recording and produces one
    `CanonicalEpisode` per `profile`'s topic mapping.
    """

    def parse(self, path: str, profile: EmbodimentProfile) -> CanonicalEpisode:
        index = _run_index_pass(path)
        master_name, master_cam = _find_master_clock_camera(profile)

        registry = CheckRegistry.default()
        fatal_ctx = FatalCheckContext(
            scan_error=index.scan_error,
            non_monotonic_topics=index.non_monotonic_topics,
            schema_encodings=tuple(
                (s.name, s.encoding) for s in index.schemas_by_id.values()
            ),
            master_camera_topic=master_cam.topic,
            master_camera_message_count=index.message_counts.get(master_cam.topic, 0),
        )
        fatal_results = registry.run_fatal(fatal_ctx)
        failing_fatal = [r for r in fatal_results if not r.passed]
        if failing_fatal:
            names = ", ".join(r.check_name for r in failing_fatal)
            # PROV-04: a fatally-aborted episode still gets a durable REJECT
            # record — the caller's episode index is not known here (that
            # is Phase 4's batch loop), so this only builds and attaches
            # the header a future ProvenanceRecorder.record_rejected() call
            # needs. Whatever the index pass managed to build before the
            # abort (possibly an empty inventory, if the file genuinely had
            # no channels) is carried as-is, never re-derived.
            rejected_header = EpisodeHeader(
                episode_id=path,
                fps=0.0,
                robot_type=profile.robot_type,
                task_prompt=None,
                frame_count=0,
                global_meta={
                    "profile_hash": profile_hash(profile),
                    "episode_id": path,
                    "stream_inventory": _build_stream_inventory(index, profile, {}),
                    "checks": tuple(fatal_results),
                    "disposition": Disposition.REJECT,
                },
            )
            raise GalaxeaAdapterError(
                f"fatal check(s) failed before any payload decode: {names} — "
                f"evidence: {[(r.check_name, dict(r.evidence)) for r in failing_fatal]}",
                rejected_header=rejected_header,
                source_uri=path,
            )

        state_fields_to_decode, action_resolutions = _plan_decode_requests(profile, index)

        # profile.state_vector_order is the emitted observation.state column
        # order — the sole authority for which state_keys must resolve. A
        # state_key named there that was skipped (an optional field left
        # empty, per Test 6) would otherwise silently truncate the emitted
        # width with no signal — the exact zero-filled/dropped-field failure
        # mode this engine's fail-loud philosophy exists to prevent.
        resolved_state_keys = {f.state_key for f in state_fields_to_decode.values()}
        missing_from_order = [
            key for key in profile.state_vector_order if key not in resolved_state_keys
        ]
        if missing_from_order:
            raise GalaxeaAdapterError(
                "state_vector_order names state_key(s) that did not resolve to any "
                f"decoded field: {sorted(missing_from_order)}"
            )

        raw_streams = _run_decode_pass(
            path,
            master_name=master_name,
            master_cam=master_cam,
            secondary_cameras=_secondary_cameras_to_decode(profile, master_name),
            state_fields=state_fields_to_decode,
            action_resolutions=action_resolutions,
        )

        advisory_ctx = AdvisoryCheckContext(
            camera_payload_digests=index.camera_payload_digests,
            master_camera_topic=master_cam.topic,
            joint_series=_build_joint_series(raw_streams, profile),
            cross_stream_timestamps=_build_cross_stream_timestamps(raw_streams),
        )
        advisory_results = registry.run_advisory(advisory_ctx)
        all_checks = fatal_results + advisory_results
        disposition = (
            Disposition.PASS
            if all(r.passed for r in advisory_results)
            else Disposition.PASS_WITH_FLAGS
        )

        resolved_action_source = {
            action_key: source.topic
            for action_key, (_field, source) in action_resolutions.items()
        }
        raw_streams["meta"] = {
            "profile_hash": profile_hash(profile),
            "episode_id": path,
            "joint_name_ordering": _joint_name_ordering(profile),
            "resolved_action_source": resolved_action_source,
            "stream_inventory": _build_stream_inventory(index, profile, action_resolutions),
            "observation_state_order": tuple(profile.state_vector_order),
            "checks": all_checks,
            "disposition": disposition,
        }
        episode = MasterClockSynchronizer().synchronize(raw_streams, profile)
        return _enforce_camera_coverage(episode, path, profile)
