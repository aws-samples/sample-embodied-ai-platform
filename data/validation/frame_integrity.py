"""VALID-01/VALID-02/D-10: three decode-based per-camera checks.

`videos.frame_count_agreement` (VALID-01) is the four-way per-camera
comparison of frames submitted to the encoder, packets muxed, frames decoded
back out of the MP4, and parquet row count.

The dropped-frame count is measured (`frames_submitted - packets_muxed`,
read from the writer-persisted `meta/encode_manifests/episode_{NNNNNN}.json`
sidecar), never inferred by subtracting two totals gathered from different
places. Every artifact open (parquet metadata, MP4 decode, manifest JSON) is
wrapped in a `try`/`except` scoped to the library's own exception types and
converted to `passed=False` with the exception text under an `error`
evidence key — mirroring `data/adapters/checks.py::_check_mcap_crc`'s
catch-and-report shape — so a deliberately corrupted fixture surfaces as one
failed named check, never an aborted run.

`videos.spot_decode_content` (VALID-02, plan 03-03) recomputes the
frame-to-source mapping by calling `data/sync/master_clock.py`'s own public
`nearest_camera_matches` — never a second copy of that arithmetic — and
compares a handful of spot-checked decoded MP4 frames against the source
recording's own pixels at a *relative*, never absolute, similarity threshold
(D-11). It authenticates its input via `resolve_source_recording`, which
re-hashes the source recording named by the episode's provenance record
before trusting it (D-12) — an absent or changed source recording is an
explicitly-unevaluated result, never a pass.

`videos.decoded_pixel_uniqueness` (D-10, plan 03-03) closes the
re-encoded-static-scene blind spot `data/adapters/checks.py`'s ingest-time
`videos.frozen_duplicate_frames` documents as its own: a stalled camera
driver that re-encodes an identical picture per frame produces
byte-different JPEGs (defeating a compressed-payload digest) but
pixel-identical decoded frames — only a post-write decoded-pixel digest can
see that.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np
import pyarrow.parquet as pq

from data.adapters.galaxea import GalaxeaAdapterError, iter_decoded_messages
from data.contracts.containers import CheckResult
from data.decoders import get_decoder_backend
from data.provenance.recorder import _hash_file_sha256
from data.sync.master_clock import nearest_camera_matches
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY

CHECK_NAME = "videos.frame_count_agreement"
CHECK_NAME_SPOT_DECODE = "videos.spot_decode_content"
CHECK_NAME_PIXEL_UNIQUENESS = "videos.decoded_pixel_uniqueness"

SPOT_CHECK_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
"""VALID-02's spot-check index policy (D-11): the requirement names middle
and end; start plus quarters cost nothing extra while covering the
boundaries a mid-stream content shift most often lands near. The actual
index list is the deduplicated, ascending set of
`floor(f * (decoded_frame_count - 1))` over these fractions — deduplication
is what makes a one-frame camera check index 0 exactly once."""

NEIGHBOUR_WINDOW = 2
"""VALID-02's comparison window (D-11): the target decoded frame's error
against its mapped source frame must be strictly lower than every
neighbour's error at offsets `-NEIGHBOUR_WINDOW..+NEIGHBOUR_WINDOW`
(excluding 0 and any offset outside the decoded sequence's own range)."""

SIMILARITY_METRIC_NAME = "mean_absolute_error"
"""Echoed into every `videos.spot_decode_content` result's evidence so a
later switch to a different metric (e.g. PSNR) is a one-line change visible
in every record already written under this name."""

_BLIND_SPOT_CLOSED_MARKER = "re_encoded_static_scene_not_detected"
"""The same blind-spot marker text `data/adapters/checks.py`'s
`_STATIC_SCENE_BLIND_SPOT_MARKER` carries — repeated here as a literal, not
imported (D-01's registry boundary forbids this module from importing the
ingest-time check package), so `videos.decoded_pixel_uniqueness`'s evidence
can name the blind spot it closes without crossing that boundary."""


def _parquet_row_count(path: Any) -> int | None:
    """Reads the parquet file's own row-group metadata as an int — `None`
    when the file is absent (a zero-frame episode never writes one)."""
    if not path.exists():
        return None
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def _decoded_frame_count(path: Any) -> int:
    """Sequential decode of every video frame — no seeking: with GOP size 2
    seeking would probably work, but at ~1200 frames per episode sequential
    decode removes the keyframe-rounding question entirely."""
    container = av.open(str(path))
    try:
        count = 0
        for _ in container.decode(video=0):
            count += 1
        return count
    finally:
        container.close()


def _iter_decoded_frames(path: Any) -> Iterator[np.ndarray]:
    """Sequential decode of every frame at `path` — no seeking, matching
    `_decoded_frame_count`'s own rationale. Yields one HxWx3 `uint8` RGB
    ndarray per decoded frame; the caller controls how much of the
    generator it consumes, so a caller that only needs a handful of frames
    (VALID-02's spot-check) never holds the whole decoded sequence in
    memory."""
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _decode_frames_at(path: Any, wanted: set[int]) -> dict[int, np.ndarray]:
    """One sequential decode pass over `path`, keeping only the frames whose
    0-based position is in `wanted` — bounds memory to exactly the frames a
    caller asked for rather than the whole decoded sequence. Stops as soon
    as the highest wanted position has been seen."""
    if not wanted:
        return {}
    highest = max(wanted)
    out: dict[int, np.ndarray] = {}
    for i, frame in enumerate(_iter_decoded_frames(path)):
        if i in wanted:
            out[i] = frame
        if i >= highest:
            break
    return out


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """VALID-02/D-11's similarity metric (`SIMILARITY_METRIC_NAME`): the
    mean absolute per-channel difference between two equally-shaped decoded
    RGB arrays, computed in `float64` so no `uint8` wraparound corrupts the
    subtraction."""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _codec_name(path: Any) -> str | None:
    """Read the codec name back from the encoded file, so a decode failure
    caused by an unsupported codec is distinguishable from real content
    corruption. Returns `None` if the codec name cannot even be read."""
    try:
        container = av.open(str(path))
        try:
            return str(container.streams.video[0].codec_context.name)
        finally:
            container.close()
    except Exception:  # noqa: BLE001 - reported not raised, best-effort only
        return None


def _read_manifest(path: Any) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _unevaluated_result(video_key: str, *, reason: str) -> CheckResult:
    return CheckResult(
        CHECK_NAME,
        "fatal",
        False,
        {
            UNEVALUATED_KEY: True,
            SKIP_REASON_KEY: reason,
            "video_key": video_key,
            "parquet_row_count": 0,
        },
    )


def check_frame_count_agreement(
    ctx: ValidationContext, episode_index: int
) -> tuple[CheckResult, ...]:
    """One `CheckResult` per emitted camera. Pass requires all four counters
    — `frames_submitted`, `packets_muxed`, `decoded_frame_count`,
    `parquet_row_count` — to be exactly equal Python ints, compared with no
    tolerance and no rounding. On disagreement, names every disagreeing
    counter pair explicitly. A secondary camera whose decoded frame count is
    strictly less than the parquet row count always fails — even if some
    other counter also happens to match — because a secondary camera's MP4
    strictly shorter than the parquet means video frame N no longer
    corresponds to parquet row N, a genuine dataset defect GR00T's
    frame-indexed video loading cannot tolerate.
    """
    parquet_row_count = _parquet_row_count(ctx.parquet_path(episode_index))

    if parquet_row_count is None:
        return tuple(
            _unevaluated_result(video_key, reason="no parquet written for this episode")
            for video_key in ctx.emitted_video_keys
        )

    manifest = _read_manifest(ctx.encode_manifest_path(episode_index))
    role_by_video_key = {
        cam.video_key: cam.role for cam in ctx.profile.cameras.values() if cam.video_key
    }

    results: list[CheckResult] = []
    for video_key in sorted(ctx.emitted_video_keys):
        video_path = ctx.video_path(episode_index, video_key)
        if not video_path.exists():
            results.append(
                CheckResult(
                    CHECK_NAME,
                    "fatal",
                    False,
                    {
                        "video_key": video_key,
                        "error": f"video file absent: {video_path}",
                        "parquet_row_count": parquet_row_count,
                    },
                )
            )
            continue

        try:
            cam_manifest = manifest[video_key]
            frames_submitted = int(cam_manifest["frames_submitted"])
            packets_muxed = int(cam_manifest["packets_muxed"])
            decoded_frame_count = _decoded_frame_count(video_path)
        except Exception as e:  # noqa: BLE001 - reported not raised
            results.append(
                CheckResult(
                    CHECK_NAME,
                    "fatal",
                    False,
                    {
                        "video_key": video_key,
                        "error": f"{type(e).__name__}: {e}",
                        "parquet_row_count": parquet_row_count,
                    },
                )
            )
            continue

        codec = _codec_name(video_path)
        counters = {
            "frames_submitted": frames_submitted,
            "packets_muxed": packets_muxed,
            "decoded_frame_count": decoded_frame_count,
            "parquet_row_count": parquet_row_count,
        }
        dropped_frames = frames_submitted - packets_muxed
        role = role_by_video_key.get(video_key)

        if role == "secondary" and decoded_frame_count < parquet_row_count:
            results.append(
                CheckResult(
                    CHECK_NAME,
                    "fatal",
                    False,
                    {
                        "video_key": video_key,
                        "role": role,
                        "codec": codec,
                        "dropped_frames": dropped_frames,
                        "unmatched_grid_ticks": parquet_row_count - decoded_frame_count,
                        **counters,
                    },
                )
            )
            continue

        distinct_values = set(counters.values())
        if len(distinct_values) == 1:
            results.append(
                CheckResult(
                    CHECK_NAME,
                    "fatal",
                    True,
                    {"video_key": video_key, "codec": codec, **counters},
                )
            )
            continue

        names = sorted(counters)
        disagreeing_pairs = [
            f"{names[i]}!={names[j]}"
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if counters[names[i]] != counters[names[j]]
        ]
        results.append(
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {
                    "video_key": video_key,
                    "codec": codec,
                    "dropped_frames": dropped_frames,
                    "disagreeing_counters": disagreeing_pairs,
                    **counters,
                },
            )
        )

    return tuple(results)


# ---------------------------------------------------------------------------
# VALID-02 (plan 03-03): source resolution and the spot-decode content check.
# ---------------------------------------------------------------------------


def resolve_source_recording(
    ctx: ValidationContext, episode_index: int
) -> tuple[Path | None, str | None]:
    """Resolves the real source MCAP `videos.spot_decode_content` recomputes
    its mapping against, authenticated by re-hashing (D-12): reads that
    episode's provenance record's `source_uri`/`source_sha256`, and returns
    the path only when it exists **and** re-hashing it with
    `data/provenance/recorder.py::_hash_file_sha256` reproduces the recorded
    digest. Otherwise returns `(None, reason)` with one of three distinct
    named reasons (no provenance record at all, an absent source path, or a
    hash mismatch) — never a guess based on the path alone, since a source
    file that changed under a stable path is exactly the silent-drift case
    this check exists to catch.
    """
    record_path = ctx.provenance_record_path(episode_index)
    if not record_path.exists():
        return None, f"no provenance record found at {record_path}"

    with open(record_path, encoding="utf-8") as f:
        payload = json.load(f)

    source_uri = payload.get("source_uri")
    if not source_uri:
        return None, "provenance record carries no source_uri"

    source_path = Path(source_uri)
    if not source_path.exists():
        return None, f"source recording absent: {source_path}"

    recorded_sha256 = payload.get("source_sha256")
    actual_sha256 = _hash_file_sha256(source_path)
    if actual_sha256 != recorded_sha256:
        return None, (
            f"source recording sha256 mismatch for {source_path}: recorded "
            f"{recorded_sha256}, actual {actual_sha256}"
        )
    return source_path, None


def _unevaluated_spot_result(video_key: str, *, reason: str) -> CheckResult:
    return CheckResult(
        CHECK_NAME_SPOT_DECODE,
        "fatal",
        False,
        {UNEVALUATED_KEY: True, SKIP_REASON_KEY: reason, "video_key": video_key},
    )


def _camera_spec_by_video_key(ctx: ValidationContext) -> dict[str, Any]:
    return {
        cam.video_key: cam for cam in ctx.profile.cameras.values() if cam.video_key is not None
    }


def _master_camera_spec(ctx: ValidationContext) -> Any:
    for cam in ctx.profile.cameras.values():
        if cam.role == "master_clock":
            return cam
    raise ValueError(  # pragma: no cover - guarded by EmbodimentProfile validation
        "profile declares no role: master_clock camera — should be unreachable"
    )


def _read_source_frames(source_path: Path, topic: str) -> list[tuple[int, bytes]]:
    """Buffers `(log_time_ns, raw_jpeg_bytes)` pairs for `topic` from
    `source_path`, sorted ascending — the same timestamp
    `data/adapters/galaxea.py::_run_decode_pass` buffers, so the two agree
    by construction (D-12)."""
    pairs = [
        (ts, bytes(msg.data))
        for _topic, ts, msg in iter_decoded_messages(str(source_path), [topic])
    ]
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def check_spot_decode_content(
    ctx: ValidationContext, episode_index: int
) -> tuple[CheckResult, ...]:
    """VALID-02/D-11/D-12: a **relative** content-shift check. For each
    emitted camera, recomputes the frame-to-source mapping by calling
    `data/sync/master_clock.py::nearest_camera_matches` (never a second
    implementation of that arithmetic), decodes a handful of spot-checked
    MP4 frame positions plus their +/- `NEIGHBOUR_WINDOW` neighbours, and
    requires the mapped-source-frame error at the target position to be
    strictly lower than every neighbour's error. A tie fails — a tie carries
    no evidence the target matched better than a shifted neighbour would
    have. An absent or changed source recording is explicitly unevaluated,
    never a pass. A camera whose every spot-checked index has zero valid
    neighbour offsets (only reachable when `decoded_frame_count == 1`) is
    also explicitly unevaluated (CR-01, 03-REVIEW.md) — with no neighbour to
    compare against, nothing was actually verified, so this never falls
    through to a vacuous `passed=True`.
    """
    parquet_row_count = _parquet_row_count(ctx.parquet_path(episode_index))
    if parquet_row_count is None:
        return tuple(
            _unevaluated_spot_result(
                video_key, reason="zero-frame episode — no parquet written"
            )
            for video_key in ctx.emitted_video_keys
        )

    source_path, skip_reason = resolve_source_recording(ctx, episode_index)
    if source_path is None:
        return tuple(
            _unevaluated_spot_result(video_key, reason=skip_reason or "source unresolved")
            for video_key in ctx.emitted_video_keys
        )

    with open(ctx.provenance_record_path(episode_index), encoding="utf-8") as f:
        record_payload = json.load(f)
    grid_hz = record_payload.get("clock_report", {}).get("grid_hz")

    master_cam = _master_camera_spec(ctx)
    cam_specs = _camera_spec_by_video_key(ctx)

    try:
        master_frames = _read_source_frames(source_path, master_cam.topic)
    except GalaxeaAdapterError as e:
        return tuple(
            CheckResult(
                CHECK_NAME_SPOT_DECODE,
                "fatal",
                False,
                {"video_key": video_key, "error": f"{type(e).__name__}: {e}"},
            )
            for video_key in ctx.emitted_video_keys
        )
    grid_ts_ns = np.array([ts for ts, _ in master_frames], dtype=np.int64)
    master_source_bytes = [b for _, b in master_frames]

    results: list[CheckResult] = []
    for video_key in sorted(ctx.emitted_video_keys):
        cam_spec = cam_specs.get(video_key)
        video_path = ctx.video_path(episode_index, video_key)
        if not video_path.exists():
            results.append(
                CheckResult(
                    CHECK_NAME_SPOT_DECODE,
                    "fatal",
                    False,
                    {"video_key": video_key, "error": f"video file absent: {video_path}"},
                )
            )
            continue

        role = cam_spec.role if cam_spec is not None else None
        if cam_spec is None and role != "master_clock":
            results.append(
                CheckResult(
                    CHECK_NAME_SPOT_DECODE,
                    "fatal",
                    False,
                    {
                        "video_key": video_key,
                        "error": f"video_key {video_key!r} has no matching profile.cameras entry",
                    },
                )
            )
            continue
        try:
            if role == "master_clock":
                source_bytes_by_index = master_source_bytes
                source_index_for_mp4_frame = list(range(len(master_source_bytes)))
            else:
                assert cam_spec is not None  # guarded above
                cam_frames = _read_source_frames(source_path, cam_spec.topic)
                frame_ts_ns = np.array([ts for ts, _ in cam_frames], dtype=np.int64)
                source_bytes_by_index = [b for _, b in cam_frames]
                max_gap_ns = int((cam_spec.max_sync_gap_ms or 0) * 1_000_000)
                matched_idx, _gap_ns = nearest_camera_matches(
                    frame_ts_ns, grid_ts_ns, max_gap_ns
                )
                ticks_with_match = [
                    g for g in range(grid_ts_ns.shape[0]) if matched_idx[g] >= 0
                ]
                source_index_for_mp4_frame = [int(matched_idx[g]) for g in ticks_with_match]

            codec = _codec_name(video_path)
            decoded_frame_count = _decoded_frame_count(video_path)

            if len(source_index_for_mp4_frame) != decoded_frame_count:
                results.append(
                    CheckResult(
                        CHECK_NAME_SPOT_DECODE,
                        "fatal",
                        False,
                        {
                            "video_key": video_key,
                            "role": role,
                            "codec": codec,
                            "error": (
                                "recomputed source-mapping length disagrees with the "
                                "decoded frame count"
                            ),
                            "matched_tick_count": len(source_index_for_mp4_frame),
                            "decoded_frame_count": decoded_frame_count,
                        },
                    )
                )
                continue

            if decoded_frame_count == 0:
                results.append(
                    CheckResult(
                        CHECK_NAME_SPOT_DECODE,
                        "fatal",
                        False,
                        {
                            "video_key": video_key,
                            "role": role,
                            "codec": codec,
                            "error": "camera decoded to zero frames",
                        },
                    )
                )
                continue

            spot_indices = sorted(
                {int((f * (decoded_frame_count - 1)) // 1) for f in SPOT_CHECK_FRACTIONS}
            )

            wanted: set[int] = set()
            for idx in spot_indices:
                wanted.add(idx)
                for offset in range(-NEIGHBOUR_WINDOW, NEIGHBOUR_WINDOW + 1):
                    if offset == 0:
                        continue
                    pos = idx + offset
                    if 0 <= pos < decoded_frame_count:
                        wanted.add(pos)
            decoded_frames = _decode_frames_at(video_path, wanted)

            per_index_evidence: list[dict[str, Any]] = []
            passed = True
            any_neighbour_compared = False
            for idx in spot_indices:
                source_ordinal = source_index_for_mp4_frame[idx]
                expected_pixels = get_decoder_backend().decode(
                    source_bytes_by_index[source_ordinal]
                )
                target_pixels = decoded_frames[idx]
                target_error = mean_absolute_error(target_pixels, expected_pixels)

                neighbour_errors: dict[int, float] = {}
                for offset in range(-NEIGHBOUR_WINDOW, NEIGHBOUR_WINDOW + 1):
                    if offset == 0:
                        continue
                    pos = idx + offset
                    if pos in decoded_frames:
                        neighbour_errors[offset] = mean_absolute_error(
                            decoded_frames[pos], expected_pixels
                        )

                if neighbour_errors:
                    any_neighbour_compared = True

                combined = {0: target_error, **neighbour_errors}
                winning_offset = min(combined, key=lambda k: combined[k])
                # `all(... for e in neighbour_errors.values())` is `True`
                # vacuously when `neighbour_errors` is empty (CR-01,
                # 03-REVIEW.md) — require at least one real comparison
                # before an index can pass.
                idx_passed = bool(neighbour_errors) and all(
                    target_error < e for e in neighbour_errors.values()
                )
                if not idx_passed:
                    passed = False

                per_index_evidence.append(
                    {
                        "index": idx,
                        "target_error": target_error,
                        "neighbour_errors": {
                            str(o): e for o, e in sorted(neighbour_errors.items())
                        },
                        "winning_offset": winning_offset,
                    }
                )

            if not any_neighbour_compared:
                # Every spot-checked index had zero valid neighbour offsets
                # (only reachable when `decoded_frame_count == 1`, given
                # `NEIGHBOUR_WINDOW`) — nothing was ever compared against,
                # so this camera's content was never actually verified.
                # Route to UNEVALUATED_KEY rather than the vacuous
                # `idx_passed=True` this used to fall through to.
                results.append(
                    CheckResult(
                        CHECK_NAME_SPOT_DECODE,
                        "fatal",
                        False,
                        {
                            UNEVALUATED_KEY: True,
                            SKIP_REASON_KEY: (
                                f"decoded_frame_count {decoded_frame_count} leaves no "
                                "neighbour frame to compare against for any "
                                "spot-checked index — nothing was verified"
                            ),
                            "video_key": video_key,
                            "role": role,
                            "codec": codec,
                            "decoded_frame_count": decoded_frame_count,
                            "per_index": per_index_evidence,
                        },
                    )
                )
                continue

            results.append(
                CheckResult(
                    CHECK_NAME_SPOT_DECODE,
                    "fatal",
                    passed,
                    {
                        "video_key": video_key,
                        "role": role,
                        "codec": codec,
                        "similarity_metric": SIMILARITY_METRIC_NAME,
                        "spot_check_fractions": list(SPOT_CHECK_FRACTIONS),
                        "neighbour_window": NEIGHBOUR_WINDOW,
                        "grid_hz": grid_hz,
                        "decoded_frame_count": decoded_frame_count,
                        "per_index": per_index_evidence,
                    },
                )
            )
        except (av.FFmpegError, GalaxeaAdapterError) as e:
            results.append(
                CheckResult(
                    CHECK_NAME_SPOT_DECODE,
                    "fatal",
                    False,
                    {
                        "video_key": video_key,
                        "role": role,
                        "codec": _codec_name(video_path),
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
            )

    return tuple(results)


# ---------------------------------------------------------------------------
# D-10 (plan 03-03): post-write decoded-pixel uniqueness check.
# ---------------------------------------------------------------------------


def _digest_decoded_frames(path: Any) -> tuple[int, int, int]:
    """One sequential decode pass over `path`, digesting each decoded frame
    with `hashlib.sha256` over a contiguous `uint8` buffer — the project's
    one approved content-identity digest (never a second algorithm, never a
    perceptual hash). Returns `(decoded_frame_count, distinct_digest_count,
    consecutive_identical_pairs)` — only the digests (cheap strings) are
    retained across the pass, never the decoded frames themselves."""
    digests: list[str] = []
    count = 0
    for frame in _iter_decoded_frames(path):
        digest = hashlib.sha256(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()).hexdigest()
        digests.append(digest)
        count += 1
    distinct_digest_count = len(set(digests))
    consecutive_identical_pairs = sum(1 for a, b in zip(digests, digests[1:]) if a == b)
    return count, distinct_digest_count, consecutive_identical_pairs


def check_decoded_pixel_uniqueness(
    ctx: ValidationContext, episode_index: int
) -> tuple[CheckResult, ...]:
    """D-10: fails when a camera's decoded frames collapse to a single pixel
    digest while it decoded more than one frame — a camera that produced
    many frames and one picture. Closes the re-encoded-static-scene blind
    spot `data/adapters/checks.py`'s ingest-time `videos.frozen_duplicate_frames`
    documents as its own: that check hashes compressed *payload* bytes, so a
    stalled driver that re-encodes an identical picture per frame (different
    JPEG bytes, identical pixels) defeats it — this check works on decoded
    pixels instead. Severity is `fatal` deliberately: a still camera is
    always a broken sensor (unlike a still joint, which can be a genuine,
    intentional part of a task), and a dataset trained on one broken camera
    view is poisoned in a way no downstream stage recovers from. A camera
    with fewer than two decoded frames is explicitly unevaluated — one frame
    cannot be non-unique, and reporting that as a pass would be a silent
    green.
    """
    results: list[CheckResult] = []
    for video_key in sorted(ctx.emitted_video_keys):
        video_path = ctx.video_path(episode_index, video_key)
        if not video_path.exists():
            results.append(
                CheckResult(
                    CHECK_NAME_PIXEL_UNIQUENESS,
                    "fatal",
                    False,
                    {"video_key": video_key, "error": f"video file absent: {video_path}"},
                )
            )
            continue

        codec = _codec_name(video_path)
        try:
            decoded_frame_count, distinct_digest_count, consecutive_identical_pairs = (
                _digest_decoded_frames(video_path)
            )
        except av.FFmpegError as e:
            results.append(
                CheckResult(
                    CHECK_NAME_PIXEL_UNIQUENESS,
                    "fatal",
                    False,
                    {"video_key": video_key, "codec": codec, "error": f"{type(e).__name__}: {e}"},
                )
            )
            continue

        if decoded_frame_count < 2:
            results.append(
                CheckResult(
                    CHECK_NAME_PIXEL_UNIQUENESS,
                    "fatal",
                    False,
                    {
                        UNEVALUATED_KEY: True,
                        SKIP_REASON_KEY: (
                            f"decoded frame count {decoded_frame_count} < 2 — one frame "
                            "cannot be non-unique"
                        ),
                        "video_key": video_key,
                        "codec": codec,
                        "decoded_frame_count": decoded_frame_count,
                    },
                )
            )
            continue

        passed = distinct_digest_count > 1
        results.append(
            CheckResult(
                CHECK_NAME_PIXEL_UNIQUENESS,
                "fatal",
                passed,
                {
                    "video_key": video_key,
                    "codec": codec,
                    "decoded_frame_count": decoded_frame_count,
                    "distinct_digest_count": distinct_digest_count,
                    "consecutive_identical_pairs": consecutive_identical_pairs,
                    "closes_blind_spot": _BLIND_SPOT_CLOSED_MARKER,
                },
            )
        )

    return tuple(results)
