"""Corruption-mutator library and source-backed dataset baselines for Phase
3's negative tests. A tracked test *helper*, not a `test_*.py` module —
pytest does not collect it; importable as `data.tests.mutators`.

Every baseline this phase's fixtures are built from is backed by a real
source MCAP on disk, written through `data/tests/mcap_fixture.py`'s
`build_fixture_mcap` — never fabricated by hand. This is what makes every
ingest-time provenance record written beside a baseline resolvable by
`ProvenanceRecorder.record` (its `source_uri` names a real file to hash)
and, for `build_varying_baseline_dataset` (Task 2), what makes a *passing*
VALID-02 spot-decode verdict reachable in plan 03-03 rather than only its
two skip paths.

Every mutator operates on an already-copied dataset directory (never the
original) and returns that directory's path. No fault-injection code enters
`data/writers/` or `data/validation/` — a negative `grep` in this plan's
`<verify>` block enforces that boundary.
"""

import dataclasses
import json
import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import simplejpeg

from data.adapters.galaxea import GalaxeaInputAdapter
from data.contracts.containers import CanonicalEpisode, CheckResult, Disposition
from data.contracts.profile import EmbodimentProfile
from data.provenance.recorder import ProvenanceRecorder
from data.tests.frame_fixtures import build_synthetic_episode
from data.tests.mcap_fixture import (
    SCHEMA_NAME_COMPRESSED_IMAGE,
    SCHEMA_TEXT_COMPRESSED_IMAGE,
    TopicSpec,
    build_fixture_mcap,
    camera_messages,
    leading_target_joint_topics,
    varying_motion_joint_topics,
)
from data.writers.lerobot_v21 import (
    _CHUNK_INDEX,
    _DATA_PATH_TEMPLATE,
    _VIDEO_PATH_TEMPLATE,
    ENCODE_MANIFEST_DIRNAME,
    LeRobotV21Writer,
)

_CAMERA_TOPICS = (
    "/hdas/camera_head/left_raw/image_raw_color/compressed",
    "/hdas/camera_head/right_raw/image_raw_color/compressed",
    "/hdas/camera_wrist_left/color/image_raw/compressed",
    "/hdas/camera_wrist_right/color/image_raw/compressed",
)
"""The four camera topics `mcap_fixture.default_topics` declares — mirrored
here (not re-exported by that module) so `build_fixture_source_mcap` can
override every camera's payload with per-frame-distinct JPEGs."""

_GRADIENT_STEP = 37
"""Per-frame gradient offset (mod 256) — far larger than crf=30 quantization
noise, so two frames are never pixel-indistinguishable after re-encode."""


def _distinct_gradient_jpeg(index: int, *, width: int = 64, height: int = 48) -> bytes:
    """One deterministic RGB gradient JPEG, offset by `index` steps of
    `_GRADIENT_STEP` — every frame index produces a visually distinct
    image, unlike `mcap_fixture`'s own default shared tiny JPEG."""
    offset = (index * _GRADIENT_STEP) % 256
    x_ramp = np.linspace(0, 255, width, dtype=np.int32)
    y_ramp = np.linspace(0, 255, height, dtype=np.int32)
    red = (np.tile(x_ramp, (height, 1)) + offset) % 256
    green = (np.tile(y_ramp.reshape(-1, 1), (1, width)) + offset) % 256
    blue = np.full((height, width), (128 + offset) % 256, dtype=np.int32)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return simplejpeg.encode_jpeg(rgb, colorspace="rgb")


def build_fixture_source_mcap(
    dest_dir: str | Path,
    *,
    camera_frames: int,
    distinct_frames: bool = True,
    joint_motion: Literal["ramp", "varying", "leading_target"] = "ramp",
    camera_frames_by_topic: dict[str, int] | None = None,
) -> Path:
    """Writes one real Galaxea MCAP to `dest_dir/episode_source.mcap` by
    calling `data/tests/mcap_fixture.py::build_fixture_mcap`, and returns
    that path. When `distinct_frames` is true, overrides each of the four
    `CompressedImage` camera topics with `camera_frames` per-frame-distinct
    JPEG payloads — `mcap_fixture`'s default camera payload is one shared
    tiny JPEG repeated for every message, which would make any
    content-sensitive negative fixture pixel-identical and therefore
    undetectable.

    `camera_frames_by_topic` (default `None`, additive keyword) overrides
    ONE OR MORE of the four camera topics' own frame count instead of
    `camera_frames` — e.g. `{topic: 8}` gives that one camera 8 distinct-
    gradient frames while the other three keep `camera_frames`, so a
    shorter camera's stream genuinely ends earlier in time
    (G-01/VERIFY-01, 04-04-PLAN.md). Threaded HERE, not only into
    `mcap_fixture.default_topics`, because this function's own
    `distinct_frames` branch is the one every real-pipeline fixture
    actually uses and it unconditionally rebuilds ALL FOUR camera topics at
    `camera_frames` count — without threading the override through here, it
    would be silently clobbered on the one code path that needs it.

    `joint_motion` (default `"ramp"`, additive keyword — every existing
    call site is unaffected) opts into `mcap_fixture.varying_motion_joint_topics`
    (achieved-DELTA action pairing) or `mcap_fixture.leading_target_joint_topics`
    (absolute-position-TARGET action pairing, VALID-03 Option B — plan
    03-11) instead of `mcap_fixture.default_topics`'s own linear
    per-channel ramp for the six state topics and six action-source
    topics. `"ramp"` MUST stay the default: `test_no_declared_expected_lag_is_unevaluated`
    (plan 03-11) is real coverage of `check_action_state_lag`'s
    no-declared-expectation abstention branch built on it, and flipping
    this default would silently retarget that test.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    mcap_path = dest / "episode_source.mcap"

    camera_topics: dict[str, TopicSpec | None] = {}
    if distinct_frames:
        camera_topics = {
            topic: TopicSpec(
                SCHEMA_NAME_COMPRESSED_IMAGE,
                SCHEMA_TEXT_COMPRESSED_IMAGE,
                camera_messages(
                    count=(camera_frames_by_topic or {}).get(topic, camera_frames),
                    jpeg_bytes=[
                        _distinct_gradient_jpeg(i)
                        for i in range((camera_frames_by_topic or {}).get(topic, camera_frames))
                    ],
                ),
            )
            for topic in _CAMERA_TOPICS
        }

    joint_topics: dict[str, TopicSpec | None] = {}
    if joint_motion == "varying":
        joint_topics = dict(varying_motion_joint_topics(joint_samples=camera_frames))
    elif joint_motion == "leading_target":
        joint_topics = dict(leading_target_joint_topics(joint_samples=camera_frames))

    topics: dict[str, TopicSpec | None] | None = None
    if camera_topics or joint_topics:
        topics = {**camera_topics, **joint_topics}

    build_fixture_mcap(
        str(mcap_path),
        camera_frames=camera_frames,
        joint_samples=camera_frames,
        topics=topics,
        camera_frames_by_topic=None if distinct_frames else camera_frames_by_topic,
    )
    return mcap_path


def build_baseline_dataset(
    dest_dir: str | Path,
    profile: EmbodimentProfile,
    *,
    frame_count: int,
    fps: float = 30.0,
    emit_cameras: list[str] | None = None,
) -> Path:
    """The **structural** baseline, for fixtures whose pixel content is
    irrelevant (zero-frame, count-only, schema-shape). Writes one real
    source MCAP first (via `build_fixture_source_mcap`), builds an episode
    with `build_synthetic_episode`, then rebuilds its header so
    `episode_id` is the source MCAP's real on-disk path (never the
    fixture's own hardcoded `"synthetic-episode"`, which is not a path
    `ProvenanceRecorder.record` can hash) and `global_meta` carries a small
    non-empty tuple of plausible ingest-time `CheckResult`s plus a
    `Disposition` — the entries a later amend must preserve at their
    original list positions.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    source_mcap = build_fixture_source_mcap(dest / "sources", camera_frames=max(frame_count, 1))

    episode = build_synthetic_episode(
        frame_count=frame_count,
        fps=fps,
        profile=profile,
        emit_cameras=emit_cameras,
        task_prompt="baseline task" if frame_count else None,
    )

    ingest_time_checks: tuple[CheckResult, ...] = (
        CheckResult("metadata.mcap_crc", "fatal", True, {}),
        CheckResult("videos.master_camera_frames", "fatal", True, {}),
    )
    rebuilt_header = dataclasses.replace(
        episode.header,
        episode_id=str(source_mcap),
        global_meta={
            **episode.header.global_meta,
            "checks": ingest_time_checks,
            "disposition": Disposition.PASS,
        },
    )
    rebuilt_episode = CanonicalEpisode(header=rebuilt_header, frames=episode.frames)

    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    write_result = writer.write_episode(rebuilt_episode)
    writer.finalize()

    ProvenanceRecorder(dest, profile).record(rebuilt_header, write_result)

    return dest


def copy_dataset(src: str | Path, dst: str | Path) -> Path:
    """A `shutil.copytree` — every mutator operates on the returned copy,
    never the original."""
    src_path, dst_path = Path(src), Path(dst)
    shutil.copytree(src_path, dst_path)
    return dst_path


def _encode_manifest_path(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / "meta" / ENCODE_MANIFEST_DIRNAME / f"episode_{episode_index:06d}.json"


def _decrement_manifest_packets_muxed(
    dataset_dir: Path, *, video_key: str, episode_index: int, by: int
) -> None:
    """Keeps the per-camera encode manifest internally consistent with a
    post-hoc video-content mutation: `packets_muxed` for `video_key` is
    decremented by `by`, `frames_submitted` is left untouched (representing
    frames that genuinely reached the encoder but were not muxed) — the
    scenario `truncate_camera_mp4` simulates. A no-op if no manifest entry
    exists for `video_key`."""
    manifest_path = _encode_manifest_path(dataset_dir, episode_index)
    if not manifest_path.exists():
        raise ValueError(f"no encode manifest at {manifest_path} — nothing to decrement")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    entry = manifest.get(video_key)
    if entry is None:
        raise ValueError(f"encode manifest at {manifest_path} has no entry for {video_key!r}")
    entry["packets_muxed"] = max(int(entry["packets_muxed"]) - by, 0)

    tmp_path = manifest_path.parent / f".{manifest_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
    os.replace(tmp_path, manifest_path)


def _video_file_path(dataset_dir: Path, video_key: str, episode_index: int) -> Path:
    return dataset_dir / _VIDEO_PATH_TEMPLATE.format(
        episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=episode_index
    )


def _reencode(
    frames: list[np.ndarray],
    out_path: Path,
    *,
    codec: str,
    rate: int,
    width: int,
    height: int,
) -> None:
    """Re-muxes `frames` (a list of HxWx3 uint8 RGB arrays) into `out_path`,
    matching the writer's own codec/rate/pix_fmt/options
    (`data/writers/lerobot_v21.py::_CameraEncodeState._open`) so the only
    observable difference from the original is the mutation applied to
    `frames` itself."""
    tmp_path = out_path.parent / f".{out_path.name}.tmp"
    container = av.open(str(tmp_path), mode="w", format="mp4")
    try:
        stream: Any = container.add_stream(codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        time_base = Fraction(1, rate)
        stream.codec_context.time_base = time_base
        stream.options = {"g": "2", "crf": "30"}
        for i, rgb in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            video_frame.pts = i
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    os.replace(tmp_path, out_path)


def _decode_all_frames(path: Path) -> tuple[list[np.ndarray], str, int, int, int]:
    """Returns `(frames, codec_name, rate, width, height)` for the video at
    `path`, decoded sequentially."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        codec_name = str(stream.codec_context.name)
        rate = int(round(float(stream.average_rate))) if stream.average_rate else 1
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        return frames, codec_name, rate, width, height
    finally:
        container.close()


def _require_video(dataset_dir: Path, video_key: str, episode_index: int) -> Path:
    path = _video_file_path(dataset_dir, video_key, episode_index)
    if not path.exists():
        raise ValueError(f"no MP4 found at {path} for video_key {video_key!r} — nothing to mutate")
    return path


def truncate_camera_mp4(
    dataset_dir: str | Path,
    *,
    video_key: str,
    drop_last_n: int,
    episode_index: int = 0,
) -> Path:
    """Decodes the target MP4 and re-muxes all but the last `drop_last_n`
    frames, using the same codec, rate, pix_fmt and `{"g": "2", "crf": "30"}`
    options the writer sets. Also decrements that camera's encode-manifest
    `packets_muxed` by the same amount — simulating the scenario this
    mutator exists to prove: the encoder genuinely muxed fewer packets than
    it submitted, so `packets_muxed`/`decoded_frame_count` disagree with
    `frames_submitted`/`parquet_row_count`, and the measured
    `frames_submitted - packets_muxed` dropped-frame count is exactly
    `drop_last_n`."""
    dataset_dir = Path(dataset_dir)
    video_path = _require_video(dataset_dir, video_key, episode_index)
    frames, codec, rate, width, height = _decode_all_frames(video_path)
    keep = max(len(frames) - drop_last_n, 0)
    actual_drop = len(frames) - keep
    _reencode(frames[:keep], video_path, codec=codec, rate=rate, width=width, height=height)
    if actual_drop > 0:
        _decrement_manifest_packets_muxed(
            dataset_dir, video_key=video_key, episode_index=episode_index, by=actual_drop
        )
    return dataset_dir


# ---------------------------------------------------------------------------
# Task 2: the source-backed, per-frame-distinct baseline and the remaining
# corruption mutators — one per named check this phase registers.
# ---------------------------------------------------------------------------


def build_varying_baseline_dataset(
    dest_dir: str | Path,
    profile: EmbodimentProfile,
    *,
    camera_frames: int = 12,
    episodes: int = 1,
    joint_motion: Literal["ramp", "varying", "leading_target"] = "ramp",
) -> Path:
    """The **source-backed, per-frame-distinct** baseline every
    content-sensitive negative test in this phase builds on. For each
    episode, writes one real MCAP under a `sources/` sibling of the dataset
    prefix (via `build_fixture_source_mcap(..., distinct_frames=True)`),
    then runs that file through `GalaxeaInputAdapter().parse` and hands the
    resulting `CanonicalEpisode` straight to `LeRobotV21Writer.write_episode`
    — going through the real adapter rather than `build_synthetic_episode`
    is load-bearing: the written MP4 frames must genuinely derive from the
    source recording's own camera payloads, or a spot-decode comparison
    (plan 03-03's VALID-02) has nothing that can *match*.

    The adapter's own `parse()` always sets `header.task_prompt = None`
    (real recordings never carry a language string embedded in the MCAP),
    which would make the writer's D-09 fail-loud task-string policy raise
    under `stdin_isatty=False` — this rebuilds the header with the literal
    `"auto"` opt-in value (`AUTO_TASK_OPT_IN`) so every episode still writes
    successfully with the self-evidently-invalid `AUTO_TASK_SENTINEL`
    placeholder, never a plausible-looking default. The adapter also
    populates `episode_id`, `checks` and `disposition` itself, so no further
    header rebuild is needed on this path and the recorded ingest-time
    evidence is real rather than fabricated.

    `joint_motion` (default `"ramp"`) is threaded straight through to
    `build_fixture_source_mcap`. Naming hazard worth flagging explicitly:
    "varying" in THIS FUNCTION'S NAME has always meant per-frame-distinct
    camera *pixels* (`distinct_frames=True` on the source builder), while
    the `joint_motion` parameter controls *joint* motion — two independent
    axes that happen to share a word. The function is not renamed to
    disambiguate, since doing so would churn every existing call site for a
    naming collision, not a behavioural one.

    Returns the dataset prefix only — the source recordings are discoverable
    from each episode's own `source_uri`, exactly how plan 03-03's
    `resolve_source_recording` finds them.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    recorder = ProvenanceRecorder(dest, profile)

    for ep in range(episodes):
        source_mcap = build_fixture_source_mcap(
            dest / "sources" / f"episode_{ep:06d}",
            camera_frames=camera_frames,
            joint_motion=joint_motion,
        )
        episode = GalaxeaInputAdapter().parse(str(source_mcap), profile)
        header = dataclasses.replace(episode.header, task_prompt="auto")
        rebuilt = CanonicalEpisode(header=header, frames=episode.frames)
        write_result = writer.write_episode(rebuilt)
        recorder.record(header, write_result)

    writer.finalize()
    return dest


def _discovered_episode_indices(dataset_dir: Path) -> list[int]:
    chunk_dir = dataset_dir / "data" / f"chunk-{_CHUNK_INDEX:03d}"
    if not chunk_dir.is_dir():
        return []
    indices: list[int] = []
    for path in chunk_dir.glob("episode_*.parquet"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(indices)


def _parquet_file_path(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / _DATA_PATH_TEMPLATE.format(
        episode_chunk=_CHUNK_INDEX, episode_index=episode_index
    )


def _require_parquet(dataset_dir: Path, episode_index: int) -> Path:
    path = _parquet_file_path(dataset_dir, episode_index)
    if not path.exists():
        raise ValueError(f"no parquet at {path} — nothing to mutate")
    return path


def _rewrite_fixed_size_list_column(
    table: "pa.Table", column_name: str, rows: list[list[float]], width: int
) -> "pa.Table":
    flat = [v for row in rows for v in row]
    new_col = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)
    col_idx = table.schema.get_field_index(column_name)
    return table.set_column(col_idx, column_name, new_col)


def shift_camera_frames(
    dataset_dir: str | Path, *, video_key: str, episode_index: int = 0, shift: int = 1
) -> Path:
    """Decodes the target MP4 and re-encodes the same number of frames with
    the sequence rotated by `shift` (drops the leading `shift` frames,
    repeats the final frame `shift` times) — frame count and container
    metadata are unchanged, only content moved. Target:
    `videos.spot_decode_content`."""
    dataset_dir = Path(dataset_dir)
    video_path = _require_video(dataset_dir, video_key, episode_index)
    frames, codec, rate, width, height = _decode_all_frames(video_path)
    if shift <= 0 or shift >= len(frames):
        raise ValueError(
            f"shift={shift} invalid for {len(frames)} decoded frames — must be in [1, "
            f"{len(frames) - 1}]"
        )
    shifted = frames[shift:] + [frames[-1]] * shift
    _reencode(shifted, video_path, codec=codec, rate=rate, width=width, height=height)
    return dataset_dir


def freeze_camera_frames(
    dataset_dir: str | Path, *, video_key: str, episode_index: int = 0
) -> Path:
    """Re-encodes with every frame equal to frame 0. Target:
    `videos.decoded_pixel_uniqueness`."""
    dataset_dir = Path(dataset_dir)
    video_path = _require_video(dataset_dir, video_key, episode_index)
    frames, codec, rate, width, height = _decode_all_frames(video_path)
    if not frames:
        raise ValueError(f"{video_path} decodes to zero frames — nothing to freeze")
    frozen = [frames[0]] * len(frames)
    _reencode(frozen, video_path, codec=codec, rate=rate, width=width, height=height)
    return dataset_dir


def perturb_stats_value(
    dataset_dir: str | Path, *, feature: str, stat: str, channel_index: int = 0, delta: float
) -> Path:
    """Loads `meta/stats.json`, adds `delta` to one channel of one of the
    seven statistics, re-writes with the writer's own `sort_keys=True,
    indent=2` convention. Target: `statistics.stats_recomputation`."""
    dataset_dir = Path(dataset_dir)
    stats_path = dataset_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise ValueError(f"no meta/stats.json at {stats_path} — nothing to perturb")
    with open(stats_path, encoding="utf-8") as f:
        payload = json.load(f)
    if feature not in payload or stat not in payload[feature]:
        raise ValueError(f"stats.json has no {feature}.{stat} entry to perturb")
    payload[feature][stat][channel_index] = payload[feature][stat][channel_index] + delta
    tmp_path = stats_path.parent / f".{stats_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, stats_path)
    return dataset_dir


def inject_nan_into_stats(
    dataset_dir: str | Path, *, feature: str, stat: str, channel_index: int = 0
) -> Path:
    """Sets one value in `meta/stats.json` to a non-finite float. Python's
    json writer emits the non-standard `NaN`/`Infinity` tokens by default
    and its reader accepts them, which is exactly why this corruption is
    silent on disk. Target: `anomalies.nan_inf`."""
    dataset_dir = Path(dataset_dir)
    stats_path = dataset_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise ValueError(f"no meta/stats.json at {stats_path} — nothing to corrupt")
    with open(stats_path, encoding="utf-8") as f:
        payload = json.load(f)
    if feature not in payload or stat not in payload[feature]:
        raise ValueError(f"stats.json has no {feature}.{stat} entry to corrupt")
    payload[feature][stat][channel_index] = float("nan")
    tmp_path = stats_path.parent / f".{stats_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, stats_path)
    return dataset_dir


def inject_nan_into_parquet(
    dataset_dir: str | Path,
    *,
    feature: str,
    row_index: int = 0,
    channel_index: int = 0,
    episode_index: int = 0,
) -> Path:
    """Reads the table with pyarrow, replaces one element of one
    fixed-size-list row with a non-finite float, re-writes preserving the
    exact schema `_build_table` declares. Target: `anomalies.nan_inf`."""
    dataset_dir = Path(dataset_dir)
    parquet_path = _require_parquet(dataset_dir, episode_index)
    table = pq.read_table(parquet_path)
    if feature not in table.schema.names:
        raise ValueError(f"parquet has no column {feature!r} to corrupt")
    field_type = table.schema.field(feature).type
    if not pa.types.is_fixed_size_list(field_type):
        raise ValueError(f"column {feature!r} is not a fixed-size list column")
    width = field_type.list_size
    rows = table.column(feature).to_pylist()
    rows[row_index][channel_index] = float("nan")
    new_table = _rewrite_fixed_size_list_column(table, feature, rows, width)
    tmp_path = parquet_path.parent / f".{parquet_path.name}.tmp"
    pq.write_table(new_table, tmp_path)
    os.replace(tmp_path, parquet_path)
    return dataset_dir


def drop_tasks_row(dataset_dir: str | Path, *, task_index: int) -> Path:
    """Removes exactly that row from `meta/tasks.jsonl`, touching no
    parquet. Target: `metadata.task_index_join`."""
    dataset_dir = Path(dataset_dir)
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise ValueError(f"no meta/tasks.jsonl at {tasks_path} — nothing to drop")
    rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remaining = [row for row in rows if row.get("task_index") != task_index]
    if len(remaining) == len(rows):
        raise ValueError(f"tasks.jsonl has no row with task_index={task_index!r} to drop")
    tmp_path = tasks_path.parent / f".{tasks_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in remaining:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, tasks_path)
    return dataset_dir


def shift_action_column(
    dataset_dir: str | Path, *, episode_index: int = 0, shift: int = 1
) -> Path:
    """Rotates the `action` column's rows by `shift`, preserving row count,
    dtype, and the fixed-size-list width. Target:
    `consistency.action_state_lag`."""
    dataset_dir = Path(dataset_dir)
    parquet_path = _require_parquet(dataset_dir, episode_index)
    table = pq.read_table(parquet_path)
    field_type = table.schema.field("action").type
    width = field_type.list_size
    rows = table.column("action").to_pylist()
    n = len(rows)
    if n == 0:
        raise ValueError(f"{parquet_path}: action column has zero rows — nothing to shift")
    shift_amt = shift % n
    rotated = rows[shift_amt:] + rows[:shift_amt]
    new_table = _rewrite_fixed_size_list_column(table, "action", rotated, width)
    tmp_path = parquet_path.parent / f".{parquet_path.name}.tmp"
    pq.write_table(new_table, tmp_path)
    os.replace(tmp_path, parquet_path)
    return dataset_dir


def overlap_modality_slice(
    dataset_dir: str | Path, *, key: str, scope: str = "state", by: int = 1
) -> Path:
    """Widens one range in `meta/modality.json` so it overlaps its
    neighbour by `by` indices, leaving the parquet untouched. Target:
    `consistency.modality_agreement`."""
    dataset_dir = Path(dataset_dir)
    modality_path = dataset_dir / "meta" / "modality.json"
    if not modality_path.exists():
        raise ValueError(f"no meta/modality.json at {modality_path} — nothing to overlap")
    with open(modality_path, encoding="utf-8") as f:
        payload = json.load(f)
    if scope not in payload or key not in payload[scope]:
        raise ValueError(f"modality.json has no {scope}.{key} entry to overlap")
    payload[scope][key]["end"] = payload[scope][key]["end"] + by
    tmp_path = modality_path.parent / f".{modality_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, modality_path)
    return dataset_dir


def corrupt_episodes_length(
    dataset_dir: str | Path, *, episode_index: int = 0, delta: int = 1
) -> Path:
    """Changes that row's `length` in `meta/episodes.jsonl` by `delta`.
    Target: `metadata.episode_length_agreement`."""
    dataset_dir = Path(dataset_dir)
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise ValueError(f"no meta/episodes.jsonl at {episodes_path} — nothing to corrupt")
    rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matched = False
    for row in rows:
        if row.get("episode_index") == episode_index:
            row["length"] = row["length"] + delta
            matched = True
            break
    if not matched:
        raise ValueError(f"episodes.jsonl has no row with episode_index={episode_index!r}")
    tmp_path = episodes_path.parent / f".{episodes_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, episodes_path)
    return dataset_dir


def perturb_histogram_counts(
    dataset_dir: str | Path,
    *,
    video_key: str,
    episode_index: int,
    bin_index: int,
    delta: int,
) -> Path:
    """Loads that episode's stats partial, adds `delta` to
    `histograms[video_key][0][bin_index]` (channel 0's bin `bin_index`),
    and rewrites the file with the same plain `json.dump` (no `sort_keys`)
    convention `write_partial_file`/`_force_constant_in_stats_partial`
    already use for this exact file, leaving every other key untouched. The
    one genuinely new mutation `03-09-PLAN.md` needs — no existing mutator
    reaches a persisted histogram (`perturb_stats_value` targets
    `meta/stats.json`; the constant-channel helpers target `features`, not
    `histograms`). Target: `statistics.image_histogram_integrity`."""
    dataset_dir = Path(dataset_dir)
    partial_path = dataset_dir / "meta" / "stats_partials" / f"episode_{episode_index:06d}.json"
    if not partial_path.exists():
        raise ValueError(f"no stats partial at {partial_path} — nothing to perturb")
    with open(partial_path, encoding="utf-8") as f:
        payload = json.load(f)
    histograms = payload.get("histograms", {})
    if video_key not in histograms:
        raise ValueError(f"stats partial at {partial_path} has no histogram for {video_key!r}")
    counts = histograms[video_key]
    if not counts or bin_index >= len(counts[0]):
        raise ValueError(
            f"stats partial at {partial_path}: histogram {video_key!r} has no bin {bin_index}"
        )
    counts[0][bin_index] = counts[0][bin_index] + delta
    tmp_path = partial_path.parent / f".{partial_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, partial_path)
    return dataset_dir


def _force_constant_in_stats_partial(
    dataset_dir: Path, episode_index: int, feature: str, channel_index: int, constant: float
) -> None:
    partial_path = (
        dataset_dir / "meta" / "stats_partials" / f"episode_{episode_index:06d}.json"
    )
    if not partial_path.exists():
        raise ValueError(f"no stats partial at {partial_path} — nothing to force constant")
    with open(partial_path, encoding="utf-8") as f:
        payload = json.load(f)
    entry = payload["features"][feature]
    count = entry["count"]
    entry["mean"][channel_index] = constant
    entry["m2"][channel_index] = 0.0
    entry["min"][channel_index] = constant
    entry["max"][channel_index] = constant
    entry["sum"][channel_index] = constant * count
    for value_row in entry["values"]:
        value_row[channel_index] = constant
    tmp_path = partial_path.parent / f".{partial_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, partial_path)


def _force_constant_in_merged_stats(
    dataset_dir: Path, feature: str, channel_index: int, constant: float
) -> None:
    stats_path = dataset_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise ValueError(f"no meta/stats.json at {stats_path} — nothing to force constant")
    with open(stats_path, encoding="utf-8") as f:
        payload = json.load(f)
    if feature not in payload:
        raise ValueError(f"stats.json has no {feature!r} entry to force constant")
    entry = payload[feature]
    for stat_key in ("mean", "min", "max", "q01", "q99"):
        entry[stat_key][channel_index] = constant
    entry["std"][channel_index] = 0.0
    tmp_path = stats_path.parent / f".{stats_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, stats_path)


def force_constant_channel(
    dataset_dir: str | Path,
    *,
    feature: str,
    channel_index: int,
    episode_index: int | None = None,
) -> Path:
    """Overwrites one channel with a single repeated value across every
    episode's parquet (or just `episode_index`, when given) and updates the
    corresponding retained values, min and max in each
    `meta/stats_partials/episode_*.json` and in `meta/stats.json` so the
    dataset-wide merged min equals its max for that channel. Target:
    `statistics.constant_channel`."""
    dataset_dir = Path(dataset_dir)
    episode_indices = (
        [episode_index] if episode_index is not None else _discovered_episode_indices(dataset_dir)
    )
    if not episode_indices:
        raise ValueError(f"{dataset_dir}: no episodes discovered to force a constant channel on")

    constant_value: float | None = None
    for idx in episode_indices:
        parquet_path = _require_parquet(dataset_dir, idx)
        table = pq.read_table(parquet_path)
        if feature not in table.schema.names:
            raise ValueError(f"parquet has no column {feature!r} to corrupt")
        field_type = table.schema.field(feature).type
        width = field_type.list_size
        rows = table.column(feature).to_pylist()
        if constant_value is None:
            constant_value = float(rows[0][channel_index]) if rows else 0.0
        for row in rows:
            row[channel_index] = constant_value
        new_table = _rewrite_fixed_size_list_column(table, feature, rows, width)
        tmp_path = parquet_path.parent / f".{parquet_path.name}.tmp"
        pq.write_table(new_table, tmp_path)
        os.replace(tmp_path, parquet_path)

        _force_constant_in_stats_partial(dataset_dir, idx, feature, channel_index, constant_value)

    assert constant_value is not None
    _force_constant_in_merged_stats(dataset_dir, feature, channel_index, constant_value)
    return dataset_dir
