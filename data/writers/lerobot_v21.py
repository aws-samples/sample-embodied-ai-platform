"""Writes a `CanonicalEpisode` to an on-disk LeRobot v2.1 dataset — the exact
files `gr00t.data.dataset.lerobot_episode_loader` reads at the pinned commit
(`.planning/research/DERISK-FINDINGS.md` § DERISK-04), never LeRobot's own
docs and never `training/sample_dataset/`'s file listing (both give the
wrong answer for this consumer — see that section's Pitfall 2).

Every feature width, name, and camera key is derived from the
`EmbodimentProfile` handed to the constructor — `observation.state`'s width
is the sum of `len(joint_names)` over `profile.state_vector_order`, `action`
the same over `profile.action_primary`, and the emitted camera set is every
`cameras` entry with `emit: true`, keyed by its `video_key`. No width,
camera name, or feature name is ever a literal in this module.

**Memory shape, stated explicitly (WRITE-02):** every `emit: true` camera's
PyAV encoder is kept open simultaneously for the duration of one episode —
three for the shipped Galaxea profile. The single frame pass yields one
frame per camera per tick, so re-opening a container per frame is not
possible for a streaming mux; the `ImageRef` contract already caps resident
decoded pixels at one frame at a time (never a whole episode), so the
memory cost of N open encoders is N cameras' worth of libav internal
buffers, not N cameras' worth of decoded frames.

**Codec policy (WRITE-03):** the default codec is `h264`. AV1 is an
explicit opt-in (`codec="av1"`) and is documented here as **unverified in
this configuration** — GR00T decodes video only through its own codec
stack (`torchcodec`, per `.planning/research/DERISK-FINDINGS.md`'s
Correction 6), and `training/sample_dataset/` being AV1 is not evidence
that AV1 loads under this project's own GR00T pin. `training/sample_dataset/`
is structural ground truth for feature and `video_info` *shape* only, never
for codec-choice or file-set decisions (Pitfall 2, same section).

`LeRobotV21Writer` satisfies `BaseDatasetWriter` (data/contracts/protocols.py)
structurally — this module never imports from, nor inherits, that Protocol.
"""

import dataclasses
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.contracts.containers import CanonicalEpisode, EpisodeWriteResult
from data.contracts.profile import EmbodimentProfile
from data.writers import modality, stats

_CHUNK_INDEX = 0
"""Single-chunk layout for Phase 2 — chunking across `chunk-{NNN}` boundaries
is out of this phase's scope."""

_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

ENCODE_MANIFEST_DIRNAME = "encode_manifests"
"""The one place the per-episode encode-manifest directory name lives
(D-14). `meta/{ENCODE_MANIFEST_DIRNAME}/episode_{NNNNNN:06d}.json` carries,
per emitted camera video_key, the writer's own measured `frames_submitted`/
`packets_muxed` counters — the source `data/validation/frame_integrity.py`
reads its dropped-frame count from, never a subtraction of totals derived
elsewhere."""

AUTO_TASK_OPT_IN = "auto"
"""The literal `--task auto` flag value — passed straight through as
`resolve_task_string`'s `explicit` argument. Deliberately not a separate
keyword parameter: this is what lets the whole D-09/D-10 policy live in a
3-parameter function with no CLI present (the CLI itself is Phase 4)."""

AUTO_TASK_SENTINEL = (
    "<<AUTO-UNLABELED>> no task string was supplied for this episode"
)
"""D-10's opt-in placeholder text — self-evidently not a language
instruction, so a reader or a downstream policy can never mistake it for
one. Every episode written under the `auto` opt-in carries this exact text
in its `meta/tasks.jsonl` row, each under its own distinct `task_index`."""


def resolve_task_string(
    explicit: str | None,
    *,
    stdin_isatty: bool,
    prompt: Callable[[], str] | None = None,
) -> str:
    """Resolves the task string used to label one episode (D-09/D-10).

    - `explicit == "auto"` (the literal `--task auto` opt-in value): returns
      `AUTO_TASK_SENTINEL`. This is the one case where `explicit` is not
      used verbatim — the flag value opts into a placeholder, it is never
      itself a real task description.
    - `explicit` is any other non-`None` string: used verbatim.
    - `explicit is None`: no task string was supplied at all. Per D-09,
      prompts through `prompt` when `stdin_isatty` is True; otherwise
      raises, naming the `--task auto` flag the operator has to type to
      opt into the placeholder. Never falls back to the placeholder by
      default — GR00T embeds the task text as the policy's language
      instruction, so a dataset silently labelled with a plausible-looking
      default trains on a meaningless prompt.
    """
    if explicit == AUTO_TASK_OPT_IN:
        return AUTO_TASK_SENTINEL
    if explicit is not None:
        return explicit
    if stdin_isatty:
        if prompt is None:
            raise ValueError(
                "stdin is a TTY but no interactive prompt callable was supplied"
            )
        return prompt()
    raise ValueError(
        "no task string was supplied for this episode and stdin is not "
        "interactive; pass an explicit task string, type the literal flag "
        "value 'auto' to opt into the self-evidently-invalid placeholder, "
        "or run interactively so the operator can be prompted"
    )


def _state_layout(profile: EmbodimentProfile) -> list[tuple[str, int]]:
    """`(state_key, width)` pairs in `profile.state_vector_order`'s order."""
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    return [
        (state_key, len(by_state_key[state_key].joint_names))
        for state_key in profile.state_vector_order
    ]


def _action_layout(profile: EmbodimentProfile) -> list[tuple[str, int]]:
    """`(action_key, width)` pairs in `profile.action_primary`'s order."""
    by_action_key = {f.action_key: f for f in profile.action_topics.values()}
    return [
        (action_key, len(by_action_key[action_key].joint_names))
        for action_key in profile.action_primary
    ]


def _assemble_vector(mapping: Any, layout: list[tuple[str, int]]) -> np.ndarray:
    """Concatenates `mapping`'s named entries in `layout`'s declared order
    into one `float32` vector. A layout key absent from `mapping` (should
    not happen for a required field, but the writer must not crash if it
    does) contributes a zero vector of the declared width."""
    if not layout:
        return np.zeros(0, dtype=np.float32)
    parts = []
    for key, width in layout:
        value = mapping.get(key) if hasattr(mapping, "get") else None
        if value is None:
            parts.append(np.zeros(width, dtype=np.float32))
        else:
            parts.append(np.asarray(value, dtype=np.float32).reshape(width))
    return np.concatenate(parts)


def _default_video_info(codec: str, target_fps: float | None) -> dict[str, Any]:
    """The `video_info` sub-object used before any real file has been
    encoded for a camera — `finalize()` must emit a fully-populated
    `features` map even after zero episodes (WRITE-04), so every emitted
    camera gets an entry regardless of whether any frame was ever muxed."""
    return {
        "video.height": 0,
        "video.width": 0,
        "video.codec": codec,
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": float(target_fps) if target_fps else 0.0,
        "video.channels": 3,
        "has_audio": False,
    }


def _read_video_info(path: Path) -> dict[str, Any]:
    """Reads codec, pixel format, dimensions, and frame rate back from the
    encoded file at `path` — never assumed from what the writer intended to
    write, so `info.json`'s `video_info` cannot drift from the actual bytes."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        return {
            "video.height": int(stream.codec_context.height),
            "video.width": int(stream.codec_context.width),
            "video.codec": str(stream.codec_context.name),
            "video.pix_fmt": str(stream.codec_context.pix_fmt),
            "video.is_depth_map": False,
            "video.fps": float(stream.average_rate) if stream.average_rate else 0.0,
            "video.channels": 3,
            "has_audio": False,
        }
    finally:
        container.close()


def _parse_jsonl_strict(path: Path, *, required_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Parses `path` line by line, returning every non-blank line's parsed
    JSON object in file order. Raises `ValueError` naming `path` and the
    1-based line number on a line that fails to parse as JSON, that does
    not parse to a JSON object, or that lacks any of `required_keys` — used
    by `_rehydrate_from_disk` (D-02) so a corrupted `meta/episodes.jsonl`/
    `meta/tasks.jsonl` line fails loud rather than silently truncating a
    resumed dataset's rehydrated totals."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: line {line_number}: could not parse as JSON: {e}"
                ) from e
            if not isinstance(row, dict) or any(key not in row for key in required_keys):
                raise ValueError(
                    f"{path}: line {line_number}: missing required key(s) among "
                    f"{required_keys!r} in {row!r}"
                )
            rows.append(row)
    return rows


@dataclasses.dataclass
class _CameraEncodeState:
    """One emitted camera's PyAV encode session for one episode. Opened lazily
    on the first frame that actually carries this camera's `ImageRef` — a
    camera absent from every frame of a non-empty episode never opens a
    container at all, so it is reported as absent in `output_paths` rather
    than as a zero-byte file (WRITE-02)."""

    video_key: str
    tmp_path: Path
    final_path: Path
    fps: float
    codec: str
    container: Any = None
    stream: Any = None
    time_base: Any = None
    frame_count: int = 0
    opened: bool = False
    packets_muxed: int = 0
    """D-14: incremented once per `self.container.mux(packet)` call — never
    per `encode_frame` call. `stream.encode()` legitimately yields zero or
    several packets for one input frame (B-frame reordering / GOP
    buffering), and that asymmetry between frames *submitted* and packets
    *muxed* is exactly what VALID-01 must be able to see."""

    def _open(self, height: int, width: int) -> None:
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        rate = int(round(self.fps)) if self.fps and self.fps > 0 else 1
        # format="mp4" explicit: the temp path's name does not end in .mp4
        # (it carries a .tmp suffix so it never collides with, or is
        # mistaken for, the final filename), so PyAV cannot infer the
        # container format from the extension alone.
        self.container = av.open(str(self.tmp_path), mode="w", format="mp4")
        self.stream = self.container.add_stream(self.codec, rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.time_base = Fraction(1, rate)
        # Assign BEFORE any encode() call — reading stream.time_base back
        # before assigning it raises AttributeError on None (documented
        # first failure mode, .planning/research/DERISK-FINDINGS.md
        # "Carried-forward pitfalls for Phase 2" (a)).
        self.stream.codec_context.time_base = self.time_base
        self.stream.options = {"g": "2", "crf": "30"}
        self.opened = True

    def encode_frame(self, rgb: np.ndarray) -> None:
        if not self.opened:
            self._open(height=rgb.shape[0], width=rgb.shape[1])
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = self.time_base  # the SAME Fraction object
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)
            self.packets_muxed += 1
        self.frame_count += 1

    def finalize(self) -> dict[str, Any] | None:
        """Flushes the encoder, closes the container, and atomically renames
        the temp file into place. Returns `None` (and discards the temp
        file) if zero frames were ever muxed for this camera."""
        if not self.opened or self.frame_count == 0:
            self.discard()
            return None
        for packet in self.stream.encode():  # mandatory delayed-frame flush
            self.container.mux(packet)
            self.packets_muxed += 1
        self.container.close()
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.tmp_path, self.final_path)
        return _read_video_info(self.final_path)

    def discard(self) -> None:
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def _assert_camera_row_parity(
    camera_states: dict[str, "_CameraEncodeState"], row_count: int, episode_index: int
) -> None:
    """The pre-commit half of the four-counter invariant
    `data/validation/frame_integrity.py::check_frame_count_agreement`
    completes post-write (VALID-01): that check can only see
    `frames_submitted` against `parquet_row_count` and the other two
    counters (`packets_muxed`, `decoded_frame_count`) after the fact,
    because `packets_muxed` needs the encoder flush and
    `decoded_frame_count` needs the finished file — this guard sees the
    `frames_submitted`-vs-row-count half the moment the frame loop ends,
    before anything is written.

    Scoped deliberately to cameras whose encode state was actually
    **opened** — meaning they submitted at least one frame. A camera that
    submitted zero frames for the whole episode is a different condition:
    no video file is produced for it at all, so no frame index can ever
    point into a short video, and several existing writer tests
    (`data/tests/test_writer_tracer.py`, `data/tests/test_writers_lerobot_v21.py`)
    deliberately build episodes carrying images for only a SUBSET of the
    profile's emitted cameras via `emit_cameras` — narrowing to opened
    states is what lets this guard catch the real defect's shape (300
    submitted against 304 rows) without breaking that pre-existing,
    intentional coverage. The absent-video case remains reported by
    `check_frame_count_agreement`, which fails it fatally post-write.

    Task 1's grid truncation (`data/sync/master_clock.py`) plus Task 2's
    pre-write rejection (`data/adapters/galaxea.py::_enforce_camera_coverage`)
    make this path unreachable for any episode the shipped adapter
    produces — it exists so the invariant holds by construction for any
    future caller, writer path, or hand-built episode; its firing means a
    bug upstream, never a data-quality verdict, which is why it raises
    rather than grading.
    """
    for video_key, cam_state in sorted(camera_states.items()):
        if not cam_state.opened:
            continue
        if cam_state.frame_count != row_count:
            for other in camera_states.values():
                other.discard()
            raise ValueError(
                f"episode {episode_index}: camera {video_key!r} submitted "
                f"{cam_state.frame_count} frame(s) but the parquet has "
                f"{row_count} row(s) -- refusing to commit a camera whose "
                "frame count disagrees with the episode's row count"
            )


class LeRobotV21Writer:
    """Streams `CanonicalEpisode`s to a LeRobot v2.1 dataset directory.

    `write_episode` assigns `episode_index` and the dataset-wide,
    strictly-increasing `index` column solely at the writer — the episode
    handed in never carries a batch position, so an adapter never learns
    about one.
    """

    def __init__(
        self,
        out_dir: str | Path,
        profile: EmbodimentProfile,
        *,
        codec: str = "h264",
        emit_episodes_stats: bool = False,
        prompt: Callable[[], str] | None = None,
        stdin_isatty: Callable[[], bool] | None = None,
        resume: bool = False,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._profile = profile
        self._codec = codec
        self._emit_episodes_stats = emit_episodes_stats
        self._prompt = prompt
        self._stdin_isatty = stdin_isatty or (lambda: sys.stdin.isatty())
        self._episodes_meta: list[dict[str, Any]] = []
        self._tasks_meta: list[dict[str, Any]] = []

        self._state_layout = _state_layout(profile)
        self._action_layout = _action_layout(profile)
        self._state_width = sum(w for _, w in self._state_layout)
        self._action_width = sum(w for _, w in self._action_layout)
        self._state_names = [
            joint
            for state_key, _ in self._state_layout
            for joint in self._joint_names_for_state(state_key)
        ]
        self._action_names = [
            joint
            for action_key, _ in self._action_layout
            for joint in self._joint_names_for_action(action_key)
        ]

        self._emitted_video_keys = [
            cam.video_key
            for cam in profile.cameras.values()
            if cam.emit and cam.video_key is not None
        ]

        target_fps = profile.master_clock.target_fps
        self._camera_video_info: dict[str, dict[str, Any]] = {
            vk: _default_video_info(codec, target_fps) for vk in self._emitted_video_keys
        }

        self._next_episode_index = 0
        self._next_dataset_index = 0
        self._last_fps_seen: float | None = None
        self._total_videos = 0

        # WRITE-06: dataset-wide merge over every included episode's
        # per-feature partial — never re-scans a parquet, re-decodes a
        # video, or re-reads an MCAP (D-06).
        self._stats_merged: dict[str, stats.WelfordPartial] = {
            "observation.state": stats.WelfordPartial.empty(self._state_width),
            "action": stats.WelfordPartial.empty(self._action_width),
            "timestamp": stats.WelfordPartial.empty(1),
        }

        # RESEARCH.md Open Question 7: cumulative-since-construction wall-clock
        # seconds spent decoding camera JPEGs versus encoding video frames —
        # NEVER per-episode. See the `decode_seconds`/`encode_seconds`
        # properties' own docstrings for why a caller wanting a per-episode
        # figure must difference two readings across one `write_episode` call.
        self._decode_seconds = 0.0
        self._encode_seconds = 0.0

        if resume:
            # D-02: rehydrates _episodes_meta/_tasks_meta/_next_episode_index/
            # _next_dataset_index/_stats_merged/_total_videos from what is
            # already on disk under out_dir — the LAST statement of __init__,
            # so rehydration overwrites the empty/zeroed values above rather
            # than racing them.
            self._rehydrate_from_disk()

    def _rehydrate_from_disk(self) -> None:
        """D-02's opt-in resume rehydration. Reads exactly four things from
        `self._out_dir` and nothing else — never a parquet scan, never a
        video re-decode, never an MCAP re-read:

        1. `meta/episodes.jsonl`, if present: extends `self._episodes_meta`
           in file order and sets `self._next_episode_index` to
           `max(episode_index) + 1` (never the row count — a gapped prefix,
           e.g. a reclaimed index a later run has not yet refilled, must not
           hand back an already-occupied index) and `self._next_dataset_index`
           to the summed `length` across those rows.
        2. `meta/tasks.jsonl`, if present: extends `self._tasks_meta` in
           file order.
        3. `meta/stats_partials/episode_*.json`, sorted by filename: for
           each file whose `included_in_merge` is true, rebuilds a
           `WelfordPartial` per feature (including its retained `values` —
           load-bearing for `finalize_stats`'s `q01`/`q99`) and folds it
           into `self._stats_merged` via `stats.merge_partials`. A feature
           name absent from `self._stats_merged` is skipped rather than
           inventing a partial for it.
        4. `meta/info.json`, if present: for each `video_key` in
           `self._emitted_video_keys` whose `features[video_key].video_info`
           is present, replaces `self._camera_video_info[video_key]` with
           that persisted mapping. This is metadata-only — a plain
           `json.load`, never a video re-decode — and it is what
           `_build_features_map()` (the sole reader of
           `self._camera_video_info`) writes back into a resumed
           `finalize()`'s `meta/info.json`. Without it, a resume that writes
           zero new episodes never calls `write_episode` (the only other
           writer of `_camera_video_info`), so `finalize()` would clobber
           the real, previously-measured `video.height`/`video.width` back
           down to `_default_video_info`'s zero-valued placeholder — closing
           WINDOWS.md id 9 (`statistics.image_histogram_integrity`'s
           `mass_identity` finding with `expected_mass=0` on a second
           validated pass over an already-converted prefix). A missing
           `meta/info.json`, a missing `features` key, a missing camera
           entry, or a missing `video_info` all leave the
           `_default_video_info` placeholder in place — a genuinely absent
           dimension stays visibly zero rather than being invented.

        Also sets `self._total_videos` to the count of MP4 files already
        present under `videos/chunk-{_CHUNK_INDEX:03d}/*/`. Leaves
        `self._last_fps_seen` as `None` — `finalize()` already falls back to
        the profile's `master_clock.target_fps` when it is `None`, and a
        resumed run that writes at least one episode sets it from that
        episode.

        Raises `ValueError` naming the file and the 1-based line number on a
        malformed or field-missing `episodes.jsonl`/`tasks.jsonl` line (see
        `_parse_jsonl_strict`).
        """
        meta_dir = self._out_dir / "meta"

        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            rows = _parse_jsonl_strict(episodes_path, required_keys=("episode_index", "length"))
            self._episodes_meta.extend(rows)
            if rows:
                self._next_episode_index = max(row["episode_index"] for row in rows) + 1
                self._next_dataset_index = sum(row["length"] for row in rows)

        tasks_path = meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            rows = _parse_jsonl_strict(tasks_path, required_keys=("task_index", "task"))
            self._tasks_meta.extend(rows)

        partials_dir = meta_dir / "stats_partials"
        if partials_dir.exists():
            for partial_path in sorted(partials_dir.glob("episode_*.json")):
                with open(partial_path, encoding="utf-8") as f:
                    payload = json.load(f)
                if not payload.get("included_in_merge", False):
                    continue
                for name, feature_payload in payload.get("features", {}).items():
                    if name not in self._stats_merged:
                        continue
                    values = np.asarray(feature_payload["values"], dtype=np.float32)
                    partial = stats.WelfordPartial.from_json_dict(feature_payload, values=values)
                    self._stats_merged[name] = stats.merge_partials(
                        self._stats_merged[name], partial
                    )

        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            features = info.get("features", {})
            for video_key in self._emitted_video_keys:
                feature = features.get(video_key)
                if feature is None:
                    continue
                video_info = feature.get("video_info")
                if video_info is None:
                    continue
                self._camera_video_info[video_key] = dict(video_info)

        videos_dir = self._out_dir / "videos" / f"chunk-{_CHUNK_INDEX:03d}"
        if videos_dir.exists():
            self._total_videos = sum(1 for _ in videos_dir.glob("*/*.mp4"))

    def _joint_names_for_state(self, state_key: str) -> list[str]:
        by_state_key = {f.state_key: f for f in self._profile.state_topics.values()}
        return list(by_state_key[state_key].joint_names)

    def _joint_names_for_action(self, action_key: str) -> list[str]:
        by_action_key = {f.action_key: f for f in self._profile.action_topics.values()}
        return list(by_action_key[action_key].joint_names)

    @property
    def decode_seconds(self) -> float:
        """Cumulative wall-clock seconds spent in every emitted camera's
        `ImageRef.decode()` call, summed since this writer was constructed —
        **cumulative, not per-episode**. A caller wanting one episode's own
        decode cost must read this property immediately before and
        immediately after that episode's own `write_episode` call and take
        the difference; recording the raw value once per episode yields a
        monotonically inflating series that looks plausible and is wrong
        (RESEARCH.md Open Question 7)."""
        return self._decode_seconds

    @property
    def encode_seconds(self) -> float:
        """Cumulative wall-clock seconds spent in every emitted camera's
        `_CameraEncodeState.encode_frame()` call, summed since this writer
        was constructed — **cumulative, not per-episode**, for the exact
        same reason as `decode_seconds`: difference two readings across one
        `write_episode` call to get that episode's own figure."""
        return self._encode_seconds

    @property
    def episodes_meta_count(self) -> int:
        """`len(self._episodes_meta)` — the number of episode rows this
        writer will serialize into `meta/episodes.jsonl` at `finalize()`
        time, exposed read-only so a caller (e.g. `data/cli.py`'s WR-02
        exception guard) can assert this stays equal to `next_episode_index`
        without reaching into this writer's private state (CR-01,
        04-REVIEW.md fix option (c))."""
        return len(self._episodes_meta)

    @property
    def next_episode_index(self) -> int:
        """The episode index the *next* `write_episode` call will be
        assigned — read-only counterpart to `episodes_meta_count` for the
        same CR-01 invariant check."""
        return self._next_episode_index

    def write_episode(self, episode: CanonicalEpisode) -> EpisodeWriteResult:
        episode_index = self._next_episode_index
        fps = episode.header.fps

        camera_states: dict[str, _CameraEncodeState] = {
            video_key: _CameraEncodeState(
                video_key=video_key,
                tmp_path=self._out_dir
                / f"videos/chunk-{_CHUNK_INDEX:03d}/{video_key}/.episode_{episode_index:06d}.mp4.tmp",
                final_path=self._out_dir
                / _VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=episode_index
                ),
                fps=fps,
                codec=self._codec,
            )
            for video_key in self._emitted_video_keys
        }

        state_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        timestamp_rows: list[float] = []
        frame_index_rows: list[int] = []
        index_rows: list[int] = []
        task_index_rows: list[int] = []

        # task_index: D-11, one index per episode, no cross-episode dedup in
        # Phase 2.
        task_index = episode_index

        # WRITE-06: one Welford update per float feature per frame, and one
        # uint8 histogram update per emitted camera per frame — no second
        # pass over anything.
        episode_partials: dict[str, stats.WelfordPartial] = {
            "observation.state": stats.WelfordPartial.empty(self._state_width),
            "action": stats.WelfordPartial.empty(self._action_width),
            "timestamp": stats.WelfordPartial.empty(1),
        }
        camera_histograms: dict[str, stats.Uint8Histogram] = {
            video_key: stats.Uint8Histogram.empty(3) for video_key in self._emitted_video_keys
        }

        completed = False
        try:
            frame_count = 0
            for frame in episode.frames:
                state_vec = _assemble_vector(frame.state, self._state_layout)
                action_vec = _assemble_vector(frame.action, self._action_layout)
                state_rows.append(state_vec)
                action_rows.append(action_vec)
                # Quantize to float32 ONCE, here, and feed that exact value to
                # both the parquet row and the Welford partial. `_build_table`
                # writes this column as `pa.float32()` (never float64), and
                # `state_vec`/`action_vec` are already float32 by construction
                # (`_assemble_vector`) before their own `.update()` calls below
                # — computing this partial from the pre-quantization float64
                # value would silently desync `meta/stats.json` from what is
                # actually persisted in the parquet file (VALID-09 finding).
                timestamp_value = float(np.float32(frame.timestamp_ns / 1e9))
                timestamp_rows.append(timestamp_value)
                frame_index_rows.append(frame.frame_index)
                index_rows.append(self._next_dataset_index + frame_count)
                task_index_rows.append(task_index)

                episode_partials["observation.state"].update(state_vec)
                episode_partials["action"].update(action_vec)
                episode_partials["timestamp"].update(
                    np.array([timestamp_value], dtype=np.float64)
                )

                for video_key, cam_state in camera_states.items():
                    image_ref = frame.images.get(video_key)
                    if image_ref is None:
                        continue
                    decode_start = time.monotonic()
                    rgb = image_ref.decode()
                    self._decode_seconds += time.monotonic() - decode_start
                    camera_histograms[video_key].update(rgb)
                    encode_start = time.monotonic()
                    cam_state.encode_frame(rgb)
                    self._encode_seconds += time.monotonic() - encode_start
                    del rgb

                frame_count += 1
            completed = True
        finally:
            if not completed:
                for cam_state in camera_states.values():
                    cam_state.discard()

        output_paths: dict[str, str] = {}

        if frame_count == 0:
            for cam_state in camera_states.values():
                cam_state.discard()
        else:
            _assert_camera_row_parity(camera_states, frame_count, episode_index)
            manifest_entries: dict[str, dict[str, int]] = {}
            finalized_video_paths: list[Path] = []
            for video_key, cam_state in camera_states.items():
                info = cam_state.finalize()
                if info is not None:
                    output_paths[video_key] = str(cam_state.final_path)
                    finalized_video_paths.append(cam_state.final_path)
                    self._camera_video_info[video_key] = info
                    self._total_videos += 1
                if cam_state.opened:
                    # D-14: every camera that had an open encode state gets a
                    # manifest entry, including one whose finalize() returned
                    # None (zero packets muxed) — that is exactly the case
                    # VALID-01 must not lose.
                    manifest_entries[video_key] = {
                        "frames_submitted": cam_state.frame_count,
                        "packets_muxed": cam_state.packets_muxed,
                    }

            manifest_path = (
                self._out_dir
                / "meta"
                / ENCODE_MANIFEST_DIRNAME
                / f"episode_{episode_index:06d}.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_manifest_path = manifest_path.parent / f".{manifest_path.name}.tmp"
            with open(tmp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_entries, f, sort_keys=True, indent=2)
            os.replace(tmp_manifest_path, manifest_path)
            output_paths["encode_manifest"] = str(manifest_path)

            table = self._build_table(
                state_rows,
                action_rows,
                timestamp_rows,
                frame_index_rows,
                index_rows,
                episode_index,
                task_index_rows,
            )
            final_parquet_path = (
                self._out_dir
                / _DATA_PATH_TEMPLATE.format(
                    episode_chunk=_CHUNK_INDEX, episode_index=episode_index
                )
            )
            final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_parquet_path = final_parquet_path.parent / f".{final_parquet_path.name}.tmp"
            pq.write_table(table, tmp_parquet_path)
            os.replace(tmp_parquet_path, final_parquet_path)
            output_paths["parquet"] = str(final_parquet_path)

            task_string = resolve_task_string(
                episode.header.task_prompt,
                stdin_isatty=self._stdin_isatty(),
                prompt=self._prompt,
            )
            self._tasks_meta.append({"task_index": task_index, "task": task_string})
            self._episodes_meta.append(
                {"episode_index": episode_index, "tasks": [task_string], "length": frame_count}
            )

            # CR-01 (04-REVIEW.md): everything from here through
            # `_next_episode_index += 1` below MUST be transactional with
            # respect to the `_tasks_meta`/`_episodes_meta` appends just
            # above. The parquet is already durably committed via
            # `os.replace` at this point, but the writer's own counters have
            # NOT advanced yet. If `stats.write_partial_file`/
            # `merge_partials` raises here (e.g. a transient EFS I/O fault —
            # exactly the class of fault `data/cli.py`'s WR-02 guard exists
            # to survive), a bare `except Exception` upstream resumes the
            # batch and the *next* successful `write_episode` call reuses
            # this same `episode_index` (because `_next_episode_index` never
            # advanced), appending a SECOND `_episodes_meta` row at the same
            # index while this failed call's row is never removed — two
            # `episodes.jsonl` rows at one index, only one of them backed by
            # a real parquet file. Roll back the just-appended meta rows and
            # this call's committed artifacts on any failure in this region,
            # then re-raise so the caller's guard still observes the
            # failure and can record the input as rejected.
            partial_path = (
                self._out_dir / "meta" / "stats_partials" / f"episode_{episode_index:06d}.json"
            )
            try:
                disposition = episode.header.global_meta.get("disposition")
                included = stats.disposition_included_in_merge(disposition)
                stats.write_partial_file(
                    partial_path,
                    disposition=disposition,
                    included_in_merge=included,
                    feature_partials=episode_partials,
                    histograms=camera_histograms,
                )
                output_paths["stats_partial"] = str(partial_path)
                if included:
                    # Build the merged result in a scratch dict first and
                    # only publish it into `self._stats_merged` once every
                    # feature has merged successfully — otherwise a raise
                    # partway through this loop would leave some features
                    # merged (mutated in place) and others not, which no
                    # rollback below could undo.
                    merged_partials: dict[str, stats.WelfordPartial] = {}
                    for name, partial in episode_partials.items():
                        merged_partials[name] = stats.merge_partials(
                            self._stats_merged[name], partial
                        )
                    self._stats_merged.update(merged_partials)
            except Exception:
                self._tasks_meta.pop()
                self._episodes_meta.pop()
                self._total_videos -= len(finalized_video_paths)
                for stale_path in (
                    partial_path,
                    final_parquet_path,
                    manifest_path,
                    *finalized_video_paths,
                ):
                    try:
                        stale_path.unlink(missing_ok=True)
                    except OSError:
                        # Best-effort disk cleanup — never let a secondary
                        # failure here mask the original exception being
                        # re-raised below.
                        pass
                raise

            self._last_fps_seen = fps
            self._next_dataset_index += frame_count
            # WRITE-01/WRITE-08: only a real write consumes an episode index.
            # Advancing this for a zero-frame episode would leave a gap in the
            # episode_{NNNNNN}.parquet sequence that GR00T's loader reads as a
            # missing episode, and would desync info.json's totals from the
            # episodes.jsonl/tasks.jsonl rows appended just above.
            self._next_episode_index += 1

        return EpisodeWriteResult(
            episode_index=episode_index,
            frame_count=frame_count,
            output_paths=output_paths,
        )

    def _build_table(
        self,
        state_rows: list[np.ndarray],
        action_rows: list[np.ndarray],
        timestamp_rows: list[float],
        frame_index_rows: list[int],
        index_rows: list[int],
        episode_index: int,
        task_index_rows: list[int],
    ) -> pa.Table:
        """Builds the parquet `pyarrow.Table` directly, with an explicit
        schema declaring `observation.state`/`action` as fixed-size float32
        list columns — never round-tripped through a pandas `DataFrame`
        (Arrow list columns decay to Python-object columns there, and
        Python float literals infer as `float64`, silently upcasting the
        state/action dtype)."""
        n = len(state_rows)
        state_flat = pa.array(
            np.concatenate(state_rows) if n else np.zeros(0, dtype=np.float32),
            type=pa.float32(),
        )
        action_flat = pa.array(
            np.concatenate(action_rows) if n else np.zeros(0, dtype=np.float32),
            type=pa.float32(),
        )
        state_col = pa.FixedSizeListArray.from_arrays(state_flat, self._state_width)
        action_col = pa.FixedSizeListArray.from_arrays(action_flat, self._action_width)

        schema = pa.schema(
            [
                pa.field("observation.state", pa.list_(pa.float32(), self._state_width)),
                pa.field("action", pa.list_(pa.float32(), self._action_width)),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
            ]
        )
        return pa.Table.from_arrays(
            [
                state_col,
                action_col,
                pa.array(np.asarray(timestamp_rows, dtype=np.float32)),
                pa.array(np.asarray(frame_index_rows, dtype=np.int64)),
                pa.array(np.full(n, episode_index, dtype=np.int64)),
                pa.array(np.asarray(index_rows, dtype=np.int64)),
                pa.array(np.asarray(task_index_rows, dtype=np.int64)),
            ],
            schema=schema,
        )

    def _build_features_map(self) -> dict[str, Any]:
        """Every declared feature's shape/dtype/names, derived solely from
        `self._profile` — never from observed data, so this is fully
        populated even when called after zero episodes have been written
        (WRITE-04)."""
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self._state_width],
                "names": self._state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [self._action_width],
                "names": self._action_names,
            },
        }
        for video_key in self._emitted_video_keys:
            info = self._camera_video_info[video_key]
            features[video_key] = {
                "dtype": "video",
                "shape": [info["video.height"], info["video.width"], 3],
                "names": ["height", "width", "channels"],
                "video_info": dict(info),
            }
        for name, dtype in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ):
            features[name] = {"dtype": dtype, "shape": [1], "names": None}
        return features

    def finalize(self) -> None:
        """Writes `meta/info.json` — `codebase_version` exactly `v2.1`, the
        profile-derived `features` map, the `data_path`/`video_path`
        templates, `fps`, and the running episode/frame/chunk totals.

        The totals below are already correct across a resume (D-02) with NO
        change to this method: they count rows actually appended to
        `self._episodes_meta`/`self._tasks_meta` and the dataset-wide
        `self._next_dataset_index` counter, never a literal — and
        `_rehydrate_from_disk` (called from `__init__` when `resume=True`)
        is what makes those lists/counters include the prior run's state
        before this method ever reads them. Do not "fix" this derivation to
        special-case a resumed run; that would be redundant at best and
        wrong at worst."""
        meta_dir = self._out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        target_fps = self._profile.master_clock.target_fps
        fps = self._last_fps_seen if self._last_fps_seen is not None else (target_fps or 0.0)

        info: dict[str, Any] = {
            "codebase_version": "v2.1",
            "robot_type": self._profile.robot_type,
            # Counted off the rows actually appended, not off the running
            # index counter — so these totals cannot drift from
            # episodes.jsonl/tasks.jsonl even if a future caller finds another
            # way to advance the counter without writing. total_tasks equals
            # total_episodes by D-11 (one task index per episode, no
            # cross-episode dedup in Phase 2), so it is counted the same way.
            "total_episodes": len(self._episodes_meta),
            "total_frames": self._next_dataset_index,
            "total_tasks": len(self._tasks_meta),
            "total_videos": self._total_videos,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "data_path": _DATA_PATH_TEMPLATE,
            "video_path": _VIDEO_PATH_TEMPLATE,
            "features": self._build_features_map(),
        }

        info_path = meta_dir / "info.json"
        tmp_info_path = meta_dir / ".info.json.tmp"
        with open(tmp_info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        os.replace(tmp_info_path, info_path)

        self._write_jsonl(meta_dir / "tasks.jsonl", self._tasks_meta)
        self._write_jsonl(meta_dir / "episodes.jsonl", self._episodes_meta)

        # WRITE-06: meta/stats.json covers exactly the features whose
        # info.json dtype string contains "float" — derived programmatically
        # from the features map this finalize() call just built, never from
        # a hardcoded name list. No camera feature (dtype "video") is ever a
        # key here, by construction of that same substring test.
        float_feature_names = sorted(
            name for name, feat in info["features"].items() if "float" in feat["dtype"]
        )
        stats_payload: dict[str, Any] = {}
        for name in float_feature_names:
            partial = self._stats_merged.get(name)
            if partial is None or partial.count == 0:
                continue
            stats_payload[name] = stats.finalize_stats(partial)

        stats_path = meta_dir / "stats.json"
        tmp_stats_path = meta_dir / ".stats.json.tmp"
        with open(tmp_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, sort_keys=True, indent=2)
        os.replace(tmp_stats_path, stats_path)

        # WRITE-07: meta/modality.json + the MODALITY_CONFIG_PATH Python
        # module, generated from this same self._profile — then the
        # existence guard that makes the absent-file state unreachable for
        # any dataset this writer finalizes. N1.7's finetune script writes
        # an ungated hardcoded 6-DoF SO-101 slice map when modality.json is
        # missing, silently mis-slicing a higher-DoF dataset rather than
        # failing (see data/writers/modality.py's module docstring).
        modality.write_modality_artifacts(self._profile, self._out_dir, fps=fps)
        modality.assert_modality_artifacts_present(self._out_dir)

        if self._emit_episodes_stats:
            # WRITE-05: GR00T's loader never reads this file by name — it is
            # the LeRobot-conventional artifact `training/sample_dataset/`
            # ships, emitted here only on an explicit opt-in, never by
            # default. Row shape carries *finished* per-episode stats (not
            # a mergeable partial — see data/writers/stats.py's own
            # docstring for why the two are deliberately not the same
            # shape).
            self._write_jsonl(
                meta_dir / "episodes_stats.jsonl",
                [{"episode_index": row["episode_index"]} for row in self._episodes_meta],
            )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        tmp_path = path.parent / f".{path.name}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp_path, path)


_WRITER_REGISTRY: dict[str, Callable[..., Any]] = {}


def _v3_factory(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "the v3 dataset format is deferred to a later milestone — this "
        "engine writes v2.1 only (format key 'v2.1')"
    )


_WRITER_REGISTRY["v2.1"] = LeRobotV21Writer
_WRITER_REGISTRY["v3"] = _v3_factory


def get_writer(format_key: str, *args: Any, **kwargs: Any) -> Any:
    """Resolves `format_key` to a dataset writer via the format-keyed
    registry — the primary entry point (WRITE-10): the registry is the
    noun, `LeRobotV21Writer` is one registered entry, not a privileged
    default with a v3 guard clause bolted on.

    Keys are matched by exact string equality — no case-folding, no
    whitespace-stripping. A registered-but-deferred key (`v3`) raises
    `NotImplementedError` naming the deferred milestone; an unregistered
    or malformed key raises `ValueError` naming every registered key.
    """
    try:
        factory = _WRITER_REGISTRY[format_key]
    except KeyError:
        raise ValueError(
            f"unknown dataset format {format_key!r}; registered formats: "
            f"{sorted(_WRITER_REGISTRY)}"
        ) from None
    return factory(*args, **kwargs)
