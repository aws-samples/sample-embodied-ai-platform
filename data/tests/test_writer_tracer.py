"""Track B's end-to-end tracer: one synthetic episode, written through
`LeRobotV21Writer`, read back off disk — never a per-layer unit test. Uses
the real shipped Galaxea profile for state/action width, but only ONE
emitted camera (the master-clock camera) so the whole layer wiring is
proven with no batching and no second call site.

Zero dependency on `data.adapters` — this module and everything it imports
never touches a real MCAP file or the Galaxea adapter.
"""

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.contracts.containers import CanonicalEpisode, EpisodeWriteResult
from data.contracts.profile import load_profile
from data.tests.frame_fixtures import build_synthetic_episode
from data.writers.lerobot_v21 import LeRobotV21Writer

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
PROFILE = load_profile(PROFILE_PATH)
MASTER_VIDEO_KEY = "observation.images.head"

EXPECTED_STATE_WIDTH = 20  # sum over state_vector_order's joint_names, per this profile
EXPECTED_PARQUET_COLUMNS = {
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


# --- CONT-02/03 protocol conformance: no-inheritance, mypy-checked
# annotated assignment, following test_contracts.py's StubWriter pattern.


class _StubWriterForTracer:
    def write_episode(self, episode: CanonicalEpisode) -> EpisodeWriteResult:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError


def test_lerobot_v21_writer_satisfies_base_dataset_writer_with_no_inheritance() -> None:
    from data.contracts.protocols import BaseDatasetWriter

    stub: BaseDatasetWriter = _StubWriterForTracer()
    assert stub is not None
    assert LeRobotV21Writer.__mro__ == (LeRobotV21Writer, object)
    assert "data.contracts" not in LeRobotV21Writer.__module__


def test_tracer_end_to_end(tmp_path: Path) -> None:
    episode = build_synthetic_episode(
        frame_count=5,
        fps=25.0,
        profile=PROFILE,
        emit_cameras=[MASTER_VIDEO_KEY],
        task_prompt="pick up the orange",
    )

    writer = LeRobotV21Writer(tmp_path, PROFILE)
    result = writer.write_episode(episode)
    writer.finalize()

    assert result.episode_index == 0
    assert result.frame_count == 5

    info_path = tmp_path / "meta" / "info.json"
    assert info_path.exists()
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    assert info["codebase_version"] == "v2.1"

    parquet_path = tmp_path / info["data_path"].format(episode_chunk=0, episode_index=0)
    assert parquet_path.exists()
    assert parquet_path == tmp_path / "data/chunk-000/episode_000000.parquet"

    table = pq.read_table(parquet_path)
    assert set(table.schema.names) == EXPECTED_PARQUET_COLUMNS
    state_type = table.schema.field("observation.state").type
    assert pa.types.is_fixed_size_list(state_type)
    assert state_type.value_type == pa.float32()
    assert state_type.list_size == EXPECTED_STATE_WIDTH

    frame_index_col = table.column("frame_index").to_pylist()
    assert frame_index_col == sorted(frame_index_col)
    assert frame_index_col == list(range(5))

    video_path = tmp_path / info["video_path"].format(
        episode_chunk=0, video_key=MASTER_VIDEO_KEY, episode_index=0
    )
    assert video_path.exists()
    probe = subprocess.run(
        [
            "ffprobe",
            "-count_frames",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(probe.stdout.strip()) == 5


def test_zero_frame_camera_absent_from_output_paths(tmp_path: Path) -> None:
    """Only one camera (`emit_cameras=[MASTER_VIDEO_KEY]`) ever appears in a
    frame, so the other two `emit: true` cameras never open a container and
    must be absent from `output_paths` rather than pointing at a zero-byte
    file."""
    episode = build_synthetic_episode(
        frame_count=3, fps=25.0, profile=PROFILE, emit_cameras=[MASTER_VIDEO_KEY], task_prompt="t"
    )
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    result = writer.write_episode(episode)
    writer.finalize()

    assert MASTER_VIDEO_KEY in result.output_paths
    assert "observation.images.wrist_left" not in result.output_paths
    assert "observation.images.wrist_right" not in result.output_paths
    assert not (tmp_path / "videos/chunk-000/observation.images.wrist_left").exists()


def test_zero_frame_episode_writes_no_parquet(tmp_path: Path) -> None:
    episode = build_synthetic_episode(
        frame_count=0, fps=25.0, profile=PROFILE, emit_cameras=[MASTER_VIDEO_KEY]
    )
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    result = writer.write_episode(episode)

    assert result.frame_count == 0
    assert result.output_paths == {}
    assert not (tmp_path / "data/chunk-000/episode_000000.parquet").exists()


def test_zero_frame_episode_does_not_consume_an_episode_index(tmp_path: Path) -> None:
    """WRITE-01/WRITE-08 regression: a zero-frame episode must not burn an
    episode index. If it does, the next real episode lands at
    `episode_000001.parquet` while `episode_000000.parquet` never exists —
    a gap GR00T's loader reads as a missing episode — and `info.json`'s
    `total_episodes`/`total_tasks` overcount the actual jsonl rows.

    The pre-existing zero-frame test only ever called `write_episode()` once,
    so it could not observe the desync; this one chains an empty write with a
    real one, which is the sequence Phase 4's batch CLI will drive.
    """
    empty = build_synthetic_episode(
        frame_count=0, fps=25.0, profile=PROFILE, emit_cameras=[MASTER_VIDEO_KEY]
    )
    real = build_synthetic_episode(
        frame_count=3, fps=25.0, profile=PROFILE, emit_cameras=[MASTER_VIDEO_KEY], task_prompt="t"
    )

    writer = LeRobotV21Writer(tmp_path, PROFILE)
    empty_result = writer.write_episode(empty)
    real_result = writer.write_episode(real)
    writer.finalize()

    # The skipped episode released its index back to the next real write.
    assert empty_result.frame_count == 0
    assert real_result.episode_index == 0
    assert (tmp_path / "data/chunk-000/episode_000000.parquet").exists()
    assert not (tmp_path / "data/chunk-000/episode_000001.parquet").exists()

    # info.json totals agree with the rows actually on disk.
    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes_rows = [
        line
        for line in (tmp_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks_rows = [
        line
        for line in (tmp_path / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert info["total_episodes"] == len(episodes_rows) == 1
    assert info["total_tasks"] == len(tasks_rows) == 1
    assert info["total_frames"] == 3

    # episode_index is contiguous from 0 across the surviving rows.
    assert [json.loads(line)["episode_index"] for line in episodes_rows] == [0]


def test_interrupted_write_leaves_no_partial_parquet(tmp_path: Path) -> None:
    from data.contracts.containers import CanonicalEpisode, CanonicalFrame, EpisodeHeader

    def _raising_frames():
        for i in range(2):
            yield CanonicalFrame(
                timestamp_ns=i, frame_index=i, images={}, state={}, action={}
            )
        raise RuntimeError("simulated interruption mid-episode")

    header = EpisodeHeader(
        episode_id="interrupted", fps=25.0, robot_type=PROFILE.robot_type, task_prompt=None
    )
    episode = CanonicalEpisode(header=header, frames=_raising_frames())

    writer = LeRobotV21Writer(tmp_path, PROFILE)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        writer.write_episode(episode)

    assert not (tmp_path / "data/chunk-000/episode_000000.parquet").exists()


def test_fixture_frames_are_single_pass(tmp_path: Path) -> None:
    episode = build_synthetic_episode(
        frame_count=3, fps=25.0, profile=PROFILE, emit_cameras=[MASTER_VIDEO_KEY]
    )
    first_pass = list(episode.frames)
    assert len(first_pass) == 3
    second_pass = list(episode.frames)
    assert second_pass == []
