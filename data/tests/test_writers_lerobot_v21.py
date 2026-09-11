"""Expansion tests for `LeRobotV21Writer`: every emitted camera, the meta
files GR00T's loader actually reads, the task-string policy (D-09/D-10), and
the format-keyed writer registry (WRITE-10).

RED phase: every test below fails against the tracer-only `lerobot_v21.py`
(no registry, no `resolve_task_string`, only one camera ever gets images in
the tracer's own fixture calls, no `meta/tasks.jsonl`/`episodes.jsonl`).
"""

import io
import json
from pathlib import Path

import pytest

from data.contracts.containers import Disposition
from data.contracts.profile import load_profile
from data.tests.frame_fixtures import build_synthetic_episode
from data.tests.mutators import build_varying_baseline_dataset
from data.validation.context import ValidationContext
from data.validation.image_histogram import check_image_histogram_integrity
from data.writers.lerobot_v21 import (
    AUTO_TASK_SENTINEL,
    LeRobotV21Writer,
    get_writer,
    resolve_task_string,
)

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
PROFILE = load_profile(PROFILE_PATH)

EMIT_TRUE_VIDEO_KEYS = {
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
}


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Test 1: all three emit:true cameras produce one MP4 each; the
# emit:false stereo-right camera produces none and has no info.json feature.


def test_all_emit_true_cameras_produce_one_mp4_each(tmp_path: Path) -> None:
    episode = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    result = writer.write_episode(episode)
    writer.finalize()

    for video_key in EMIT_TRUE_VIDEO_KEYS:
        assert video_key in result.output_paths
        assert Path(result.output_paths[video_key]).exists()

    info = _read_json(tmp_path / "meta" / "info.json")
    image_features = {k for k in info["features"] if k.startswith("observation.images.")}
    assert image_features == EMIT_TRUE_VIDEO_KEYS
    assert "observation.images.head_right" not in info["features"]


# --- Test 2: three episodes get episode_index 0,1,2 and a dataset-wide
# index that continues across episode boundaries with no gap or repeat.


def test_sequential_episode_and_dataset_index(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    frame_counts = [3, 2, 4]
    results = []
    for n in frame_counts:
        episode = build_synthetic_episode(frame_count=n, fps=25.0, profile=PROFILE, task_prompt="t")
        results.append(writer.write_episode(episode))
    writer.finalize()

    assert [r.episode_index for r in results] == [0, 1, 2]

    all_index_values: list[int] = []
    for i in range(3):
        import pyarrow.parquet as pq

        table = pq.read_table(tmp_path / f"data/chunk-000/episode_{i:06d}.parquet")
        all_index_values.extend(table.column("index").to_pylist())

    assert all_index_values == list(range(sum(frame_counts)))


# --- Test 3: meta/episodes.jsonl has one row per episode whose declared
# length equals that episode's actual parquet row count.


def test_episodes_jsonl_length_matches_parquet_row_count(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    for n in (3, 2):
        episode = build_synthetic_episode(frame_count=n, fps=25.0, profile=PROFILE, task_prompt="t")
        writer.write_episode(episode)
    writer.finalize()

    rows = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    assert len(rows) == 2
    for row, expected_length in zip(rows, (3, 2)):
        assert row["length"] == expected_length


# --- Test 4: tasks.jsonl rows carry an integer index and text; parquet's
# task_index column is int64; no parquet column contains the task text.


def test_tasks_jsonl_and_parquet_task_index(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    episode = build_synthetic_episode(
        frame_count=2, fps=25.0, profile=PROFILE, task_prompt="pick up the orange"
    )
    writer.write_episode(episode)
    writer.finalize()

    tasks = _read_jsonl(tmp_path / "meta" / "tasks.jsonl")
    assert tasks == [{"task_index": 0, "task": "pick up the orange"}]

    import pyarrow.parquet as pq

    table = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    assert table.schema.field("task_index").type == table.schema.field("task_index").type
    assert str(table.schema.field("task_index").type) == "int64"
    for col_name in table.schema.names:
        if col_name == "task_index":
            continue
        col = table.column(col_name)
        assert "pick up the orange" not in str(col.to_pylist())


# --- Test 5: episodes_stats.jsonl absent by default, present only when
# explicitly requested.


def test_episodes_stats_jsonl_absent_by_default(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
    writer.write_episode(episode)
    writer.finalize()
    assert not (tmp_path / "meta" / "episodes_stats.jsonl").exists()


def test_episodes_stats_jsonl_present_when_requested(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE, emit_episodes_stats=True)
    episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
    writer.write_episode(episode)
    writer.finalize()
    assert (tmp_path / "meta" / "episodes_stats.jsonl").exists()


# --- Test 6: finalize() after zero episodes writes a fully-populated
# features map and an empty tasks.jsonl.


def test_finalize_after_zero_episodes(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    writer.finalize()

    info = _read_json(tmp_path / "meta" / "info.json")
    assert set(info["features"]) == {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        *EMIT_TRUE_VIDEO_KEYS,
    }
    tasks_path = tmp_path / "meta" / "tasks.jsonl"
    assert tasks_path.exists()
    assert tasks_path.read_text(encoding="utf-8") == ""


# --- Test 7: an explicit task string is used verbatim.


def test_resolve_task_string_explicit_verbatim() -> None:
    assert resolve_task_string("do the thing", stdin_isatty=False) == "do the thing"
    assert resolve_task_string("do the thing", stdin_isatty=True, prompt=lambda: "unused") == (
        "do the thing"
    )


# --- Test 8: opt-in placeholder assigns one distinct task_index per
# episode, all sharing the same sentinel text.


def test_auto_placeholder_distinct_index_shared_sentinel(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    for _ in range(3):
        episode = build_synthetic_episode(
            frame_count=1, fps=25.0, profile=PROFILE, task_prompt="auto"
        )
        writer.write_episode(episode)
    writer.finalize()

    tasks = _read_jsonl(tmp_path / "meta" / "tasks.jsonl")
    assert [row["task_index"] for row in tasks] == [0, 1, 2]
    assert all(row["task"] == AUTO_TASK_SENTINEL for row in tasks)
    assert all("<<AUTO-UNLABELED>>" in row["task"] for row in tasks)


# --- Test 9 (module-level, on resolve_task_string directly): headless +
# no task string raises, naming the operator-typed flag.


def test_resolve_task_string_headless_no_task_raises() -> None:
    with pytest.raises(ValueError, match="auto"):
        resolve_task_string(None, stdin_isatty=False)


# --- Test 10: interactive stdin invokes the injected prompt exactly once.


def test_resolve_task_string_interactive_prompts_once() -> None:
    calls = []

    def _prompt() -> str:
        calls.append(1)
        return "prompted task"

    result = resolve_task_string(None, stdin_isatty=True, prompt=_prompt)
    assert result == "prompted task"
    assert len(calls) == 1


# --- Test 11: registry — v3 raises NotImplementedError; malformed keys
# raise ValueError naming the registered v2.1 key.


def test_registry_v3_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="v3"):
        get_writer("v3", tmp_path, PROFILE)


@pytest.mark.parametrize("bad_key", ["V2.1", " v2.1", "v2.1 ", "", "bogus-format"])
def test_registry_malformed_or_unknown_format_raises(tmp_path: Path, bad_key: str) -> None:
    with pytest.raises(ValueError, match="v2.1"):
        get_writer(bad_key, tmp_path, PROFILE)


def test_registry_v21_resolves_to_lerobot_v21_writer(tmp_path: Path) -> None:
    writer = get_writer("v2.1", tmp_path, PROFILE)
    assert isinstance(writer, LeRobotV21Writer)


# --- Test 12: default codec h264; AV1 opt-in succeeds and is documented
# unverified.


def test_default_codec_is_h264(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    assert writer._codec == "h264"


def test_av1_codec_opt_in_succeeds(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE, codec="av1")
    episode = build_synthetic_episode(
        frame_count=2, fps=25.0, profile=PROFILE, emit_cameras=["observation.images.head"],
        task_prompt="t",
    )
    result = writer.write_episode(episode)
    writer.finalize()
    assert "observation.images.head" in result.output_paths


def test_module_docstring_documents_av1_as_unverified_opt_in() -> None:
    import data.writers.lerobot_v21 as mod

    doc = mod.__doc__ or ""
    assert "AV1" in doc
    assert "unverified" in doc.lower()
    assert "sample_dataset" in doc


# --- Test 13: output_paths names every file produced; zero-frame camera
# reported absent (already covered by tracer tests; re-asserted here across
# all three cameras with one camera never appearing in any frame).


def test_output_paths_names_every_produced_file_multi_camera(tmp_path: Path) -> None:
    episode = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        emit_cameras=["observation.images.head", "observation.images.wrist_left"],
        task_prompt="t",
    )
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    result = writer.write_episode(episode)
    writer.finalize()

    assert "observation.images.head" in result.output_paths
    assert "observation.images.wrist_left" in result.output_paths
    assert "observation.images.wrist_right" not in result.output_paths
    assert "parquet" in result.output_paths


# ---------------------------------------------------------------------------
# Phase 4 Plan 2, Task 1: D-02's opt-in resume seam and decode/encode timing.
# ---------------------------------------------------------------------------


def test_fresh_two_episode_run_unchanged(tmp_path: Path) -> None:
    """Regression guard (RESEARCH.md Pitfall 2): a fresh, non-resumed
    two-episode run's finalize() output is unaffected by D-02's changes."""
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    for n in (2, 2):
        episode = build_synthetic_episode(frame_count=n, fps=25.0, profile=PROFILE, task_prompt="t")
        writer.write_episode(episode)
    writer.finalize()

    info = _read_json(tmp_path / "meta" / "info.json")
    assert info["total_episodes"] == 2
    assert len(_read_jsonl(tmp_path / "meta" / "episodes.jsonl")) == 2
    assert len(_read_jsonl(tmp_path / "meta" / "tasks.jsonl")) == 2


def test_resume_over_two_episode_prefix_writes_third_additively(tmp_path: Path) -> None:
    writer1 = LeRobotV21Writer(tmp_path, PROFILE)
    for n in (2, 2):
        episode = build_synthetic_episode(frame_count=n, fps=25.0, profile=PROFILE, task_prompt="t")
        writer1.write_episode(episode)
    writer1.finalize()

    writer2 = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    episode3 = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    result3 = writer2.write_episode(episode3)
    writer2.finalize()

    assert result3.episode_index == 2

    info = _read_json(tmp_path / "meta" / "info.json")
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 2 + 2 + 3

    episodes_rows = _read_jsonl(tmp_path / "meta" / "episodes.jsonl")
    tasks_rows = _read_jsonl(tmp_path / "meta" / "tasks.jsonl")
    assert len(episodes_rows) == 3
    assert len(tasks_rows) == 3


def test_resume_stats_json_merges_across_runs_and_quantiles_shift(tmp_path: Path) -> None:
    writer1 = LeRobotV21Writer(tmp_path, PROFILE)
    for _ in range(2):
        episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
        writer1.write_episode(episode)
    writer1.finalize()

    single_run_stats = _read_json(tmp_path / "meta" / "stats.json")
    single_run_q99 = single_run_stats["observation.state"]["q99"]

    writer2 = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    episode3 = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    writer2.write_episode(episode3)
    writer2.finalize()

    resumed_stats = _read_json(tmp_path / "meta" / "stats.json")
    resumed_q99 = resumed_stats["observation.state"]["q99"]

    # 2 episodes of 2 frames each (values 0,1,0,1) merged then a 3rd episode
    # of 3 frames (values 0,1,2) added -- q99 shifts from 1.0 to 2.0. This
    # would only hold if the resumed writer's rehydration actually retained
    # and re-merged the prior run's per-episode retained values.
    assert resumed_q99 != single_run_q99
    assert resumed_stats["observation.state"]["count"] == [2 + 2 + 3]


def test_resume_with_no_episodes_jsonl_is_noop_like_fresh_run(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
    result = writer.write_episode(episode)
    writer.finalize()

    assert result.episode_index == 0
    info = _read_json(tmp_path / "meta" / "info.json")
    assert info["total_episodes"] == 1


def test_resume_excludes_partial_marked_not_included_in_merge(tmp_path: Path) -> None:
    writer1 = LeRobotV21Writer(tmp_path, PROFILE)
    pass_episode = build_synthetic_episode(
        frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t", disposition=Disposition.PASS
    )
    writer1.write_episode(pass_episode)
    reject_episode = build_synthetic_episode(
        frame_count=5, fps=25.0, profile=PROFILE, task_prompt="t", disposition=Disposition.REJECT
    )
    writer1.write_episode(reject_episode)
    writer1.finalize()

    writer2 = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    # Only the PASS episode's 2 frames should be reflected in the rehydrated
    # merge -- the REJECT episode's partial is on disk but excluded.
    assert writer2._stats_merged["observation.state"].count == 2


def test_resume_malformed_episodes_jsonl_line_raises_naming_file_and_line(tmp_path: Path) -> None:
    writer1 = LeRobotV21Writer(tmp_path, PROFILE)
    episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
    writer1.write_episode(episode)
    writer1.finalize()

    episodes_path = tmp_path / "meta" / "episodes.jsonl"
    with open(episodes_path, "a", encoding="utf-8") as f:
        f.write("{not valid json\n")

    with pytest.raises(ValueError, match=r"line 2") as exc_info:
        LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    assert str(episodes_path) in str(exc_info.value)


def test_decode_and_encode_seconds_zero_before_write_and_positive_after(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    assert writer.decode_seconds == 0.0
    assert writer.encode_seconds == 0.0

    episode = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        emit_cameras=["observation.images.head"],
        task_prompt="t",
    )
    writer.write_episode(episode)
    writer.finalize()

    assert writer.decode_seconds > 0.0
    assert writer.encode_seconds > 0.0


def test_decode_seconds_strictly_increases_across_second_episode(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    episode1 = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        emit_cameras=["observation.images.head"],
        task_prompt="t",
    )
    writer.write_episode(episode1)
    after_first = writer.decode_seconds

    episode2 = build_synthetic_episode(
        frame_count=2,
        fps=25.0,
        profile=PROFILE,
        emit_cameras=["observation.images.head"],
        task_prompt="t",
    )
    writer.write_episode(episode2)
    writer.finalize()
    after_second = writer.decode_seconds

    assert after_second > after_first


def test_zero_frame_write_leaves_decode_encode_seconds_unchanged(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE)
    before_decode, before_encode = writer.decode_seconds, writer.encode_seconds

    zero_frame_episode = build_synthetic_episode(
        frame_count=0, fps=25.0, profile=PROFILE, task_prompt="t"
    )
    result = writer.write_episode(zero_frame_episode)

    assert writer.decode_seconds == before_decode
    assert writer.encode_seconds == before_encode
    assert result.frame_count == 0
    assert result.episode_index == 0


# ---------------------------------------------------------------------------
# Plan 04-10 (WINDOWS.md id 9 gap closure): a zero-new-episode resume must
# not clobber meta/info.json's previously-measured per-camera frame
# dimensions back down to _default_video_info's zero-valued placeholder.
# ---------------------------------------------------------------------------


def test_gap_resumed_finalize_preserves_recorded_video_dimensions(tmp_path: Path) -> None:
    writer1 = LeRobotV21Writer(tmp_path, PROFILE)
    for _ in range(2):
        episode = build_synthetic_episode(frame_count=2, fps=25.0, profile=PROFILE, task_prompt="t")
        writer1.write_episode(episode)
    writer1.finalize()

    first_info = _read_json(tmp_path / "meta" / "info.json")
    recorded_dimensions: dict[str, tuple[int, int]] = {}
    for video_key in EMIT_TRUE_VIDEO_KEYS:
        video_info = first_info["features"][video_key]["video_info"]
        height = int(video_info["video.height"])
        width = int(video_info["video.width"])
        assert height > 0
        assert width > 0
        recorded_dimensions[video_key] = (height, width)

    writer2 = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    # No write_episode call at all — this is the zero-new-episode resume
    # that is the whole point of this regression.
    writer2.finalize()

    second_info = _read_json(tmp_path / "meta" / "info.json")
    for video_key in EMIT_TRUE_VIDEO_KEYS:
        height, width = recorded_dimensions[video_key]
        video_info = second_info["features"][video_key]["video_info"]
        assert int(video_info["video.height"]) == height
        assert int(video_info["video.width"]) == width
        assert second_info["features"][video_key]["shape"] == [height, width, 3]


def test_gap_resume_with_no_info_json_keeps_placeholder_dimensions(tmp_path: Path) -> None:
    # No prior run at all -- meta/info.json does not exist under tmp_path.
    writer = LeRobotV21Writer(tmp_path, PROFILE, resume=True)
    writer.finalize()

    info = _read_json(tmp_path / "meta" / "info.json")
    for video_key in EMIT_TRUE_VIDEO_KEYS:
        video_info = info["features"][video_key]["video_info"]
        assert int(video_info["video.height"]) == 0
        assert int(video_info["video.width"]) == 0
        assert info["features"][video_key]["shape"] == [0, 0, 3]


def test_gap_second_validation_pass_over_resumed_prefix_reports_no_histogram_finding(
    tmp_path: Path,
) -> None:
    """Names the exact WINDOWS id 9 evidence shape: a second validated pass
    over a resumed (zero-new-episode) prefix must not report a
    `mass_identity` finding with `expected_mass == 0`. Deliberately calls
    `check_image_histogram_integrity` directly, never `validate_dataset` --
    the shipped profile's other standing findings
    (`kinematics.velocity_acceleration_bounds` abstaining for lack of
    declared physics limits, `consistency.action_state_lag` at this row
    count) would otherwise dominate the outcome and this test would assert
    about the wrong check."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", PROFILE, camera_frames=6, episodes=2)

    ctx1 = ValidationContext(out_dir=dataset_dir, profile=PROFILE, baseline=None, use_local_golden=False)
    result1 = check_image_histogram_integrity(ctx1)[0]
    assert result1.passed is True

    writer2 = LeRobotV21Writer(dataset_dir, PROFILE, resume=True, stdin_isatty=lambda: False)
    # No write_episode call -- this is the zero-new-episode resume.
    writer2.finalize()

    ctx2 = ValidationContext(out_dir=dataset_dir, profile=PROFILE, baseline=None, use_local_golden=False)
    result2 = check_image_histogram_integrity(ctx2)[0]
    assert result2.passed is True

    findings = result2.evidence.get("findings", [])
    assert findings == []
    assert not any(f.get("invariant") == "mass_identity" for f in findings)
    assert not any(f.get("expected_mass") == 0 for f in findings)
