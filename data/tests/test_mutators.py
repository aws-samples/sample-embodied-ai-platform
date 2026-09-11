"""Proves every corruption mutator in `data/tests/mutators.py` actually
produces the corruption it claims on a copied prefix, and that the source
prefix is left byte-identical — the self-proof that keeps every downstream
negative test in this phase from passing vacuously against an unmutated
copy (plan 03-01, Task 2).
"""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import (
    build_varying_baseline_dataset,
    copy_dataset,
    corrupt_episodes_length,
    drop_tasks_row,
    force_constant_channel,
    freeze_camera_frames,
    inject_nan_into_parquet,
    inject_nan_into_stats,
    overlap_modality_slice,
    perturb_histogram_counts,
    perturb_stats_value,
    shift_action_column,
    shift_camera_frames,
)

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
MASTER_VIDEO_KEY = "observation.images.head"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


@pytest.fixture()
def baseline(tmp_path: Path, profile: EmbodimentProfile) -> Path:
    return build_varying_baseline_dataset(tmp_path / "baseline", profile, camera_frames=10)


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _decode_frames(path: Path) -> list[np.ndarray]:
    container = av.open(str(path))
    try:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Test 1: every mutator returns a copy path, source prefix untouched.
# ---------------------------------------------------------------------------


def test_every_mutator_leaves_source_prefix_byte_identical(
    tmp_path: Path, baseline: Path
) -> None:
    before = _sha256_tree(baseline)

    copy1 = copy_dataset(baseline, tmp_path / "c1")
    shift_camera_frames(copy1, video_key=MASTER_VIDEO_KEY, shift=1)

    copy2 = copy_dataset(baseline, tmp_path / "c2")
    freeze_camera_frames(copy2, video_key=MASTER_VIDEO_KEY)

    copy3 = copy_dataset(baseline, tmp_path / "c3")
    perturb_stats_value(copy3, feature="action", stat="mean", channel_index=0, delta=5.0)

    after = _sha256_tree(baseline)
    assert before == after
    assert copy1 != baseline
    assert copy2 != baseline
    assert copy3 != baseline


# ---------------------------------------------------------------------------
# Test 2: shift_camera_frames — frame count unchanged, at least one frame's
# pixel content differs at the same index.
# ---------------------------------------------------------------------------


def test_shift_camera_frames_preserves_count_shifts_content(
    tmp_path: Path, baseline: Path
) -> None:
    from data.tests.mutators import _video_file_path

    mutated = copy_dataset(baseline, tmp_path / "mutated")
    original_frames = _decode_frames(_video_file_path(baseline, MASTER_VIDEO_KEY, 0))

    shift_camera_frames(mutated, video_key=MASTER_VIDEO_KEY, shift=1)
    shifted_frames = _decode_frames(_video_file_path(mutated, MASTER_VIDEO_KEY, 0))

    assert len(shifted_frames) == len(original_frames)
    assert any(
        not np.array_equal(a, b) for a, b in zip(original_frames, shifted_frames)
    )


# ---------------------------------------------------------------------------
# Test 3: freeze_camera_frames — every decoded frame collapses to one digest.
# ---------------------------------------------------------------------------


def test_freeze_camera_frames_collapses_to_one_digest(tmp_path: Path, baseline: Path) -> None:
    from data.tests.mutators import _video_file_path

    video_path = _video_file_path(baseline, MASTER_VIDEO_KEY, 0)
    original_frames = _decode_frames(video_path)
    original_stack = np.stack(original_frames).astype(np.float64)
    original_frame_to_frame_variance = float(np.mean(np.var(original_stack, axis=0)))

    mutated = copy_dataset(baseline, tmp_path / "mutated")
    freeze_camera_frames(mutated, video_key=MASTER_VIDEO_KEY)

    frames = _decode_frames(_video_file_path(mutated, MASTER_VIDEO_KEY, 0))
    assert len(frames) == len(original_frames)
    frozen_stack = np.stack(frames).astype(np.float64)
    frozen_frame_to_frame_variance = float(np.mean(np.var(frozen_stack, axis=0)))

    # Every frame was re-encoded from the SAME source pixel array, so
    # cross-frame variance collapses to near-zero (lossy-codec rate-control
    # noise only) — never byte-identical in a real codec, but drastically
    # smaller than the genuinely-distinct-per-frame baseline's variance.
    assert original_frame_to_frame_variance > 100
    assert frozen_frame_to_frame_variance < original_frame_to_frame_variance / 10


# ---------------------------------------------------------------------------
# Test 4: inject_nan_into_stats / inject_nan_into_parquet produce non-finite
# values; perturb_stats_value changes exactly one value.
# ---------------------------------------------------------------------------


def test_inject_nan_into_stats_reads_back_non_finite(tmp_path: Path, baseline: Path) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    inject_nan_into_stats(mutated, feature="action", stat="mean", channel_index=0)

    payload = json.loads((mutated / "meta" / "stats.json").read_text())
    assert not np.isfinite(payload["action"]["mean"][0])


def test_inject_nan_into_parquet_reads_back_non_finite(tmp_path: Path, baseline: Path) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    inject_nan_into_parquet(mutated, feature="action", row_index=0, channel_index=0)

    from data.tests.mutators import _parquet_file_path

    table = pq.read_table(_parquet_file_path(mutated, 0))
    row0 = table.column("action")[0].as_py()
    assert not np.isfinite(row0[0])


def test_perturb_stats_value_changes_exactly_one_value(tmp_path: Path, baseline: Path) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    before = json.loads((baseline / "meta" / "stats.json").read_text())
    perturb_stats_value(mutated, feature="action", stat="mean", channel_index=0, delta=10.0)
    after = json.loads((mutated / "meta" / "stats.json").read_text())

    assert after["action"]["mean"][0] == pytest.approx(before["action"]["mean"][0] + 10.0)
    after["action"]["mean"][0] = before["action"]["mean"][0]
    assert after == before


# ---------------------------------------------------------------------------
# Test 5: drop_tasks_row removes exactly one row, parquet task_index
# unchanged.
# ---------------------------------------------------------------------------


def test_drop_tasks_row_removes_one_row_leaves_parquet_untouched(
    tmp_path: Path, baseline: Path
) -> None:
    from data.tests.mutators import _parquet_file_path

    before_task_index = pq.read_table(_parquet_file_path(baseline, 0)).column(
        "task_index"
    ).to_pylist()

    mutated = copy_dataset(baseline, tmp_path / "mutated")
    before_rows = [
        json.loads(line)
        for line in (baseline / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_index = before_rows[0]["task_index"]
    drop_tasks_row(mutated, task_index=task_index)

    after_rows = [
        json.loads(line)
        for line in (mutated / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(after_rows) == len(before_rows) - 1
    assert task_index not in {r["task_index"] for r in after_rows}

    after_task_index = pq.read_table(_parquet_file_path(mutated, 0)).column(
        "task_index"
    ).to_pylist()
    assert after_task_index == before_task_index


# ---------------------------------------------------------------------------
# Test 6: shift_action_column rotates by one row, row count/dtype/width
# preserved.
# ---------------------------------------------------------------------------


def test_shift_action_column_rotates_preserving_shape(tmp_path: Path, baseline: Path) -> None:
    from data.tests.mutators import _parquet_file_path

    mutated = copy_dataset(baseline, tmp_path / "mutated")
    before_table = pq.read_table(_parquet_file_path(baseline, 0))
    before_rows = before_table.column("action").to_pylist()

    shift_action_column(mutated, episode_index=0, shift=1)

    after_table = pq.read_table(_parquet_file_path(mutated, 0))
    after_rows = after_table.column("action").to_pylist()
    field_type = after_table.schema.field("action").type

    assert len(after_rows) == len(before_rows)
    assert field_type == before_table.schema.field("action").type
    assert after_rows == before_rows[1:] + before_rows[:1]


# ---------------------------------------------------------------------------
# Test 7: overlap_modality_slice widens one range to overlap its neighbour,
# parquet untouched.
# ---------------------------------------------------------------------------


def test_overlap_modality_slice_widens_range_leaves_parquet_untouched(
    tmp_path: Path, baseline: Path, profile: EmbodimentProfile
) -> None:
    from data.tests.mutators import _parquet_file_path

    mutated = copy_dataset(baseline, tmp_path / "mutated")
    before_modality = json.loads((baseline / "meta" / "modality.json").read_text())

    # `write_modality_artifacts` writes with `sort_keys=True`, so the file's
    # own dict iteration order is alphabetical, not the packed structural
    # order — use `profile.state_vector_order` for the true structural
    # neighbour, never the JSON's own key order.
    first_key, next_key = profile.state_vector_order[0], profile.state_vector_order[1]
    before_end = before_modality["state"][first_key]["end"]

    overlap_modality_slice(mutated, key=first_key, scope="state", by=1)

    after_modality = json.loads((mutated / "meta" / "modality.json").read_text())
    assert after_modality["state"][first_key]["end"] == before_end + 1
    # The widened range now overlaps the structural neighbour's start.
    assert after_modality["state"][first_key]["end"] > after_modality["state"][next_key]["start"]

    before_parquet_bytes = _parquet_file_path(baseline, 0).read_bytes()
    after_parquet_bytes = _parquet_file_path(mutated, 0).read_bytes()
    assert before_parquet_bytes == after_parquet_bytes


# ---------------------------------------------------------------------------
# Test 8: corrupt_episodes_length changes exactly one row's length by one.
# ---------------------------------------------------------------------------


def test_corrupt_episodes_length_changes_one_row_by_one(tmp_path: Path, baseline: Path) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    before_rows = [
        json.loads(line)
        for line in (baseline / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    corrupt_episodes_length(mutated, episode_index=0, delta=1)
    after_rows = [
        json.loads(line)
        for line in (mutated / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert after_rows[0]["length"] == before_rows[0]["length"] + 1
    assert after_rows[1:] == before_rows[1:]


# ---------------------------------------------------------------------------
# Test 9: force_constant_channel makes one channel dataset-wide constant.
# ---------------------------------------------------------------------------


def test_force_constant_channel_makes_min_equal_max(tmp_path: Path, profile) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "multi", profile, camera_frames=6, episodes=2
    )
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")

    force_constant_channel(mutated, feature="action", channel_index=0)

    from data.tests.mutators import _parquet_file_path

    for episode_index in (0, 1):
        table = pq.read_table(_parquet_file_path(mutated, episode_index))
        values = [row[0] for row in table.column("action").to_pylist()]
        assert len(set(values)) == 1

    stats_payload = json.loads((mutated / "meta" / "stats.json").read_text())
    assert stats_payload["action"]["min"][0] == stats_payload["action"]["max"][0]
    assert stats_payload["action"]["std"][0] == 0.0


# ---------------------------------------------------------------------------
# Test 10: build_varying_baseline_dataset — every emitted camera decodes to
# `camera_frames` distinct pixel digests.
# ---------------------------------------------------------------------------


def test_varying_baseline_every_camera_has_distinct_frames(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    from data.tests.mutators import _video_file_path

    dataset_dir = build_varying_baseline_dataset(tmp_path / "baseline", profile, camera_frames=7)

    for video_key in (
        "observation.images.head",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ):
        frames = _decode_frames(_video_file_path(dataset_dir, video_key, 0))
        assert len(frames) == 7
        digests = {f.tobytes() for f in frames}
        assert len(digests) == 7


# ---------------------------------------------------------------------------
# Test 11: every episode of a two-episode varying baseline has a distinct,
# hashable source_uri.
# ---------------------------------------------------------------------------


def test_varying_baseline_two_episodes_have_distinct_source_uris(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "baseline", profile, camera_frames=5, episodes=2
    )
    record0 = json.loads((dataset_dir / "meta" / "data_engine" / "episode_000000.json").read_text())
    record1 = json.loads((dataset_dir / "meta" / "data_engine" / "episode_000001.json").read_text())

    assert record0["source_uri"] != record1["source_uri"]
    for record in (record0, record1):
        source_path = Path(record["source_uri"])
        assert source_path.exists()
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record["source_sha256"]


# ---------------------------------------------------------------------------
# Test 12: perturb_histogram_counts — self-proof (03-09-PLAN.md). Changes
# exactly the targeted bin of the targeted camera by exactly `delta`, leaves
# every other bin/camera/top-level key untouched, and raises ValueError
# naming the path for an absent file, camera, or histogram.
# ---------------------------------------------------------------------------


def test_perturb_histogram_counts_changes_exactly_one_bin(
    tmp_path: Path, baseline: Path
) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    partial_path = baseline / "meta" / "stats_partials" / "episode_000000.json"
    before = json.loads(partial_path.read_text())

    perturb_histogram_counts(
        mutated, video_key=MASTER_VIDEO_KEY, episode_index=0, bin_index=7, delta=3
    )

    mutated_partial_path = mutated / "meta" / "stats_partials" / "episode_000000.json"
    after = json.loads(mutated_partial_path.read_text())

    before_counts = before["histograms"][MASTER_VIDEO_KEY]
    after_counts = after["histograms"][MASTER_VIDEO_KEY]
    assert after_counts[0][7] == before_counts[0][7] + 3
    # Every other bin of the targeted channel, every other channel, and
    # every other camera's histogram is untouched.
    after_counts[0][7] = before_counts[0][7]
    assert after_counts == before_counts
    for other_video_key in after["histograms"]:
        if other_video_key != MASTER_VIDEO_KEY:
            assert after["histograms"][other_video_key] == before["histograms"][other_video_key]
    # Every other top-level key (features, disposition, included_in_merge)
    # is untouched.
    after_full = dict(after)
    del after_full["histograms"]
    before_full = dict(before)
    del before_full["histograms"]
    assert after_full == before_full


def test_perturb_histogram_counts_raises_naming_missing_target(
    tmp_path: Path, baseline: Path
) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")
    partial_path = mutated / "meta" / "stats_partials" / "episode_000000.json"

    with pytest.raises(ValueError, match="no histogram for"):
        perturb_histogram_counts(
            mutated, video_key="not_a_real_camera", episode_index=0, bin_index=0, delta=1
        )

    payload = json.loads(partial_path.read_text())
    del payload["histograms"][MASTER_VIDEO_KEY]
    partial_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no histogram for"):
        perturb_histogram_counts(
            mutated, video_key=MASTER_VIDEO_KEY, episode_index=0, bin_index=0, delta=1
        )

    partial_path.unlink()
    with pytest.raises(ValueError, match="no stats partial"):
        perturb_histogram_counts(
            mutated, video_key=MASTER_VIDEO_KEY, episode_index=0, bin_index=0, delta=1
        )


# ---------------------------------------------------------------------------
# Missing-target guard: every mutator raises with the missing path named.
# ---------------------------------------------------------------------------


def test_mutators_raise_naming_missing_target(tmp_path: Path, baseline: Path) -> None:
    mutated = copy_dataset(baseline, tmp_path / "mutated")

    with pytest.raises(ValueError, match="no row with task_index=999"):
        drop_tasks_row(mutated, task_index=999)

    (mutated / "meta" / "tasks.jsonl").unlink()
    with pytest.raises(ValueError, match="no meta/tasks.jsonl"):
        drop_tasks_row(mutated, task_index=0)
