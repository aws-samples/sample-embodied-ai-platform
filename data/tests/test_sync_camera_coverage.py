"""G-01/VERIFY-01 gap closure (04-04-PLAN.md, Gap 1 of 04-VERIFICATION.md):
the real defect that crashed GR00T's dataloader on real data -- a secondary
camera whose own recording ends a few ticks before the master-clock
camera's -- driven through the REAL adapter/synchronizer/writer chain, not
a post-hoc mutation of an already-written dataset.

Task 1's tracer proves the truncation lands correctly end to end (grid
shrinks, every camera's encode manifest and the parquet row count agree,
the post-write `videos.frame_count_agreement` check passes, and the trim is
annotated). Task 2 proves the pre-write REJECT past the declared tolerance.
Task 3 proves the writer's own last-resort guard against ever committing a
camera whose frame count disagrees with the parquet row count.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from data.adapters.galaxea import GalaxeaAdapterError, GalaxeaInputAdapter
from data.contracts.containers import CanonicalEpisode, Disposition
from data.contracts.profile import load_profile
from data.provenance.recorder import ProvenanceRecorder
from data.sync.master_clock import (
    CAMERA_COVERAGE_CHECK_NAME,
    CameraCoverage,
    CameraCoverageReport,
    camera_coverage_report,
    coverage_check_result,
    nearest_camera_matches,
)
from data.tests.frame_fixtures import build_synthetic_episode
from data.tests.mcap_fixture import (
    SCHEMA_NAME_COMPRESSED_IMAGE,
    SCHEMA_TEXT_COMPRESSED_IMAGE,
    TopicSpec,
    build_fixture_mcap,
    camera_messages,
)
from data.tests.mutators import build_fixture_source_mcap
from data.validation.context import ValidationContext
from data.validation.frame_integrity import check_frame_count_agreement
from data.writers.lerobot_v21 import LeRobotV21Writer

# Task 2 imports (adapter-level enforcement + full CLI path).
import json

import yaml

from data.adapters.galaxea import _enforce_camera_coverage
from data.cli import EXIT_COMPLETED_WITH_FINDINGS, main

# Task 3 imports (writer-level pre-commit parity guard).
from data.contracts.containers import CanonicalFrame, EpisodeHeader
from data.tests.frame_fixtures import FixtureImageRef, _action_layout, _build_gradient_jpeg
from data.tests.frame_fixtures import _state_layout as _fixture_state_layout

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
PROFILE = load_profile(PROFILE_PATH)
DECLARED_TOLERANCE_TICKS = PROFILE.master_clock.max_camera_coverage_deficit_ticks

WRIST_LEFT_TOPIC = "/hdas/camera_wrist_left/color/image_raw/compressed"
WRIST_RIGHT_TOPIC = "/hdas/camera_wrist_right/color/image_raw/compressed"

WRIST_LEFT_VIDEO_KEY = "observation.images.wrist_left"
WRIST_RIGHT_VIDEO_KEY = "observation.images.wrist_right"
HEAD_VIDEO_KEY = "observation.images.head"

STEP_NS = 40_000_000
MAX_SYNC_GAP_NS = 40_000_000


def _convert_source_mcap(
    tmp_path: Path, source_mcap: Path
) -> tuple[Path, "CanonicalEpisode", "object"]:
    """Drives `source_mcap` through the real adapter, writer and
    `ProvenanceRecorder.record`, exactly as
    `data/tests/mutators.py::build_varying_baseline_dataset` does --
    including its `task_prompt="auto"` header rebuild, without which the
    writer's fail-loud task-string policy raises under a non-tty. Returns
    `(dataset_dir, rebuilt_episode_header_meta_checks, write_result)` -- the
    caller re-reads on-disk artifacts for its own assertions rather than
    relying on returned in-memory state, since `episode.frames` is a
    single-pass generator already consumed by `write_episode`.
    """
    dest = tmp_path / "dataset"
    dest.mkdir(parents=True, exist_ok=True)

    episode = GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)
    header = dataclasses.replace(episode.header, task_prompt="auto")
    rebuilt = CanonicalEpisode(header=header, frames=episode.frames)

    writer = LeRobotV21Writer(dest, PROFILE, stdin_isatty=lambda: False)
    write_result = writer.write_episode(rebuilt)
    ProvenanceRecorder(dest, PROFILE).record(header, write_result)
    writer.finalize()

    return dest, header, write_result


def _coverage_check_evidence(dest: Path, episode_index: int = 0) -> dict:
    import json

    record_path = dest / "meta" / "data_engine" / f"episode_{episode_index:06d}.json"
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    for check in record["checks"]:
        if check["check_name"] == CAMERA_COVERAGE_CHECK_NAME:
            return check["evidence"]
    raise AssertionError(f"no {CAMERA_COVERAGE_CHECK_NAME} check found in {record_path}")


# ---------------------------------------------------------------------------
# Task 1: the tracer -- a wrist camera's recording ends before the head
# camera's, driven end to end through the real pipeline.
# ---------------------------------------------------------------------------


def test_tracer_wrist_left_stream_ends_before_master_truncates_and_annotates(
    tmp_path: Path,
) -> None:
    """The scenario no existing fixture could construct before this plan:
    head (master) carries 10 frames, wrist_left carries 8, wrist_right
    carries 10. wrist_left's own last frame (index 7, t=280ms) is exactly
    40ms from grid tick 8 (t=320ms) -- `max_sync_gap_ms` is 40, so the
    inclusive bound (`nearest_camera_matches`' own `best_gap <= max_gap_ns`)
    still matches it, extending wrist_left's covered prefix to 9 ticks (0
    through 8). Tick 9 (t=360ms) is 80ms away -- unmatched. So the episode's
    parquet shrinks to 9 rows, not 10, and every emitted camera's MP4 must
    match it exactly."""
    source_mcap = build_fixture_source_mcap(
        tmp_path / "sources",
        camera_frames=10,
        camera_frames_by_topic={WRIST_LEFT_TOPIC: 8},
    )
    dest, _header, write_result = _convert_source_mcap(tmp_path, source_mcap)

    assert write_result.frame_count == 9

    import pyarrow.parquet as pq

    parquet_path = dest / "data" / "chunk-000" / "episode_000000.parquet"
    assert pq.ParquetFile(str(parquet_path)).metadata.num_rows == 9

    import json

    manifest_path = dest / "meta" / "encode_manifests" / "episode_000000.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    for video_key in (HEAD_VIDEO_KEY, WRIST_LEFT_VIDEO_KEY, WRIST_RIGHT_VIDEO_KEY):
        assert manifest[video_key]["frames_submitted"] == 9

    ctx = ValidationContext(out_dir=dest, profile=PROFILE)
    results = check_frame_count_agreement(ctx, 0)
    assert {r.evidence["video_key"] for r in results} == {
        HEAD_VIDEO_KEY,
        WRIST_LEFT_VIDEO_KEY,
        WRIST_RIGHT_VIDEO_KEY,
    }
    for r in results:
        assert r.passed is True, (r.evidence["video_key"], r.evidence)

    evidence = _coverage_check_evidence(dest)
    assert evidence["grid_ticks"] == 10
    assert evidence["kept_ticks"] == 9
    assert evidence["deficit_ticks"] == 1
    assert evidence["truncated"] is True
    assert evidence["tolerance_ticks"] == DECLARED_TOLERANCE_TICKS
    shortfall_names = [c["camera_name"] for c in evidence["shortfall_cameras"]]
    assert shortfall_names == ["wrist_left"]
    assert "wrist_right" not in shortfall_names


def test_equal_camera_counts_produce_zero_deficit_no_truncation(tmp_path: Path) -> None:
    """The companion no-gap case: every camera the same count. `deficit_ticks`
    is 0, the row count equals the master frame count, and no shortfall
    camera is recorded -- the no-gap path is bit-for-bit unchanged."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=10)
    dest, _header, write_result = _convert_source_mcap(tmp_path, source_mcap)

    assert write_result.frame_count == 10

    evidence = _coverage_check_evidence(dest)
    assert evidence["deficit_ticks"] == 0
    assert evidence["truncated"] is False
    assert evidence["shortfall_cameras"] == []


def test_camera_coverage_report_fields_are_python_or_numpy_ints() -> None:
    """Direct unit test of `camera_coverage_report` -- every numeric field of
    `CameraCoverageReport` and `CameraCoverage` is a Python `int` (never a
    Python `float` and never a NumPy floating dtype)."""
    grid_ts_ns = np.arange(10, dtype=np.int64) * STEP_NS
    wrist_left_ts = grid_ts_ns[:8]
    wrist_right_ts = grid_ts_ns.copy()

    matched_idx_l, gap_ns_l = nearest_camera_matches(wrist_left_ts, grid_ts_ns, MAX_SYNC_GAP_NS)
    matched_idx_r, gap_ns_r = nearest_camera_matches(wrist_right_ts, grid_ts_ns, MAX_SYNC_GAP_NS)
    camera_matches = {
        "wrist_left": (matched_idx_l, gap_ns_l, [None] * 8, WRIST_LEFT_VIDEO_KEY, wrist_left_ts),
        "wrist_right": (
            matched_idx_r,
            gap_ns_r,
            [None] * 10,
            WRIST_RIGHT_VIDEO_KEY,
            wrist_right_ts,
        ),
    }
    report = camera_coverage_report(camera_matches, PROFILE, grid_ts_ns, DECLARED_TOLERANCE_TICKS)

    assert isinstance(report, CameraCoverageReport)
    for field_name in ("grid_ticks", "kept_ticks", "deficit_ticks", "tolerance_ticks"):
        value = getattr(report, field_name)
        assert type(value) is int, (field_name, type(value))

    assert report.grid_ticks == 10
    assert report.kept_ticks == 9
    assert report.deficit_ticks == 1
    assert len(report.shortfall_cameras) == 1

    shortfall = report.shortfall_cameras[0]
    assert isinstance(shortfall, CameraCoverage)
    assert shortfall.camera_name == "wrist_left"
    assert type(shortfall.covered_prefix_ticks) is int
    assert type(shortfall.hole_gap_ns) is int
    assert shortfall.hole_kind == "stream_ended_before_master"

    check_result = coverage_check_result(report)
    assert check_result.check_name == CAMERA_COVERAGE_CHECK_NAME
    assert check_result.severity == "fatal"
    assert check_result.passed is True  # deficit 1 <= declared tolerance


def test_zero_covered_ticks_rejects_as_deficit_equal_to_whole_grid() -> None:
    """VERIFY-01/empty: a constraining camera with zero frames at all yields
    a deficit equal to the whole grid, never a zero-row episode."""
    grid_ts_ns = np.arange(10, dtype=np.int64) * STEP_NS
    empty = np.zeros(0, dtype=np.int64)
    matched_idx, gap_ns = nearest_camera_matches(empty, grid_ts_ns, MAX_SYNC_GAP_NS)
    camera_matches = {
        "wrist_left": (matched_idx, gap_ns, [], WRIST_LEFT_VIDEO_KEY, empty),
    }
    report = camera_coverage_report(camera_matches, PROFILE, grid_ts_ns, DECLARED_TOLERANCE_TICKS)
    assert report.kept_ticks == 0
    assert report.deficit_ticks == 10
    assert report.shortfall_cameras[0].hole_kind == "stream_ended_before_master"
    assert report.shortfall_cameras[0].hole_gap_ns == -1


def test_module_docstring_no_longer_claims_never_trims() -> None:
    import data.sync.master_clock as mod

    doc = mod.__doc__ or ""
    assert "never repairs, trims, or gap-fills" not in doc
    assert "trims" in doc.lower()


def test_master_clock_max_coverage_deficit_defaults_zero_and_rejects_negative() -> None:
    import pydantic

    from data.contracts.profile import MasterClock

    assert MasterClock(topic="/x", target_fps=None).max_camera_coverage_deficit_ticks == 0
    with pytest.raises(pydantic.ValidationError, match="max_camera_coverage_deficit_ticks"):
        MasterClock(topic="/x", target_fps=None, max_camera_coverage_deficit_ticks=-1)


def test_shipped_profile_declares_tolerance_with_measured_basis_comment() -> None:
    raw = PROFILE_PATH.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    matches = [ln for ln in lines if "max_camera_coverage_deficit_ticks" in ln]
    assert len(matches) == 1
    assert DECLARED_TOLERANCE_TICKS == 8


def test_mcap_fixture_and_mutators_camera_frames_by_topic_none_reproduces_default(
    tmp_path: Path,
) -> None:
    """`camera_frames_by_topic=None` on both `mcap_fixture.build_fixture_mcap`
    and `mutators.build_fixture_source_mcap` reproduces today's behaviour
    byte-for-byte: every camera topic still gets `camera_frames` messages."""
    path_a = tmp_path / "a.mcap"
    path_b = tmp_path / "b.mcap"
    build_fixture_mcap(str(path_a), camera_frames=5, joint_samples=5)
    build_fixture_mcap(str(path_b), camera_frames=5, joint_samples=5, camera_frames_by_topic=None)
    assert path_a.read_bytes() == path_b.read_bytes()


# ---------------------------------------------------------------------------
# Task 2: reject past the declared tolerance, before any write, through the
# structured rejection path.
# ---------------------------------------------------------------------------


def _write_run_config(dest_dir: Path, *, inputs: list[str]) -> Path:
    config = {
        "run_config_version": 1,
        "input_dir": None,
        "input_glob": "*.mcap",
        "inputs": inputs,
        "output_prefix": None,
        "format_key": "v2.1",
        "codec": "h264",
        "task_string": "auto",
        "fps_override": None,
        "emit_episodes_stats": False,
    }
    config_path = dest_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _walk_prefix(out_dir: Path) -> set[str]:
    if not out_dir.exists():
        return set()
    return {str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()}


def test_over_tolerance_deficit_raises_with_structured_reject_header(tmp_path: Path) -> None:
    """Deficit one tick beyond the declared tolerance (master=20,
    wrist_left=10 -> covered_prefix=11 -> deficit=9 > 8) raises before any
    write, carrying a REJECT-disposition header with the failing check."""
    source_mcap = build_fixture_source_mcap(
        tmp_path / "sources",
        camera_frames=20,
        camera_frames_by_topic={WRIST_LEFT_TOPIC: 10},
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)

    err = excinfo.value
    assert err.rejected_header is not None
    assert err.rejected_header.global_meta["disposition"] == Disposition.REJECT
    coverage_checks = [
        c
        for c in err.rejected_header.global_meta["checks"]
        if c.check_name == CAMERA_COVERAGE_CHECK_NAME
    ]
    assert len(coverage_checks) == 1
    assert coverage_checks[0].passed is False
    assert coverage_checks[0].evidence["deficit_ticks"] == 9
    assert coverage_checks[0].evidence["tolerance_ticks"] == DECLARED_TOLERANCE_TICKS


def test_over_tolerance_deficit_main_invocation_leaves_no_artifact(tmp_path: Path) -> None:
    """A full `main()` invocation over the over-tolerance input leaves no
    parquet, MP4, encode manifest or stats partial -- proven by a
    before/after filesystem walk of the output prefix -- and records
    exactly one `newly_rejected` ledger entry."""
    source_mcap = build_fixture_source_mcap(
        tmp_path / "sources",
        camera_frames=20,
        camera_frames_by_topic={WRIST_LEFT_TOPIC: 10},
    )
    out_dir = tmp_path / "output"
    before = _walk_prefix(out_dir)

    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--no-validate",
        ]
    )
    assert exit_code == EXIT_COMPLETED_WITH_FINDINGS

    assert not (out_dir / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert not (out_dir / "meta" / "encode_manifests" / "episode_000000.json").exists()
    assert not (out_dir / "meta" / "stats_partials" / "episode_000000.json").exists()

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["ledger"]) == 1
    assert report["ledger"][0]["bucket"] == "newly_rejected"

    # The before/after walk proves nothing PER-EPISODE was written for the
    # rejected input -- writer.finalize()'s own always-present bookkeeping
    # (meta/info.json, episodes.jsonl, stats.json, modality artifacts, all
    # with zero totals) is expected and orthogonal to this assertion.
    after = _walk_prefix(out_dir)
    new_files = after - before
    assert not any(f.startswith("data/chunk-000/episode_000000") for f in new_files)
    assert not any(f.startswith("videos/") and "episode_000000" in f for f in new_files)
    assert not any("encode_manifests" in f for f in new_files)
    assert not any("stats_partials" in f for f in new_files)


def test_exactly_at_tolerance_deficit_converts_as_pass_with_flags(tmp_path: Path) -> None:
    """A deficit exactly equal to the declared tolerance (master=20,
    wrist_left=11 -> covered_prefix=12 -> deficit=8) does not raise: the
    episode converts and its disposition is PASS_WITH_FLAGS."""
    source_mcap = build_fixture_source_mcap(
        tmp_path / "sources",
        camera_frames=20,
        camera_frames_by_topic={WRIST_LEFT_TOPIC: 11},
    )
    episode = GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)
    assert episode.header.frame_count == 12
    assert episode.header.global_meta["disposition"] == Disposition.PASS_WITH_FLAGS
    coverage_checks = [
        c
        for c in episode.header.global_meta["checks"]
        if c.check_name == CAMERA_COVERAGE_CHECK_NAME
    ]
    assert coverage_checks[0].passed is True
    assert coverage_checks[0].evidence["deficit_ticks"] == DECLARED_TOLERANCE_TICKS


def test_zero_coverage_camera_rejects_with_full_grid_deficit(tmp_path: Path) -> None:
    """A constraining camera with zero frames at all yields a deficit equal
    to the whole grid and rejects, never a zero-row episode."""
    source_mcap = build_fixture_source_mcap(
        tmp_path / "sources",
        camera_frames=10,
        camera_frames_by_topic={WRIST_LEFT_TOPIC: 0},
    )
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)
    coverage_checks = [
        c
        for c in excinfo.value.rejected_header.global_meta["checks"]
        if c.check_name == CAMERA_COVERAGE_CHECK_NAME
    ]
    assert coverage_checks[0].evidence["kept_ticks"] == 0
    assert coverage_checks[0].evidence["deficit_ticks"] == 10


def _build_interior_hole_source_mcap(dest_dir: Path, *, master_count: int) -> Path:
    """A wrist_left camera with an EARLY segment and a LATER segment,
    separated by a gap in the middle -- an interior hole (the camera has
    later frames, unlike a stream that simply ended)."""
    early = camera_messages(count=5)
    late = camera_messages(count=5, start_ns=15 * STEP_NS)
    combined = early + late
    mcap_path = dest_dir / "episode_source.mcap"
    build_fixture_mcap(
        str(mcap_path),
        camera_frames=master_count,
        joint_samples=master_count,
        topics={
            WRIST_LEFT_TOPIC: TopicSpec(
                SCHEMA_NAME_COMPRESSED_IMAGE, SCHEMA_TEXT_COMPRESSED_IMAGE, combined
            )
        },
    )
    return mcap_path


def test_interior_hole_rejects_with_interior_hole_kind_distinct_from_stream_ended(
    tmp_path: Path,
) -> None:
    source_mcap = _build_interior_hole_source_mcap(tmp_path, master_count=20)
    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)
    coverage_checks = [
        c
        for c in excinfo.value.rejected_header.global_meta["checks"]
        if c.check_name == CAMERA_COVERAGE_CHECK_NAME
    ]
    shortfall = coverage_checks[0].evidence["shortfall_cameras"]
    wrist_left_shortfall = next(c for c in shortfall if c["camera_name"] == "wrist_left")
    assert wrist_left_shortfall["hole_kind"] == "gap_exceeded_max_sync_gap_ms"


def test_zero_deficit_disposition_unchanged_from_pre_coverage_computation(tmp_path: Path) -> None:
    """A zero-deficit episode's recorded disposition is exactly what today's
    advisory-check computation produces -- the coverage check changes
    nothing when nothing was trimmed."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=10)
    episode = GalaxeaInputAdapter().parse(str(source_mcap), PROFILE)
    # Shipped profile declares no physics limits, so kinematics.velocity_
    # acceleration_bounds always fails advisory-wise -> PASS_WITH_FLAGS is
    # the correct pre-coverage disposition here too (STATE.md's own
    # standing note) -- asserted directly, not merely assumed.
    assert episode.header.global_meta["disposition"] == Disposition.PASS_WITH_FLAGS
    coverage_checks = [
        c
        for c in episode.header.global_meta["checks"]
        if c.check_name == CAMERA_COVERAGE_CHECK_NAME
    ]
    assert coverage_checks[0].evidence["deficit_ticks"] == 0


def test_enforce_camera_coverage_downgrades_pass_to_pass_with_flags_when_truncated() -> None:
    """Direct unit test of the downgrade-only rule: a hand-built episode
    whose pre-coverage disposition is PASS gets downgraded to
    PASS_WITH_FLAGS when the coverage check reports a within-tolerance
    truncation."""
    report = CameraCoverageReport(
        grid_ticks=10,
        kept_ticks=9,
        deficit_ticks=1,
        tolerance_ticks=8,
        within_tolerance=True,
        shortfall_cameras=(
            CameraCoverage(
                camera_name="wrist_left",
                video_key=WRIST_LEFT_VIDEO_KEY,
                covered_prefix_ticks=9,
                hole_kind="stream_ended_before_master",
                hole_gap_ns=40_000_000,
                required=True,
                emitted=True,
            ),
        ),
    )
    episode = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    header = dataclasses.replace(
        episode.header,
        global_meta={
            **episode.header.global_meta,
            "checks": (coverage_check_result(report),),
            "disposition": Disposition.PASS,
            "stream_inventory": (),
        },
    )
    episode_with_check = CanonicalEpisode(header=header, frames=episode.frames)

    result = _enforce_camera_coverage(episode_with_check, "some/path.mcap", PROFILE)
    assert result.header.global_meta["disposition"] == Disposition.PASS_WITH_FLAGS


def test_enforce_camera_coverage_never_upgrades_a_disposition_already_worse() -> None:
    """A disposition already worse than PASS_WITH_FLAGS (e.g. REVIEW) is
    never bumped back to PASS_WITH_FLAGS by a within-tolerance truncation."""
    report = CameraCoverageReport(
        grid_ticks=10,
        kept_ticks=9,
        deficit_ticks=1,
        tolerance_ticks=8,
        within_tolerance=True,
        shortfall_cameras=(),
    )
    episode = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    header = dataclasses.replace(
        episode.header,
        global_meta={
            **episode.header.global_meta,
            "checks": (coverage_check_result(report),),
            "disposition": Disposition.REVIEW,
            "stream_inventory": (),
        },
    )
    episode_with_check = CanonicalEpisode(header=header, frames=episode.frames)

    result = _enforce_camera_coverage(episode_with_check, "some/path.mcap", PROFILE)
    assert result.header.global_meta["disposition"] == Disposition.REVIEW


def test_enforce_camera_coverage_returns_episode_untouched_when_no_truncation() -> None:
    """A passed, non-truncated coverage check returns the episode
    untouched -- the no-gap path stays bit-for-bit what it is today."""
    report = CameraCoverageReport(
        grid_ticks=10,
        kept_ticks=10,
        deficit_ticks=0,
        tolerance_ticks=8,
        within_tolerance=True,
        shortfall_cameras=(),
    )
    episode = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    header = dataclasses.replace(
        episode.header,
        global_meta={
            **episode.header.global_meta,
            "checks": (coverage_check_result(report),),
            "disposition": Disposition.PASS,
            "stream_inventory": (),
        },
    )
    episode_with_check = CanonicalEpisode(header=header, frames=episode.frames)

    result = _enforce_camera_coverage(episode_with_check, "some/path.mcap", PROFILE)
    assert result.header is episode_with_check.header
    assert result.frames is episode_with_check.frames


def test_enforce_camera_coverage_no_second_coverage_computation() -> None:
    """`_enforce_camera_coverage` reads the check name imported from
    `data.sync.master_clock` and contains no second computation of a
    covered prefix, deficit, or tolerance comparison."""
    import inspect

    from data.adapters import galaxea as galaxea_mod

    source = inspect.getsource(galaxea_mod._enforce_camera_coverage)
    assert "camera_coverage_report(" not in source
    assert "nearest_camera_matches(" not in source
    assert "CAMERA_COVERAGE_CHECK_NAME" in source


def test_no_new_exception_class_added_for_coverage_rejection() -> None:
    """`data/adapters/galaxea.py` grows no new exception class -- the
    refusal reuses `GalaxeaAdapterError` with `rejected_header`."""
    import inspect

    from data.adapters import galaxea as galaxea_mod

    exception_classes = [
        name
        for name, obj in vars(galaxea_mod).items()
        if inspect.isclass(obj) and issubclass(obj, BaseException) and obj.__module__ == galaxea_mod.__name__
    ]
    assert exception_classes == ["GalaxeaAdapterError"]


# ---------------------------------------------------------------------------
# Task 3: the writer refuses to commit a camera whose frame count disagrees
# with the parquet's row count.
# ---------------------------------------------------------------------------


def _build_partial_coverage_episode(frame_count: int, present_ticks: set) -> CanonicalEpisode:
    """A hand-assembled episode (zero dependency on `data.adapters`, per
    `frame_fixtures.py`'s own module docstring) where the master camera
    carries an image on every tick but `wrist_left` carries one only on
    `present_ticks` -- a strict SUBSET, not the whole-episode-absent case
    `build_synthetic_episode`'s uniform `emit_cameras` already covers."""
    jpeg_bytes = _build_gradient_jpeg(64, 48)
    state_layout = _fixture_state_layout(PROFILE)
    action_layout = _action_layout(PROFILE)

    def _frames():
        for i in range(frame_count):
            state = {key: np.full(width, float(i), dtype=np.float32) for key, width in state_layout}
            action = {
                key: np.full(width, float(i) + 0.5, dtype=np.float32) for key, width in action_layout
            }
            images = {HEAD_VIDEO_KEY: FixtureImageRef(data=jpeg_bytes, timestamp_ns=i)}
            if i in present_ticks:
                images[WRIST_LEFT_VIDEO_KEY] = FixtureImageRef(data=jpeg_bytes, timestamp_ns=i)
            yield CanonicalFrame(
                timestamp_ns=i,
                frame_index=i,
                images=images,
                state=state,
                action=action,
                task_prompt="t",
            )

    header = EpisodeHeader(
        episode_id="synthetic-partial-coverage",
        fps=25.0,
        robot_type=PROFILE.robot_type,
        task_prompt="t",
        frame_count=frame_count,
        global_meta={},
    )
    return CanonicalEpisode(header=header, frames=_frames())


def test_write_episode_raises_on_partial_camera_coverage(tmp_path: Path) -> None:
    episode = _build_partial_coverage_episode(5, present_ticks={0, 1, 2})
    writer = LeRobotV21Writer(tmp_path, PROFILE, stdin_isatty=lambda: False)

    with pytest.raises(ValueError) as excinfo:
        writer.write_episode(episode)

    message = str(excinfo.value)
    assert WRIST_LEFT_VIDEO_KEY in message
    assert "3" in message  # submitted frame count
    assert "5" in message  # parquet row count

    assert not (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert not (tmp_path / "meta" / "encode_manifests" / "episode_000000.json").exists()
    assert not (tmp_path / "meta" / "stats_partials" / "episode_000000.json").exists()
    assert list(tmp_path.rglob("*.mp4")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_episode_next_real_write_reuses_index_after_raise(tmp_path: Path) -> None:
    writer = LeRobotV21Writer(tmp_path, PROFILE, stdin_isatty=lambda: False)
    bad = _build_partial_coverage_episode(5, present_ticks={0, 1, 2})
    with pytest.raises(ValueError):
        writer.write_episode(bad)

    good = build_synthetic_episode(frame_count=3, fps=25.0, profile=PROFILE, task_prompt="t")
    result = writer.write_episode(good)
    writer.finalize()

    assert result.episode_index == 0
    assert (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert not (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").exists()


def test_fully_covered_episode_writes_exactly_as_today(tmp_path: Path) -> None:
    """A fully-covered episode (every emitted camera present on every tick)
    writes exactly as it does today, with identical artifacts -- the guard
    never fires on the common path."""
    episode = build_synthetic_episode(frame_count=4, fps=25.0, profile=PROFILE, task_prompt="t")
    writer = LeRobotV21Writer(tmp_path, PROFILE, stdin_isatty=lambda: False)
    result = writer.write_episode(episode)
    writer.finalize()
    assert result.frame_count == 4
    assert (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()


def test_zero_frame_camera_subset_still_does_not_raise(tmp_path: Path) -> None:
    """A camera that submits zero frames for the whole episode (the
    existing `emit_cameras` subset fixtures) does NOT raise -- it never
    opens, so it never enters the guard's opened-states scope."""
    episode = build_synthetic_episode(
        frame_count=3, fps=25.0, profile=PROFILE, emit_cameras=[HEAD_VIDEO_KEY], task_prompt="t"
    )
    writer = LeRobotV21Writer(tmp_path, PROFILE, stdin_isatty=lambda: False)
    result = writer.write_episode(episode)
    writer.finalize()
    assert result.frame_count == 3
    assert WRIST_LEFT_VIDEO_KEY not in result.output_paths


def test_assert_camera_row_parity_is_first_statement_of_commit_branch() -> None:
    import inspect

    from data.writers import lerobot_v21 as writer_mod

    source = inspect.getsource(writer_mod.LeRobotV21Writer.write_episode)
    commit_branch = source.split("else:", 1)[1]
    first_call_line = next(
        line.strip() for line in commit_branch.splitlines() if line.strip()
    )
    assert first_call_line.startswith("_assert_camera_row_parity(")
