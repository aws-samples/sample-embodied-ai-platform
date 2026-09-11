"""Task 1's tracer, discovery, guard and revision tests for `data/cli.py`.

`test_tracer_...` (selected by `-k tracer`) is the end-to-end proof: one
fixture MCAP, through the real adapter/synchronizer/writer/recorder,
`finalize()`, and `validate_dataset`, into a prefix carrying a
schema-versioned run manifest, an immutable run report, and a recomputable
dataset revision identifier.
"""

import io
import json
import re
from pathlib import Path

import pytest
import yaml

from data.cli import (
    ACCEPTANCE_MODES,
    CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
    ETA_UNAVAILABLE,
    EXIT_CLEAN,
    EXIT_COMPLETED_WITH_FINDINGS,
    EXIT_CONSTANT_CHANNELS_UNACCEPTED,
    EXIT_PREFIX_REFUSED,
    EXIT_PREFIX_STATE_ERROR,
    EXIT_VALIDATION_FAILED,
    WRITE_UNEXPECTED_FAILURE_CHECK_NAME,
    ConstantChannelAcceptance,
    InputLedgerEntry,
    ProgressReporter,
    _assert_ledger_complete,
    _build_resume_maps,
    _derive_exit_code,
    _dry_run_one_input,
    _episode_written_map,
    constant_channel_finding,
    discover_inputs,
    main,
    resolve_constant_channel_acceptance,
    run_convert,
)
from data.contracts.containers import CheckResult, EpisodeHeader, IngestOutcome
from data.contracts.profile import load_profile
from data.promotion import PrefixVerdict, dataset_revision_id, evaluate_prefix, read_run_manifest
from data.provenance.recorder import PIPELINE_VERSION, ProvenanceRecorder, _resolve_git_commit
from data.run_config import RunConfig, load_run_config
from data.tests.mcap_fixture import corrupt_chunk_crc
from data.tests.mutators import build_fixture_source_mcap, force_constant_channel
from data.validation.constant_channels import CHECK_NAME as CONSTANT_CHANNEL_CHECK_NAME
from data.validation.correlation import CHECK_NAME as LAG_CHECK_NAME
from data.validation.image_histogram import CHECK_NAME as HISTOGRAM_CHECK_NAME
from data.validation.registry import UNEVALUATED_KEY
from data.validation.validate import ValidationOutcome, validate_dataset

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
PROFILE = load_profile(PROFILE_PATH)

_EMITTED_VIDEO_KEYS = [
    cam.video_key for cam in PROFILE.cameras.values() if cam.emit and cam.video_key is not None
]
"""Computed once against the shipped `PROFILE` — the exact comprehension
`run_convert`/`run_dry_run` use, reused by every 04-09 gap test below that
calls `_build_resume_maps`/`_episode_written_map` directly."""


def _write_run_config(dest_dir: Path, **overrides: object) -> Path:
    config: dict[str, object] = {
        "run_config_version": 1,
        "input_dir": None,
        "input_glob": "*.mcap",
        "inputs": None,
        "output_prefix": None,
        "format_key": "v2.1",
        "codec": "h264",
        "task_string": "auto",
        "fps_override": None,
        "emit_episodes_stats": False,
        **overrides,
    }
    config_path = dest_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _build_n_recordings_in_one_dir(
    input_dir: Path, count: int, *, camera_frames: int = 6
) -> list[Path]:
    """Builds `count` distinct fixture MCAPs directly under `input_dir` —
    each via its own staging subdirectory (so `build_fixture_source_mcap`'s
    fixed `episode_source.mcap` filename never collides), then renamed up
    into `input_dir` with a distinct name so a non-recursive `*.mcap` glob
    against `input_dir` itself discovers all of them.

    Each recording varies `camera_frames` by `i` — `build_fixture_source_mcap`
    is otherwise fully deterministic (no timestamp, UUID, or other
    per-call-varying content), so without this, `count` calls with the same
    `camera_frames` produce byte-IDENTICAL files (same sha256), which is
    genuinely wrong for a helper documented as building `count` DISTINCT
    fixture MCAPs and would make plan 04-02's content-hash duplicate
    collapsing (CLI-05) mistake these fixtures for the same recording under
    different names. Fixed here (Rule 1) rather than working around it at
    every call site."""
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        staged = build_fixture_source_mcap(
            input_dir / f"_staging_{i}", camera_frames=camera_frames + i
        )
        dest = input_dir / f"rec_{i:03d}.mcap"
        staged.rename(dest)
        paths.append(dest)
    return paths


def _build_converted_prefix(dest_dir: Path, out_dir: Path, *, camera_frames: int = 6) -> Path:
    """Converts one fixture recording into `out_dir` via the real CLI
    (`--no-validate`, so exit code is deterministic) — the RESUME/REFUSE/
    STATE_ERROR fixture every Task 3 guard test builds on."""
    source_mcap = build_fixture_source_mcap(dest_dir / "sources", camera_frames=camera_frames)
    config_path = _write_run_config(dest_dir, inputs=[str(source_mcap)])
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
    assert exit_code == EXIT_CLEAN
    return out_dir


def _profile_variant_path(tmp_path: Path) -> Path:
    """A copy of the shipped Galaxea profile with `schema_version` bumped —
    a harmless, valid edit that changes `profile_hash` without changing
    anything the adapter/writer actually reads."""
    text = PROFILE_PATH.read_text(encoding="utf-8")
    variant_text = text.replace("schema_version: 1", "schema_version: 2")
    assert variant_text != text
    variant_path = tmp_path / "profile_variant.yaml"
    variant_path.write_text(variant_text, encoding="utf-8")
    return variant_path


def _file_snapshot(root: Path) -> set[tuple[str, int, int]]:
    """(relative path, size, mtime_ns) for every file under `root` —
    proves a call left the prefix byte-identical when compared before/after."""
    return {
        (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


_KNOWN_CHANGING_RUN_METADATA_NAMES = {
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
    "meta/stats.json",
    "meta/modality.json",
    "meta/modality_config.py",
    "meta/data_engine/run_manifest.json",
    "meta/data_engine/validation_report.json",
}


def _file_snapshot_excluding_run_metadata(
    root: Path, *, extra_excluded_prefixes: tuple[str, ...] = ()
) -> set[tuple[str, int, int]]:
    """`_file_snapshot`, excluding the handful of dataset-wide files every
    run legitimately rewrites (`meta/info.json`, `meta/episodes.jsonl`,
    `meta/tasks.jsonl`, `meta/stats.json`, the run manifest, the validation
    report) and every immutable per-run report (a new file each run, never a
    modification of an existing one) — the subset a resume that appends or
    skips must leave byte-identical (CLI-08's append-only clause).

    `extra_excluded_prefixes` (default: none, so every existing caller's
    behavior is unchanged) additionally skips any relative path starting
    with one of the given prefixes — for a caller that runs `main()` twice
    WITH validation enabled, `meta/data_engine/episode_` is the prefix to
    pass: `ProvenanceRecorder.amend_checks` legitimately APPENDS to a
    written episode's `checks` list on every validated run (D-02/D-03,
    Phase 3 — pre-existing, unrelated to this batch), so a per-episode
    record's byte size genuinely grows each time validation re-runs over
    it. That growth is not a CLI-08 violation: resume logic
    (`_episode_written_map`/`_build_resume_maps`) reads only
    `ingest_outcome`, never `checks`, so the append is invisible to every
    resume decision — but it would otherwise defeat a naive byte-identity
    comparison across two validated runs for a reason unrelated to what
    that comparison exists to catch (an unintended rewrite of parquet,
    video, or a resume-read metadata field)."""
    snapshot = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel in _KNOWN_CHANGING_RUN_METADATA_NAMES:
            continue
        if rel.startswith("meta/data_engine/run_reports/"):
            continue
        if any(rel.startswith(prefix) for prefix in extra_excluded_prefixes):
            continue
        snapshot.add((rel, p.stat().st_size, p.stat().st_mtime_ns))
    return snapshot


def test_tracer_one_recording_end_to_end(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]
    )

    # The shipped Galaxea profile declares no physics limits (STATE.md), so
    # kinematics.velocity_acceleration_bounds is always unevaluated and
    # ValidationOutcome.fully_validated is always False for a real
    # conversion under it — EXIT_COMPLETED_WITH_FINDINGS, never EXIT_CLEAN,
    # is the correct and expected exit code here (Claude's Discretion item
    # 5, 04-01-PLAN.md).
    assert exit_code == EXIT_COMPLETED_WITH_FINDINGS

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 1

    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))

    manifest_path = out_dir / "meta" / "data_engine" / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 1
    assert manifest["pipeline_version"] == PIPELINE_VERSION
    assert len(manifest["runs"]) == 1
    assert manifest["runs"][0]["git_commit"]
    assert "package_versions" in manifest["runs"][0]

    report_path = out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json"
    assert report_path.exists()

    expected_short, expected_full = dataset_revision_id(
        [record["source_sha256"]],
        record["profile_hash"],
        _resolve_git_commit(),
        "v2.1",
    )
    assert manifest["dataset_revision"]["full"] == expected_full
    assert manifest["dataset_revision"]["short"] == expected_short
    assert len(expected_full) == 64
    assert expected_short == f"rev-{expected_full[:8]}"


# ---------------------------------------------------------------------------
# 04-09 gap closure: the four-consumer written-ness tracer.
#
# camera_frames=12, not 6: consistency.action_state_lag computes
# max_lag = min(MAX_LAG_SEARCH=5, row_count - 3) and abstains via its
# required_lag_magnitude > max_lag branch whenever max_lag < 4. At
# camera_frames=6, max_lag is 3, so the check abstains (exactly why
# test_tracer_one_recording_end_to_end above asserts
# EXIT_COMPLETED_WITH_FINDINGS rather than EXIT_VALIDATION_FAILED). At
# camera_frames=12, max_lag is 5, the declared -4 is inside the search
# window, and passed = (measured_lag == -4) is a real measurement that the
# default "ramp" fixture's real state/action coupling does not happen to
# satisfy (test_build_varying_baseline_dataset_ramp_default_does_not_
# falsely_pass_under_declared_lag, data/tests/test_validation_correlation.py)
# — a MEASURED fatal finding, never an abstention.
# ---------------------------------------------------------------------------


def test_gap_written_episode_stays_converted_across_all_four_consumers(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=12)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]
    )
    assert exit_code == EXIT_VALIDATION_FAILED

    # (i) the on-disk record carries disposition == REJECT (post-write
    # validation's amend) AND ingest_outcome == WRITTEN (never amended).
    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["disposition"] == "REJECT"
    assert record["ingest_outcome"] == IngestOutcome.WRITTEN.value

    # (ii) the validation report carries a MEASURED fatal finding, not an
    # abstention, for consistency.action_state_lag.
    validation_report = json.loads(
        (out_dir / "meta" / "data_engine" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    lag_result = next(
        r
        for r in validation_report["episode_results"]["0"]
        if r["check_name"] == "consistency.action_state_lag"
    )
    assert lag_result["severity"] == "fatal"
    assert lag_result["passed"] is False
    assert not lag_result["evidence"].get(UNEVALUATED_KEY, False)
    assert lag_result["evidence"]["measured_lag"] != -4

    # (iii) the run report's ledger entry for this input is bucketed
    # "converted".
    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["ledger"][0]["bucket"] == "converted"

    # (iv) _build_resume_maps agrees: the hash is converted, not rejected.
    converted_hash_to_index, rejected_hash_to_index = _build_resume_maps(
        out_dir, emitted_video_keys=_EMITTED_VIDEO_KEYS
    )
    assert record["source_sha256"] in converted_hash_to_index
    assert record["source_sha256"] not in rejected_hash_to_index

    # _episode_written_map is the ONLY place written-ness is decided —
    # asserted directly, not merely through _build_resume_maps' derived
    # buckets.
    written = _episode_written_map(out_dir, emitted_video_keys=_EMITTED_VIDEO_KEYS)
    assert written[0] is True

    # (v) the manifest's recorded dataset_revision includes exactly this
    # one hash — never a rejected input's hash, because there is none here,
    # but this pins the positive half of the same invariant Task 3 pins
    # negatively for an all-rejected batch.
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_revision"]["inputs"]["source_sha256s"] == [record["source_sha256"]]

    # (vi) already asserted above: exit_code is EXIT_VALIDATION_FAILED,
    # because a measured fatal check drives ValidationOutcome.exit_code != 0
    # and that outranks everything in _derive_exit_code.


def test_gap_retry_rejected_does_not_duplicate_a_validation_relabeled_episode(
    tmp_path: Path,
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=12)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    first_exit = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]
    )
    assert first_exit == EXIT_VALIDATION_FAILED

    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    ingest_outcome_before = json.loads(record_path.read_text(encoding="utf-8"))["ingest_outcome"]
    assert ingest_outcome_before == IngestOutcome.WRITTEN.value

    # A SECOND main() on the same prefix with --retry-rejected and
    # validation enabled. Because our fix classifies this episode as
    # converted (ingest_outcome == WRITTEN), it is bucketed resumed_skip
    # regardless of --retry-rejected — the flag only re-opens a hash that
    # is genuinely in rejected_hash_to_index.
    second_exit = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--retry-rejected",
        ]
    )
    assert second_exit == EXIT_VALIDATION_FAILED

    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["runs"]) == 2

    second_report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_report["ledger"][0]["bucket"] == "resumed_skip"

    # No second episode index consumed: exactly one provenance record, one
    # parquet.
    data_engine_dir = out_dir / "meta" / "data_engine"
    episode_records = [
        p for p in sorted(data_engine_dir.glob("episode_*.json")) if p.stem.split("_")[1].isdigit()
    ]
    assert len(episode_records) == 1

    parquet_dir = out_dir / "data" / "chunk-000"
    assert len(list(parquet_dir.glob("episode_*.parquet"))) == 1

    # Test 3 (behavior): ingest_outcome is immutable across the second
    # run's amend — disposition is permitted to differ (it does not, here,
    # both runs relabel it REJECT), but ingest_outcome must not move.
    ingest_outcome_after = json.loads(record_path.read_text(encoding="utf-8"))["ingest_outcome"]
    assert ingest_outcome_after == ingest_outcome_before == IngestOutcome.WRITTEN.value


def test_empty_run_config_raises_value_error_and_creates_nothing(tmp_path: Path) -> None:
    config_path = tmp_path / "empty_run_config.yaml"
    config_path.write_text("", encoding="utf-8")
    out_dir = tmp_path / "output"

    with pytest.raises(ValueError, match=str(config_path)):
        load_run_config(config_path)
    assert not out_dir.exists()

    with pytest.raises(ValueError, match=str(config_path)):
        main([str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)])
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Task 2: multi-input batch — deterministic discovery, resolved-path
# safety, revision over the full source set.
# ---------------------------------------------------------------------------


def test_three_recordings_convert_in_sorted_resolved_path_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

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
    assert exit_code == EXIT_CLEAN

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3

    expected_order = sorted(recordings, key=lambda p: p.resolve().as_posix())
    for idx, rec in enumerate(expected_order):
        record_path = out_dir / "meta" / "data_engine" / f"episode_{idx:06d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["source_uri"] == str(rec.resolve())


def test_two_prefixes_from_same_directory_assign_same_index_to_same_recording(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 3)
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    out_dir_1 = tmp_path / "output_1"
    out_dir_2 = tmp_path / "output_2"
    for out_dir in (out_dir_1, out_dir_2):
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
        assert exit_code == EXIT_CLEAN

    def _index_by_source(out_dir: Path) -> dict[str, int]:
        mapping = {}
        for i in range(3):
            record_path = out_dir / "meta" / "data_engine" / f"episode_{i:06d}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            mapping[record["source_uri"]] = i
        return mapping

    assert _index_by_source(out_dir_1) == _index_by_source(out_dir_2)


def test_explicit_inputs_list_uses_order_verbatim_even_reversed(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 3)
    reversed_inputs = sorted((str(p) for p in recordings), reverse=True)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=reversed_inputs)

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
    assert exit_code == EXIT_CLEAN

    for idx, path_str in enumerate(reversed_inputs):
        record_path = out_dir / "meta" / "data_engine" / f"episode_{idx:06d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["source_uri"] == str(Path(path_str).resolve())


def test_zero_glob_matches_exits_nonzero_and_creates_nothing(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty_inputs"
    input_dir.mkdir()
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    with pytest.raises(ValueError) as exc_info:
        main([str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)])

    message = str(exc_info.value)
    assert str(input_dir.resolve()) in message
    assert "*.mcap" in message
    assert not (out_dir / "meta").exists()


def test_symlink_duplicate_collapses_to_one_conversion(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    symlink_path = source_mcap.parent / "symlinked.mcap"
    symlink_path.symlink_to(source_mcap)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap), str(symlink_path)])

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
    assert exit_code == EXIT_CLEAN

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 1

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["bucket"] for entry in report["ledger"]] == ["converted", "duplicate_source"]


def test_discover_inputs_refuses_path_outside_input_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    outside_mcap = build_fixture_source_mcap(tmp_path / "outside", camera_frames=2)
    escaping_glob = f"../{outside_mcap.parent.name}/{outside_mcap.name}"
    run_cfg = RunConfig(
        run_config_version=1, input_dir=str(input_dir), input_glob=escaping_glob
    )

    with pytest.raises(ValueError, match=str(outside_mcap.resolve())):
        discover_inputs(run_cfg)


def test_all_rejected_batch_exits_with_findings_and_reports_every_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 3)
    for rec in recordings:
        corrupt_chunk_crc(rec)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

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

    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["runs"]) == 1

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["ledger"]) == 3
    assert all(entry["bucket"] == "newly_rejected" for entry in report["ledger"])


def test_dataset_revision_id_over_three_hashes_is_order_independent() -> None:
    hashes = ["aa" * 32, "bb" * 32, "cc" * 32]
    forward = dataset_revision_id(hashes, "profile-hash", "git-commit", "v2.1")
    backward = dataset_revision_id(list(reversed(hashes)), "profile-hash", "git-commit", "v2.1")
    assert forward == backward


def test_ledger_completeness_assertion_raises_on_dropped_entry() -> None:
    ledger = (
        InputLedgerEntry(
            source_path="a",
            source_sha256="x",
            bucket="converted",
            episode_index=0,
            frame_count=1,
            disposition="PASS",
            checks=(),
            elapsed_seconds=0.0,
        ),
    )
    with pytest.raises(ValueError, match="1"):
        _assert_ledger_complete(ledger, 2)


# ---------------------------------------------------------------------------
# Task 3: the CLI-08 promotion guard.
# ---------------------------------------------------------------------------


def test_evaluate_prefix_fresh_for_nonexistent_and_empty_prefix(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"
    decision = evaluate_prefix(
        nonexistent, pipeline_version="1.0.0", profile_hash_value="p", format_key="v2.1"
    )
    assert decision.verdict is PrefixVerdict.FRESH

    existing_but_empty = tmp_path / "existing_empty"
    existing_but_empty.mkdir()
    decision2 = evaluate_prefix(
        existing_but_empty, pipeline_version="1.0.0", profile_hash_value="p", format_key="v2.1"
    )
    assert decision2.verdict is PrefixVerdict.FRESH


def test_evaluate_prefix_resume_when_every_field_matches(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )

    decision = evaluate_prefix(
        out_dir,
        pipeline_version=manifest["pipeline_version"],
        profile_hash_value=manifest["profile_hash"],
        format_key=manifest["format_key"],
    )
    assert decision.verdict is PrefixVerdict.RESUME
    assert decision.recorded_source_sha256s


@pytest.mark.parametrize("field", ["profile_hash", "pipeline_version", "format_key"])
def test_evaluate_prefix_refuses_on_each_mismatched_field(tmp_path: Path, field: str) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )

    kwargs = {
        "pipeline_version": manifest["pipeline_version"],
        "profile_hash_value": manifest["profile_hash"],
        "format_key": manifest["format_key"],
    }
    key = "profile_hash_value" if field == "profile_hash" else field
    kwargs[key] = "definitely-different-value"

    decision = evaluate_prefix(out_dir, **kwargs)
    assert decision.verdict is PrefixVerdict.REFUSE
    assert decision.mismatched_field == field
    assert decision.expected is not None
    assert decision.found == "definitely-different-value"


def test_evaluate_prefix_comparison_is_case_exact(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )

    swapped_case = manifest["profile_hash"].swapcase()
    assert swapped_case != manifest["profile_hash"]

    decision = evaluate_prefix(
        out_dir,
        pipeline_version=manifest["pipeline_version"],
        profile_hash_value=swapped_case,
        format_key=manifest["format_key"],
    )
    assert decision.verdict is PrefixVerdict.REFUSE
    assert decision.mismatched_field == "profile_hash"


def test_evaluate_prefix_resume_from_records_alone_when_no_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest_path = out_dir / "meta" / "data_engine" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.unlink()

    decision = evaluate_prefix(
        out_dir,
        pipeline_version=manifest["pipeline_version"],
        profile_hash_value=manifest["profile_hash"],
        format_key=manifest["format_key"],
    )
    assert decision.verdict is PrefixVerdict.RESUME


def test_evaluate_prefix_state_error_on_manifest_record_disagreement(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest_path = out_dir / "meta" / "data_engine" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile_hash"] = "corrupted-for-state-error-test"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    decision = evaluate_prefix(
        out_dir,
        pipeline_version=manifest["pipeline_version"],
        profile_hash_value=manifest["profile_hash"],
        format_key=manifest["format_key"],
    )
    assert decision.verdict is PrefixVerdict.STATE_ERROR
    assert decision.mismatched_field == "profile_hash"
    assert "0" in decision.reason


def test_cli_refuses_mismatched_prefix_and_leaves_it_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    before = _file_snapshot(out_dir)

    variant_profile_path = _profile_variant_path(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_source = build_fixture_source_mcap(other_dir / "sources", camera_frames=6)
    other_config_path = _write_run_config(other_dir, inputs=[str(other_source)])

    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(variant_profile_path),
            "--run-config",
            str(other_config_path),
        ]
    )

    assert exit_code == EXIT_PREFIX_REFUSED
    assert _file_snapshot(out_dir) == before
    stderr = capsys.readouterr().err
    assert "profile_hash" in stderr


def test_cli_returns_state_error_exit_code_on_manifest_record_disagreement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest_path = out_dir / "meta" / "data_engine" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile_hash"] = "corrupted-for-state-error-test"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    before = _file_snapshot(out_dir)

    unused_dir = tmp_path / "unused"
    unused_dir.mkdir()
    unused_source = build_fixture_source_mcap(unused_dir / "sources", camera_frames=2)
    unused_config_path = _write_run_config(unused_dir, inputs=[str(unused_source)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(unused_config_path)]
    )

    assert exit_code == EXIT_PREFIX_STATE_ERROR
    assert _file_snapshot(out_dir) == before
    stderr = capsys.readouterr().err
    assert "profile_hash" in stderr


def test_evaluate_prefix_writes_nothing(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"
    assert not nonexistent.exists()
    fresh_decision = evaluate_prefix(
        nonexistent, pipeline_version="1.0.0", profile_hash_value="p", format_key="v2.1"
    )
    assert fresh_decision.verdict is PrefixVerdict.FRESH
    assert not nonexistent.exists()

    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )

    for kwargs, expected_verdict in (
        (
            {
                "pipeline_version": manifest["pipeline_version"],
                "profile_hash_value": manifest["profile_hash"],
                "format_key": manifest["format_key"],
            },
            PrefixVerdict.RESUME,
        ),
        (
            {
                "pipeline_version": manifest["pipeline_version"],
                "profile_hash_value": "a-completely-different-profile-hash",
                "format_key": manifest["format_key"],
            },
            PrefixVerdict.REFUSE,
        ),
    ):
        before = _file_snapshot(out_dir)
        decision = evaluate_prefix(out_dir, **kwargs)
        after = _file_snapshot(out_dir)
        assert decision.verdict is expected_verdict
        assert before == after

    manifest_path = out_dir / "meta" / "data_engine" / "run_manifest.json"
    corrupted = dict(manifest)
    corrupted["profile_hash"] = "corrupted-for-state-error-test"
    manifest_path.write_text(json.dumps(corrupted, sort_keys=True, indent=2), encoding="utf-8")

    before = _file_snapshot(out_dir)
    state_error_decision = evaluate_prefix(
        out_dir,
        pipeline_version=manifest["pipeline_version"],
        profile_hash_value=manifest["profile_hash"],
        format_key=manifest["format_key"],
    )
    after = _file_snapshot(out_dir)
    assert state_error_decision.verdict is PrefixVerdict.STATE_ERROR
    assert before == after


def test_manifest_dataset_revision_recomputes_from_its_own_recorded_inputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    inputs = manifest["dataset_revision"]["inputs"]
    _, recomputed_full = dataset_revision_id(
        inputs["source_sha256s"], inputs["profile_hash"], inputs["git_commit"], inputs["format_key"]
    )
    assert recomputed_full == manifest["dataset_revision"]["full"]


def test_promotion_module_has_no_case_or_whitespace_normalization() -> None:
    promotion_source = Path(
        Path(__file__).parent.parent / "promotion.py"
    ).read_text(encoding="utf-8")
    forbidden = (".lower(", ".upper(", ".casefold(", ".strip(", "unicodedata.normalize")
    for pattern in forbidden:
        assert pattern not in promotion_source, pattern


# ---------------------------------------------------------------------------
# Phase 4 Plan 2, Task 2: hash-backed resume skip set, crash-debris
# reclamation, and rejected-input accounting across runs.
# ---------------------------------------------------------------------------


def test_second_identical_invocation_skips_all_and_leaves_totals_unchanged(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]

    assert main(args) == EXIT_CLEAN
    info_after_first = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))

    exit_code_2 = main(args)
    assert exit_code_2 == EXIT_CLEAN

    info_after_second = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info_after_second["total_episodes"] == info_after_first["total_episodes"] == 3

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["bucket"] for entry in report["ledger"]] == ["resumed_skip"] * 3


def test_interrupted_run_resumes_converting_only_the_missing_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]
    assert main(args) == EXIT_CLEAN

    # Simulate a genuinely interrupted run per D-03: episode 2's write
    # reached the writer (parquet/video/encode-manifest/stats-partial all
    # exist) but the process died before finalize() ever ran, so its
    # provenance record AND its episodes.jsonl/tasks.jsonl rows (both
    # written only by finalize(), never incrementally) are all absent --
    # never just the provenance record alone, which would leave the
    # writer's own episodes.jsonl-derived rehydration disagreeing with the
    # CLI's provenance-record-derived skip set.
    (out_dir / "meta" / "data_engine" / "episode_000002.json").unlink()
    for name in ("episodes.jsonl", "tasks.jsonl"):
        jsonl_path = out_dir / "meta" / name
        remaining_lines = jsonl_path.read_text(encoding="utf-8").splitlines()[:-1]
        jsonl_path.write_text(
            "".join(line + "\n" for line in remaining_lines), encoding="utf-8"
        )

    exit_code = main(args)
    assert exit_code == EXIT_CLEAN

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    buckets = [entry["bucket"] for entry in report["ledger"]]
    assert buckets.count("resumed_skip") == 2
    converted_entries = [e for e in report["ledger"] if e["bucket"] == "converted"]
    assert len(converted_entries) == 1
    assert converted_entries[0]["episode_index"] == 2

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3


def test_crash_debris_reclaimed_and_recorded_indices_untouched(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]
    assert main(args) == EXIT_CLEAN

    # Plant crash debris at a brand-new index with no provenance record at
    # all — a completely separate scenario from "the last episode's record
    # is missing" (that is the interrupted-run test above).
    debris_index = 99
    parquet_path = out_dir / "data" / "chunk-000" / f"episode_{debris_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"not a real parquet file")

    manifest_dir = out_dir / "meta" / "encode_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    debris_manifest_path = manifest_dir / f"episode_{debris_index:06d}.json"
    debris_manifest_path.write_text("{}", encoding="utf-8")

    before = _file_snapshot_excluding_run_metadata(out_dir)

    exit_code = main(args)
    assert exit_code == EXIT_CLEAN

    after = _file_snapshot_excluding_run_metadata(out_dir)
    # Every pre-existing recorded episode's artifact is byte-identical;
    # the crash-debris files are gone, so `after` is a strict subset.
    assert after < before
    assert not parquet_path.exists()
    assert not debris_manifest_path.exists()

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    reclaimed_entries = [e for e in report["ledger"] if e["bucket"] == "reclaimed"]
    assert len(reclaimed_entries) == 1
    assert reclaimed_entries[0]["episode_index"] == debris_index
    assert str(parquet_path) in reclaimed_entries[0]["deleted_paths"]
    assert str(debris_manifest_path) in reclaimed_entries[0]["deleted_paths"]

    # Recorded episodes 0/1/2 are entirely untouched by reclamation.
    for i in range(3):
        record_path = out_dir / "meta" / "data_engine" / f"episode_{i:06d}.json"
        assert record_path.exists()


def test_rejected_input_skipped_then_retried_with_retry_rejected_flag(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 1)
    corrupt_chunk_crc(recordings[0])
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    base_args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]

    assert main(base_args) == EXIT_COMPLETED_WITH_FINDINGS
    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    assert record_path.exists()
    first_mtime = record_path.stat().st_mtime_ns

    # Re-run without --retry-rejected: skipped, no decode attempted, the
    # existing rejection record is byte-for-byte untouched.
    assert main(base_args) == EXIT_COMPLETED_WITH_FINDINGS
    report_2 = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [e["bucket"] for e in report_2["ledger"]] == ["rejected_skip"]
    assert report_2["ledger"][0]["disposition"] == "REJECT"
    assert record_path.stat().st_mtime_ns == first_mtime

    # Re-run WITH --retry-rejected: re-parsed at full cost, still fails (the
    # corruption is still there), and SUPERSEDES the same index rather than
    # consuming a fresh one.
    assert main([*base_args, "--retry-rejected"]) == EXIT_COMPLETED_WITH_FINDINGS
    report_3 = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_002.json").read_text(
            encoding="utf-8"
        )
    )
    assert [e["bucket"] for e in report_3["ledger"]] == ["newly_rejected"]
    assert report_3["ledger"][0]["episode_index"] == 0

    # WR-05: a retry that FAILS again overwrites at retry_target_index (an
    # index the record already occupied — never a new one), so the
    # supersession marker must never fire on this path.
    final_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert final_payload["superseded_by_episode_index"] is None


def test_every_pre_existing_episode_artifact_byte_identical_across_a_resume(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 2)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    assert (
        main(
            [
                str(out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(config_path),
                "--no-validate",
            ]
        )
        == EXIT_CLEAN
    )

    before = _file_snapshot_excluding_run_metadata(out_dir)

    # A resume that discovers one additional new input alongside the two
    # already-converted ones must leave every pre-existing entry unchanged
    # -- only new files may be added.
    input_dir_2 = tmp_path / "inputs"
    # camera_frames=99 keeps this genuinely distinct in content (and thus
    # sha256) from the two recordings _build_n_recordings_in_one_dir already
    # built (camera_frames 6 and 7) -- a collision here would make the new
    # input look already-converted (resumed_skip) and defeat this test's
    # entire premise.
    build_fixture_source_mcap(input_dir_2 / "extra", camera_frames=99).rename(
        input_dir_2 / "rec_003.mcap"
    )
    config_path_2 = _write_run_config(tmp_path, input_dir=str(input_dir_2), input_glob="*.mcap")
    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path_2),
            "--no-validate",
        ]
    )
    assert exit_code == EXIT_CLEAN

    after = _file_snapshot_excluding_run_metadata(out_dir)
    assert before <= after
    assert before != after


def test_zero_frame_write_consumes_no_record_and_next_real_episode_reuses_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import data.cli as cli_module
    from data.contracts.containers import CanonicalEpisode, EpisodeHeader

    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 2)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    real_adapter_cls = cli_module.GalaxeaInputAdapter
    call_count = {"n": 0}

    class _ZeroFrameOnFirstCallAdapter:
        def parse(self, path: str, profile: object) -> CanonicalEpisode:
            call_count["n"] += 1
            if call_count["n"] == 1:
                header = EpisodeHeader(
                    episode_id=path,
                    fps=25.0,
                    robot_type="galaxea_r1_pro",
                    task_prompt=None,
                    frame_count=0,
                    global_meta={},
                )
                return CanonicalEpisode(header=header, frames=iter(()))
            return real_adapter_cls().parse(path, profile)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "GalaxeaInputAdapter", _ZeroFrameOnFirstCallAdapter)

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

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    buckets = [entry["bucket"] for entry in report["ledger"]]
    assert buckets[0] == "newly_rejected"
    assert report["ledger"][0]["frame_count"] == 0
    assert report["ledger"][0]["episode_index"] is None

    assert buckets[1] == "converted"
    assert report["ledger"][1]["episode_index"] == 0
    assert not (out_dir / "meta" / "data_engine" / "episode_000001.json").exists()
    assert (out_dir / "meta" / "data_engine" / "episode_000000.json").exists()


# ---------------------------------------------------------------------------
# Phase 4 Plan 2, Task 3: dry run, tty-aware progress, and the
# every-input-accounted run report.
# ---------------------------------------------------------------------------


def test_dry_run_reports_action_sources_and_predicted_frame_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--dry-run",
        ]
    )
    assert exit_code == EXIT_CLEAN
    stdout = capsys.readouterr().out
    assert stdout.count("would convert") == 3
    assert "predicted_frame_count=" in stdout
    assert "resolved_action_sources=" in stdout
    assert not out_dir.exists()


def test_dry_run_reports_prefix_verdict_and_revision_short_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path), "--dry-run"]
    )
    assert exit_code == EXIT_CLEAN
    stdout = capsys.readouterr().out
    assert "prefix verdict: FRESH" in stdout
    assert re.search(r"rev-[0-9a-f]{8}\b", stdout)


def test_dry_run_leaves_nonexistent_and_existing_prefix_unchanged(
    tmp_path: Path,
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    nonexistent_out_dir = tmp_path / "nonexistent_output"
    assert (
        main(
            [
                str(nonexistent_out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(config_path),
                "--dry-run",
            ]
        )
        == EXIT_CLEAN
    )
    assert not nonexistent_out_dir.exists()

    existing_out_dir = tmp_path / "existing_output"
    _build_converted_prefix(tmp_path / "converted_source", existing_out_dir)
    before = _file_snapshot(existing_out_dir)

    assert (
        main(
            [
                str(existing_out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(config_path),
                "--dry-run",
            ]
        )
        == EXIT_CLEAN
    )
    after = _file_snapshot(existing_out_dir)
    assert before == after


def test_dry_run_never_constructs_a_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import data.cli as cli_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("get_writer must never be called during --dry-run")

    monkeypatch.setattr(cli_module, "get_writer", _boom)

    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path), "--dry-run"]
    )
    assert exit_code == EXIT_CLEAN


def test_dry_run_reports_resume_skip_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "output"
    _build_converted_prefix(tmp_path, out_dir)
    config_path = _write_run_config(tmp_path, input_dir=str(tmp_path / "sources"), input_glob="*.mcap")

    # Makes "exactly once per discovered input" mechanical rather than
    # inspection-only (04-09 Task 3): a future refactor that re-ran the
    # index pass for the prediction would still pass every other gate in
    # this file, but would fail this counter assertion.
    import data.cli as cli_module

    real_dry_run_one_input = cli_module._dry_run_one_input
    call_count = 0

    def _counting_dry_run_one_input(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return real_dry_run_one_input(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "_dry_run_one_input", _counting_dry_run_one_input)

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path), "--dry-run"]
    )
    assert exit_code == EXIT_CLEAN
    stdout = capsys.readouterr().out
    assert "resumed_skip" in stdout

    discovered_count = len(list((tmp_path / "sources").glob("*.mcap")))
    assert call_count == discovered_count == 1


def test_progress_reporter_tty_branch_carriage_return_no_intermediate_newline() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(3, stream=stream, isatty=lambda: True)
    for _ in range(3):
        reporter.update(frames_this_episode=10)
    reporter.finish()

    captured = stream.getvalue()
    assert "\r" in captured
    # Only the final finish() call may contribute a newline.
    assert captured.count("\n") == 1
    assert captured.endswith("\n")


def test_progress_reporter_non_tty_branch_appends_lines_no_carriage_return() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(3, stream=stream, isatty=lambda: False)
    for _ in range(3):
        reporter.update(frames_this_episode=10)
    reporter.finish()

    captured = stream.getvalue()
    assert "\r" not in captured
    assert captured.count("\n") == 3


def test_progress_reporter_eta_unavailable_before_two_completions() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(5, stream=stream, isatty=lambda: False)
    reporter.update(frames_this_episode=10)
    captured_after_one = stream.getvalue()
    assert ETA_UNAVAILABLE in captured_after_one

    reporter.update(frames_this_episode=10)
    captured_after_two = stream.getvalue()
    assert ETA_UNAVAILABLE not in captured_after_two.splitlines()[-1]


def test_stdout_across_full_run_contains_only_report_path_and_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path), "--no-validate"]
    )
    assert exit_code == EXIT_CLEAN
    stdout = capsys.readouterr().out
    lines = [line for line in stdout.splitlines() if line.strip()]
    report_path = out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json"
    assert lines == [str(report_path), "rev-" + json.loads(report_path.read_text(encoding="utf-8"))["dataset_revision"]["full"][:8]]
    assert "episode" not in stdout
    assert "fps" not in stdout


def test_bucket_counts_sums_to_inputs_discovered_on_a_mixed_batch(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 2)
    corrupted = build_fixture_source_mcap(tmp_path / "corrupted_source", camera_frames=30)
    corrupt_chunk_crc(corrupted)
    all_inputs = [str(p) for p in recordings] + [str(corrupted)]

    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=all_inputs)
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]
    assert main(args) == EXIT_COMPLETED_WITH_FINDINGS

    # Plant crash debris at a brand-new index with no provenance record.
    debris_index = 77
    parquet_path = out_dir / "data" / "chunk-000" / f"episode_{debris_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"debris")

    exit_code = main(args)
    assert exit_code == EXIT_COMPLETED_WITH_FINDINGS

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["inputs_discovered"] == 3
    assert report["bucket_counts"].get("reclaimed", 0) == 1
    per_input_bucket_sum = sum(
        count for bucket, count in report["bucket_counts"].items() if bucket != "reclaimed"
    )
    assert per_input_bucket_sum == report["inputs_discovered"]


def test_unevaluated_check_marked_unevaluated_and_drives_completed_with_findings(
    tmp_path: Path,
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    exit_code = main(
        [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]
    )
    # The shipped Galaxea profile declares no physics limits, so
    # kinematics.velocity_acceleration_bounds is always unevaluated.
    assert exit_code == EXIT_COMPLETED_WITH_FINDINGS

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["validation"]["invoked"] is True
    assert report["validation"]["fully_validated"] is False
    assert report["validation"]["unevaluated_count"] >= 1

    found_unevaluated = False
    for results in report["validation"]["episode_results"].values():
        for check in results:
            if check["evidence"].get("unevaluated"):
                found_unevaluated = True
    for check in report["validation"]["dataset_results"]:
        if check["evidence"].get("unevaluated"):
            found_unevaluated = True
    assert found_unevaluated


def test_no_validate_reports_validation_not_invoked(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
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
    assert exit_code == EXIT_CLEAN

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["validation"] == {
        "invoked": False,
        "fully_validated": None,
        "exit_code": None,
        "unevaluated_count": None,
        "dataset_results": [],
        "episode_results": {},
    }


def test_converted_entries_carry_positive_elapsed_decode_encode_seconds(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
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
    assert exit_code == EXIT_CLEAN

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    entry = report["ledger"][0]
    assert entry["bucket"] == "converted"
    assert entry["elapsed_seconds"] > 0.0
    assert entry["decode_seconds"] > 0.0
    assert entry["encode_seconds"] > 0.0
    assert entry["elapsed_seconds"] >= entry["decode_seconds"] + entry["encode_seconds"]


def test_per_entry_decode_encode_seconds_timing_reconciles_with_writer_cumulative(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 3)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

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
    assert exit_code == EXIT_CLEAN

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    converted_entries = [e for e in report["ledger"] if e["bucket"] == "converted"]
    assert len(converted_entries) == 3

    summed_decode = sum(e["decode_seconds"] for e in converted_entries)
    summed_encode = sum(e["encode_seconds"] for e in converted_entries)

    # A raw-cumulative-instead-of-delta bug would make this summed figure a
    # multiple of the writer's actual final cumulative total (1x + 2x + 3x =
    # 6x, not 3x) -- this equality is exactly what discriminates a correct
    # per-episode delta from a monotonically-inflating cumulative reading,
    # which a bare "is it non-zero" check cannot.
    assert summed_decode == pytest.approx(report["cumulative_decode_seconds"], abs=1e-6)
    assert summed_encode == pytest.approx(report["cumulative_encode_seconds"], abs=1e-6)


# ---------------------------------------------------------------------------
# Task 2 (G-03): the constant-channel acceptance gate.
# ---------------------------------------------------------------------------


def _constant_channel_result(
    *, passed: bool, flagged_channels: list[dict[str, object]] | None = None, unevaluated: bool = False
) -> CheckResult:
    evidence: dict[str, object] = {"flagged_channels": flagged_channels or []}
    if unevaluated:
        evidence[UNEVALUATED_KEY] = True
    return CheckResult(CONSTANT_CHANNEL_CHECK_NAME, "advisory", passed, evidence)


def _flagged(feature: str, joint_name: str) -> dict[str, object]:
    return {
        "feature": feature,
        "channel_index": 0,
        "joint_name": joint_name,
        "profile_key": "state_vector_order",
        "merged_min": 0.0,
        "merged_max": 0.0,
        "likely_cause": "possibly_intentionally_still",
    }


def _outcome_with(result: CheckResult) -> ValidationOutcome:
    return ValidationOutcome(
        episode_results={},
        dataset_results=(result,),
        fully_validated=True,
        exit_code=0,
    )


def test_constant_channel_finding_none_when_validation_not_invoked() -> None:
    assert constant_channel_finding(None) is None


def test_constant_channel_finding_returns_passing_result_verbatim() -> None:
    # constant_channel_finding only filters absence/unevaluated; the
    # "passed -> no consent needed" decision is resolve_constant_channel_
    # acceptance's job, exercised separately below.
    outcome = _outcome_with(_constant_channel_result(passed=True))
    finding = constant_channel_finding(outcome)
    assert finding is not None
    assert finding.passed is True


def test_constant_channel_finding_none_when_unevaluated() -> None:
    outcome = _outcome_with(_constant_channel_result(passed=False, unevaluated=True))
    assert constant_channel_finding(outcome) is None


def test_constant_channel_finding_present_when_flagged() -> None:
    result = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("observation.state", "arm_l_j1")]
    )
    outcome = _outcome_with(result)
    finding = constant_channel_finding(outcome)
    assert finding is not None
    assert finding.check_name == CONSTANT_CHANNEL_CHECK_NAME


def test_resolve_acceptance_not_evaluated_when_validation_not_invoked() -> None:
    acceptance = resolve_constant_channel_acceptance(
        None, invoked=False, accept_flag=False, stdin_isatty=False, prompt=None
    )
    assert acceptance.mode == "not-evaluated"
    assert acceptance.finding_present is False
    assert acceptance.accepted is None
    assert acceptance.channels == ()


def test_resolve_acceptance_no_finding_when_check_passed() -> None:
    acceptance = resolve_constant_channel_acceptance(
        None, invoked=True, accept_flag=False, stdin_isatty=False, prompt=None
    )
    assert acceptance.mode == "no-finding"
    assert acceptance.finding_present is False
    assert acceptance.accepted is None


def test_resolve_acceptance_flag_short_circuits_the_prompt() -> None:
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("observation.state", "arm_l_j1")]
    )

    def _fail_if_called(_message: str) -> str:
        raise AssertionError("prompt must not be called when accept_flag short-circuits it")

    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=True, stdin_isatty=True, prompt=_fail_if_called
    )
    assert acceptance.mode == "flag"
    assert acceptance.finding_present is True
    assert acceptance.accepted is True
    assert acceptance.channels == ("observation.state.arm_l_j1",)


def test_resolve_acceptance_prompt_accepted_on_affirmative_answer() -> None:
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("action", "arm_l_j5")]
    )
    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=False, stdin_isatty=True, prompt=lambda _m: "yes"
    )
    assert acceptance.mode == "prompt-accepted"
    assert acceptance.accepted is True


@pytest.mark.parametrize("answer", ["no", "", "n", "maybe"])
def test_resolve_acceptance_prompt_declined_on_non_affirmative_or_empty_answer(
    answer: str,
) -> None:
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("action", "arm_l_j5")]
    )
    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=False, stdin_isatty=True, prompt=lambda _m: answer
    )
    assert acceptance.mode == "prompt-declined"
    assert acceptance.accepted is False


def test_resolve_acceptance_unattended_no_flag_never_prompts() -> None:
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("observation.state", "arm_l_j1")]
    )

    def _fail_if_called(_message: str) -> str:
        raise AssertionError("prompt must never be called with no tty and no flag")

    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=False, stdin_isatty=False, prompt=_fail_if_called
    )
    assert acceptance.mode == "unattended-no-flag"
    assert acceptance.accepted is False
    assert acceptance.finding_present is True


def test_acceptance_modes_names_all_six_situations() -> None:
    assert set(ACCEPTANCE_MODES) == {
        "not-evaluated",
        "no-finding",
        "flag",
        "prompt-accepted",
        "prompt-declined",
        "unattended-no-flag",
    }
    assert len(ACCEPTANCE_MODES) == 6


def test_channels_are_feature_joined_to_joint_name_and_sorted() -> None:
    finding = _constant_channel_result(
        passed=False,
        flagged_channels=[
            _flagged("observation.state", "arm_l_j7"),
            _flagged("action", "arm_l_j1"),
        ],
    )
    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=True, stdin_isatty=False, prompt=None
    )
    assert acceptance.channels == ("action.arm_l_j1", "observation.state.arm_l_j7")


def test_training_risk_states_the_divide_by_zero_hazard() -> None:
    acceptance = resolve_constant_channel_acceptance(
        None, invoked=False, accept_flag=False, stdin_isatty=False, prompt=None
    )
    assert "min-max" in acceptance.training_risk
    assert "zero" in acceptance.training_risk.lower()


# ---------------------------------------------------------------------------
# _derive_exit_code precedence: 1 beats 5 beats 4 beats 0.
# ---------------------------------------------------------------------------


def _outcome(exit_code: int, *, fully_validated: bool = True) -> ValidationOutcome:
    return ValidationOutcome(
        episode_results={}, dataset_results=(), fully_validated=fully_validated, exit_code=exit_code
    )


def _acceptance(*, finding_present: bool, accepted: bool | None, mode: str) -> ConstantChannelAcceptance:
    return ConstantChannelAcceptance(
        finding_present=finding_present,
        accepted=accepted,
        mode=mode,
        channels=(),
        flag=CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
        training_risk="risk",
    )


def test_derive_exit_code_fatal_validation_outranks_unaccepted_finding() -> None:
    acceptance = _acceptance(finding_present=True, accepted=False, mode="unattended-no-flag")
    exit_code = _derive_exit_code((), _outcome(1, fully_validated=False), acceptance)
    assert exit_code == EXIT_VALIDATION_FAILED


def test_derive_exit_code_unaccepted_finding_outranks_completed_with_findings() -> None:
    acceptance = _acceptance(finding_present=True, accepted=False, mode="unattended-no-flag")
    exit_code = _derive_exit_code((), _outcome(0, fully_validated=False), acceptance)
    assert exit_code == EXIT_CONSTANT_CHANNELS_UNACCEPTED


def test_derive_exit_code_accepted_finding_is_completed_with_findings_never_clean() -> None:
    acceptance = _acceptance(finding_present=True, accepted=True, mode="flag")
    exit_code = _derive_exit_code((), _outcome(0, fully_validated=True), acceptance)
    assert exit_code == EXIT_COMPLETED_WITH_FINDINGS


def test_derive_exit_code_no_finding_and_fully_validated_is_clean() -> None:
    acceptance = _acceptance(finding_present=False, accepted=None, mode="no-finding")
    exit_code = _derive_exit_code((), _outcome(0, fully_validated=True), acceptance)
    assert exit_code == EXIT_CLEAN


# ---------------------------------------------------------------------------
# End-to-end: the report's constant_channel_acceptance block is present on
# every run, help text, and no config-mirroring flag was added.
# ---------------------------------------------------------------------------


def test_accept_constant_channels_flag_in_help() -> None:
    parser_help = main.__globals__["build_parser"]().format_help()
    assert CONSTANT_CHANNEL_ACCEPTANCE_FLAG in parser_help


def test_no_run_config_key_mirrors_the_acceptance_flag() -> None:
    config = RunConfig.model_fields
    assert not any("accept" in name and "constant" in name for name in config)


def test_report_carries_acceptance_block_on_clean_run_with_no_finding(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
        ]
    )

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = report["constant_channel_acceptance"]
    assert set(acceptance) == {
        "finding_present",
        "accepted",
        "mode",
        "channels",
        "flag",
        "training_risk",
    }
    assert acceptance["mode"] == "no-finding"
    assert acceptance["finding_present"] is False
    assert acceptance["accepted"] is None
    assert acceptance["channels"] == []


def test_report_carries_acceptance_block_on_no_validate_run(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
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
    assert exit_code == EXIT_CLEAN

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = report["constant_channel_acceptance"]
    assert acceptance["mode"] == "not-evaluated"
    assert acceptance["finding_present"] is False
    assert acceptance["accepted"] is None


def test_real_validate_dataset_finding_flows_to_unaccepted_exit_code(tmp_path: Path) -> None:
    """Proves the REAL `validate_dataset` check-result shape (not a
    hand-built `CheckResult`) flows through `constant_channel_finding` /
    `resolve_constant_channel_acceptance` / `_derive_exit_code` to
    `EXIT_CONSTANT_CHANNELS_UNACCEPTED`.

    A true through-`main()` end-to-end test of this path would need a
    SECOND `main()` invocation against a prefix `force_constant_channel`
    has mutated (validation only runs when re-invoked, and
    `force_constant_channel` needs an already-written prefix to mutate).
    That structural reason is unchanged and still correct, so this test
    still drives the real `validate_dataset` outcome through the three
    functions directly rather than adding a second `main()` call.

    What HAS changed since this docstring was first written: re-invoking
    `main()` a second time against a resumed prefix with validation
    enabled used to fail the persisted-histogram integrity check
    regardless of feature/channel choice (WINDOWS.md id 9) — a defect in
    the writer's resume/`finalize()` path, not in this test's own files.
    04-10 closed it at the source (`LeRobotV21Writer._rehydrate_from_disk`
    now rehydrates `_camera_video_info` from `meta/info.json`), and 04-12's
    `test_gap_two_validated_invocations_over_one_prefix_are_idempotent` and
    `test_gap_two_validated_invocations_over_a_standing_fatal_finding_resume_not_reject`
    are the through-`main()` coverage this docstring previously said did
    not exist.
    """
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
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
    assert exit_code == EXIT_CLEAN
    force_constant_channel(out_dir, feature="observation.state", channel_index=4)

    profile = load_profile(PROFILE_PATH)
    # Measured before asserting: no check drives validate_dataset's own
    # exit code non-zero (statistics.constant_channel is advisory as of
    # G-03 and therefore can't; regression.golden_fingerprint reports
    # `passed=False` too on this fixture, but carries the unevaluated
    # marker -- per registry.py's own severity semantics, an unevaluated
    # result never counts toward a REJECT/non-zero-exit-code either) --
    # and the constant-channel check specifically failed with the flagged
    # finding this test drives through the resolver.
    validation = validate_dataset(out_dir, profile)
    assert validation.exit_code == 0, [
        (r.check_name, r.evidence) for r in validation.dataset_results if not r.passed
    ]
    constant_result = next(
        r for r in validation.dataset_results if r.check_name == CONSTANT_CHANNEL_CHECK_NAME
    )
    assert constant_result.passed is False

    finding = constant_channel_finding(validation)
    assert finding is not None
    acceptance = resolve_constant_channel_acceptance(
        finding, invoked=True, accept_flag=False, stdin_isatty=False, prompt=None
    )
    assert acceptance.mode == "unattended-no-flag"
    assert acceptance.finding_present is True
    assert acceptance.accepted is False
    assert acceptance.channels == ("observation.state.arm_l_j5",)

    exit_code = _derive_exit_code((), validation, acceptance)
    assert exit_code == EXIT_CONSTANT_CHANNELS_UNACCEPTED


def test_unaccepted_finding_end_to_end_via_run_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the real `run_convert` wiring (not the pure functions in
    isolation) with `validate_dataset` monkeypatched to return a
    hand-built finding — isolating the wiring under test from every OTHER
    check a real `validate_dataset` call would also run (kinematics,
    consistency, the persisted-histogram integrity check, etc.), never
    because of an open defect. The resume/`finalize()` defect
    `test_real_validate_dataset_finding_flows_to_unaccepted_exit_code`
    above used to document (WINDOWS.md id 9) was closed by 04-10; this
    monkeypatch remains for check-isolation reasons only."""
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("observation.state", "arm_l_j5")]
    )
    fake_outcome = ValidationOutcome(
        episode_results={}, dataset_results=(finding,), fully_validated=True, exit_code=0
    )
    monkeypatch.setattr("data.cli.validate_dataset", lambda *a, **k: fake_outcome)

    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    profile = load_profile(PROFILE_PATH)
    run_cfg = load_run_config(_write_run_config(tmp_path, inputs=[str(source_mcap)]))

    outcome = run_convert(
        out_dir,
        profile=profile,
        run_cfg=run_cfg,
        cli_output_prefix=str(out_dir),
        no_validate=False,
        use_local_golden=False,
        stdin_isatty=lambda: False,
    )

    assert outcome.exit_code == EXIT_CONSTANT_CHANNELS_UNACCEPTED
    assert outcome.constant_channel_acceptance is not None
    assert outcome.constant_channel_acceptance.mode == "unattended-no-flag"
    assert outcome.constant_channel_acceptance.finding_present is True
    assert outcome.constant_channel_acceptance.accepted is False

    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert report["constant_channel_acceptance"]["mode"] == "unattended-no-flag"
    assert report["constant_channel_acceptance"]["channels"] == ["observation.state.arm_l_j5"]


def test_stdout_on_unaccepted_run_carries_only_report_path_and_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    finding = _constant_channel_result(
        passed=False, flagged_channels=[_flagged("observation.state", "arm_l_j5")]
    )
    fake_outcome = ValidationOutcome(
        episode_results={}, dataset_results=(finding,), fully_validated=True, exit_code=0
    )
    monkeypatch.setattr("data.cli.validate_dataset", lambda *a, **k: fake_outcome)

    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    capsys.readouterr()
    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
        ]
    )
    assert exit_code == EXIT_CONSTANT_CHANNELS_UNACCEPTED

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(stdout_lines) == 2
    assert stdout_lines[0].endswith(".json")
    assert re.fullmatch(r"rev-[0-9a-f]{8}", stdout_lines[1])
    assert "constant-channel" in captured.err
    assert CONSTANT_CHANNEL_ACCEPTANCE_FLAG in captured.err


# ---------------------------------------------------------------------------
# Task 3 (G-04): CLI-03's flag-override recording and CLI-06's revision
# stability -- both disclosed by 04-01-SUMMARY.md's coverage.D5 as verified
# only by an ad hoc script, never as a committed test. `--no-validate`
# throughout: these tests are about what the manifest records, not about
# validation, and staying independent of the golden-fingerprint abstention
# and Task 2's acceptance gate keeps the exit codes deterministic.
# ---------------------------------------------------------------------------


def test_override_recorded_when_positional_out_dir_differs_from_config(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    config_output_prefix = (tmp_path / "config_prefix").resolve()
    positional_out_dir = (tmp_path / "positional_prefix").resolve()
    config_path = _write_run_config(
        tmp_path, inputs=[str(source_mcap)], output_prefix=str(config_output_prefix)
    )

    exit_code = main(
        [
            str(positional_out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--no-validate",
        ]
    )
    assert exit_code == EXIT_CLEAN

    manifest = read_run_manifest(positional_out_dir)
    assert manifest is not None
    entry = manifest["runs"][-1]
    assert entry["overridden_config"] == {"output_prefix": str(config_output_prefix)}
    assert entry["effective_config"]["output_prefix"] == str(positional_out_dir)


def test_no_override_recorded_when_positional_matches_config_output_prefix(
    tmp_path: Path,
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    # `run_convert` resolves out_dir before comparing (`out_dir.resolve()`),
    # and on macOS a pytest tmp_path under a symlinked prefix (e.g.
    # /tmp -> /private/tmp) can resolve to a different string than the one
    # written into the config. Writing the config's output_prefix from the
    # ALREADY-RESOLVED path keeps this comparison textually exact, so the
    # test asserts the override rule itself, not a filesystem quirk.
    out_dir = (tmp_path / "output").resolve()
    config_path = _write_run_config(
        tmp_path, inputs=[str(source_mcap)], output_prefix=str(out_dir)
    )

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
    assert exit_code == EXIT_CLEAN

    manifest = read_run_manifest(out_dir)
    assert manifest is not None
    assert manifest["runs"][-1]["overridden_config"] == {}


def test_no_override_recorded_when_config_declares_no_output_prefix(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    # _write_run_config's own default is output_prefix: None -- not
    # overridden here, so the config genuinely declares no value.
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
    assert exit_code == EXIT_CLEAN

    manifest = read_run_manifest(out_dir)
    assert manifest is not None
    assert manifest["runs"][-1]["overridden_config"] == {}


def test_effective_config_comes_back_key_sorted_in_manifest(tmp_path: Path) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
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
    assert exit_code == EXIT_CLEAN

    # Read the raw file text's own key order (not a re-sorted Python dict)
    # -- every manifest write goes through json.dump(sort_keys=True), and
    # this asserts that convention actually reached this field.
    manifest_text = (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_text)
    effective_keys = list(manifest["runs"][-1]["effective_config"].keys())
    assert effective_keys == sorted(effective_keys)


def test_dataset_revision_full_unchanged_across_two_runs_with_identical_hash_set(
    tmp_path: Path,
) -> None:
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]

    assert main(args) == EXIT_CLEAN
    manifest_after_first = read_run_manifest(out_dir)
    assert manifest_after_first is not None
    assert len(manifest_after_first["runs"]) == 1
    first_full = manifest_after_first["dataset_revision"]["full"]

    # Identical source_mcap -> the SAME source_sha256 -> resumed_skip, so
    # this second run's converted source-hash SET is identical to the
    # first's even though it converts nothing new.
    assert main(args) == EXIT_CLEAN
    manifest_after_second = read_run_manifest(out_dir)
    assert manifest_after_second is not None
    assert len(manifest_after_second["runs"]) == 2
    second_full = manifest_after_second["dataset_revision"]["full"]

    # The FULL digest, never the short form -- the short form is a
    # truncation and could collide where the full hash would not.
    assert second_full == first_full


def test_all_rejected_batch_still_appends_manifest_entry_with_recomputable_revision(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    recordings = _build_n_recordings_in_one_dir(input_dir, 2)
    for rec in recordings:
        corrupt_chunk_crc(rec)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

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

    manifest = read_run_manifest(out_dir)
    assert manifest is not None
    assert len(manifest["runs"]) == 1
    assert len(manifest["runs"][0]["inputs"]) == 2
    assert all(e["bucket"] == "newly_rejected" for e in manifest["runs"][0]["inputs"])

    # CR-01 (04-VERIFICATION.md): a fully-rejected batch's revision is
    # derived from an EMPTY written-hash set — a rejected input never
    # contributed its source_sha256 to the digest, even though it still
    # produced a REJECT-disposition provenance record with that hash
    # recorded on disk. The property this test's NAME names (a fully-
    # rejected batch still appends a manifest entry with a recomputable
    # revision) is real and worth keeping; only the revision assertion
    # below was wrong before this fix — it used to assert the OPPOSITE
    # (both rejected inputs' hashes present and non-trivial).
    rejected_hashes = {e["source_sha256"] for e in manifest["runs"][0]["inputs"]}
    assert len(rejected_hashes) == 2  # two genuinely distinct rejected inputs

    inputs = manifest["dataset_revision"]["inputs"]
    assert inputs["source_sha256s"] == []
    for rejected_hash in rejected_hashes:
        assert rejected_hash not in inputs["source_sha256s"]

    _, recomputed_full = dataset_revision_id(
        inputs["source_sha256s"], inputs["profile_hash"], inputs["git_commit"], inputs["format_key"]
    )
    assert recomputed_full == manifest["dataset_revision"]["full"]


def test_gap_v1_record_without_ingest_outcome_resumes_from_artifact_presence(
    tmp_path: Path,
) -> None:
    """Reproduces the exact on-disk shape of an existing EFS prefix
    predating 04-09 (no `ingest_outcome` key at all, `record_schema_version
    == 1`) and proves it still resumes correctly: written-ness is derived
    from artifact presence at that episode index, never defaulted, and its
    hash is still counted toward the dataset revision."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])

    assert (
        main(
            [
                str(out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(config_path),
                "--no-validate",
            ]
        )
        == EXIT_CLEAN
    )

    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source_sha256 = record["source_sha256"]

    # Reproduce the exact v1 on-disk shape: no ingest_outcome key at all,
    # record_schema_version rolled back to 1. One json.load, a pop, an
    # assignment, one json.dump — inline here rather than a shared helper,
    # deliberately (this plan's own action), so no test exercising the v2
    # path is tempted to reuse a v1-shape helper.
    v1_payload = dict(record)
    v1_payload.pop("ingest_outcome")
    v1_payload["record_schema_version"] = 1
    record_path.write_text(json.dumps(v1_payload), encoding="utf-8")

    second_exit = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--no-validate",
        ]
    )
    assert second_exit == EXIT_CLEAN

    second_report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_report["ledger"][0]["bucket"] == "resumed_skip"

    manifest = read_run_manifest(out_dir)
    assert manifest is not None
    assert source_sha256 in manifest["dataset_revision"]["inputs"]["source_sha256s"]

    # No duplicate episode index consumed.
    data_engine_dir = out_dir / "meta" / "data_engine"
    episode_records = list(data_engine_dir.glob("episode_*.json"))
    assert len(episode_records) == 1


def test_gap_revision_digest_is_order_independent_over_written_hashes(tmp_path: Path) -> None:
    """Two prefixes built from the same two recordings supplied in opposite
    `inputs` order produce an identical `dataset_revision.full`."""
    rec_a = build_fixture_source_mcap(tmp_path / "rec_a", camera_frames=6)
    rec_b = build_fixture_source_mcap(tmp_path / "rec_b", camera_frames=8)

    forward_out_dir = tmp_path / "forward"
    forward_cfg_dir = tmp_path / "forward_cfg"
    forward_cfg_dir.mkdir()
    forward_config = _write_run_config(forward_cfg_dir, inputs=[str(rec_a), str(rec_b)])
    assert (
        main(
            [
                str(forward_out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(forward_config),
                "--no-validate",
            ]
        )
        == EXIT_CLEAN
    )

    backward_out_dir = tmp_path / "backward"
    backward_cfg_dir = tmp_path / "backward_cfg"
    backward_cfg_dir.mkdir()
    backward_config = _write_run_config(backward_cfg_dir, inputs=[str(rec_b), str(rec_a)])
    assert (
        main(
            [
                str(backward_out_dir),
                "--profile",
                str(PROFILE_PATH),
                "--run-config",
                str(backward_config),
                "--no-validate",
            ]
        )
        == EXIT_CLEAN
    )

    forward_manifest = read_run_manifest(forward_out_dir)
    backward_manifest = read_run_manifest(backward_out_dir)
    assert forward_manifest is not None
    assert backward_manifest is not None
    assert (
        forward_manifest["dataset_revision"]["full"]
        == backward_manifest["dataset_revision"]["full"]
    )
    assert (
        sorted(forward_manifest["dataset_revision"]["inputs"]["source_sha256s"])
        == sorted(backward_manifest["dataset_revision"]["inputs"]["source_sha256s"])
    )


def test_gap_retry_rejected_success_supersedes_stale_reject_record(tmp_path: Path) -> None:
    """WR-05: a successful `--retry-rejected` conversion supersedes the
    stale rejection record rather than leaving two live records for the
    same input.

    A genuinely NEW episode index for the retried input requires the
    writer's own real-write counter to already be positioned ABOVE this
    input's original reject slot at retry time — which only happens when
    some OTHER index is still an unresolved outstanding reject sitting
    BELOW this input's slot (the writer always fills the lowest open real
    slot next, regardless of which input is being processed). A synthetic
    placeholder record for that other, permanently-unresolved input is
    planted directly via `ProvenanceRecorder` (never through `main()`), so
    this test needs no second real MCAP fixture; its own fate (overwritten
    once this input's retry claims that slot) is not this test's concern.

    The input under test is a genuinely VALID recording, never corrupted —
    `--retry-rejected` recognizes a retry as the SAME input by content hash
    (`rejected_hash_to_index` is keyed by `source_sha256`), and any repair
    that changes a file's bytes back to valid necessarily changes its hash
    too (sha256 is content-derived), which would make the retry look like
    a brand-new, never-before-seen input instead. Reproducing "the ingest
    check that rejected it no longer does" therefore requires the file's
    OWN reject record to be planted directly (as a stand-in for whatever
    transient condition made the ORIGINAL parse fail) rather than induced
    via a corrupt-then-repair round trip on the real file.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    recorder = ProvenanceRecorder(out_dir, PROFILE)

    placeholder_source = tmp_path / "placeholder_other_input.bin"
    placeholder_source.write_bytes(b"a different, unrelated rejected input")
    placeholder_header = EpisodeHeader(
        episode_id=str(placeholder_source),
        fps=25.0,
        robot_type="r1pro",
        task_prompt=None,
        frame_count=None,
        global_meta={},
    )
    recorder.record_rejected(placeholder_header, str(placeholder_source), 0)

    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    reject_header = EpisodeHeader(
        episode_id=str(source_mcap),
        fps=25.0,
        robot_type="r1pro",
        task_prompt=None,
        frame_count=None,
        global_meta={},
    )
    original_index = 1  # index 0 is the synthetic placeholder above.
    recorder.record_rejected(reject_header, str(source_mcap), original_index)

    record_path = out_dir / "meta" / "data_engine" / f"episode_{original_index:06d}.json"
    original_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert original_payload["ingest_outcome"] == "REJECTED_BEFORE_WRITE"
    assert original_payload["superseded_by_episode_index"] is None

    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    exit_code = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--no-validate",
            "--retry-rejected",
        ]
    )
    assert exit_code == EXIT_CLEAN

    new_index = 0  # the writer's own real-write counter, still at 0.
    new_record_path = out_dir / "meta" / "data_engine" / f"episode_{new_index:06d}.json"
    new_payload = json.loads(new_record_path.read_text(encoding="utf-8"))
    assert new_payload["ingest_outcome"] == "WRITTEN"
    assert new_payload["superseded_by_episode_index"] is None

    original_after = json.loads(record_path.read_text(encoding="utf-8"))
    assert original_after["ingest_outcome"] == "REJECTED_BEFORE_WRITE"
    assert original_after["superseded_by_episode_index"] == new_index

    manifest = read_run_manifest(out_dir)
    assert manifest is not None
    source_sha256s = manifest["dataset_revision"]["inputs"]["source_sha256s"]
    assert source_sha256s.count(new_payload["source_sha256"]) == 1

    converted_map, rejected_map = _build_resume_maps(
        out_dir, emitted_video_keys=_EMITTED_VIDEO_KEYS
    )
    assert converted_map[new_payload["source_sha256"]] == new_index
    assert new_payload["source_sha256"] not in rejected_map

    # A SECOND main() run over the same prefix, no flags: exactly one
    # resumed_skip entry for this input, at the LIVE index — never the
    # superseded one, never two entries. This is the behavioral proof that
    # the absent-from-the-written-map state resolves correctly end to end,
    # not merely at the map level.
    second_exit = main(
        [
            str(out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path),
            "--no-validate",
        ]
    )
    assert second_exit == EXIT_CLEAN
    second_report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [e["bucket"] for e in second_report["ledger"]] == ["resumed_skip"]
    assert second_report["ledger"][0]["episode_index"] == new_index


def test_gap_unexpected_write_episode_failure_degrades_to_newly_rejected_with_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WR-02: an unexpected exception out of `write_episode` (an encoder
    fault, a corrupt JPEG, an unsupported frame shape) costs exactly the
    ONE input it happened on — the batch's run report is still written,
    and the other, unrelated input still converts normally."""
    from data.writers.lerobot_v21 import LeRobotV21Writer

    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 2)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    real_write_episode = LeRobotV21Writer.write_episode
    call_count = {"n": 0}

    def _fail_on_first_call(self: LeRobotV21Writer, episode: object) -> object:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated encoder fault")
        return real_write_episode(self, episode)  # type: ignore[arg-type]

    monkeypatch.setattr(LeRobotV21Writer, "write_episode", _fail_on_first_call)

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

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["ledger"]) == 2
    failing_entry, converted_entry = report["ledger"]

    assert failing_entry["bucket"] == "newly_rejected"
    assert failing_entry["episode_index"] is None
    assert failing_entry["frame_count"] is None
    assert len(failing_entry["checks"]) == 1
    check = failing_entry["checks"][0]
    assert check["check_name"] == WRITE_UNEXPECTED_FAILURE_CHECK_NAME
    assert check["severity"] == "fatal"
    assert check["passed"] is False
    assert check["evidence"]["exception_type"] == "RuntimeError"
    assert "simulated encoder fault" in check["evidence"]["exception_message"]

    assert converted_entry["bucket"] == "converted"
    assert converted_entry["episode_index"] == 0

    # No provenance record was written for the failing input — only the
    # converted one landed a record at all.
    data_engine_dir = out_dir / "meta" / "data_engine"
    episode_records = list(data_engine_dir.glob("episode_*.json"))
    assert len(episode_records) == 1


def test_gap_stats_write_failure_after_parquet_commit_does_not_duplicate_episode_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR-01 (04-REVIEW.md): a fault strictly AFTER `write_episode`'s parquet
    `os.replace` commit — here, `stats.write_partial_file` patched to raise
    on its first call, simulating a transient EFS I/O fault — must not leave
    a duplicate `episode_index` row permanently baked into `episodes.jsonl`
    when the WR-02 guard degrades the failure to one rejected input and the
    batch's next input reuses the same writer index. Reproduces the
    reviewer's live repro against a 2-input batch: before the CR-01 fix to
    `LeRobotV21Writer.write_episode` (rolling back its own `_tasks_meta`/
    `_episodes_meta` appends on a post-commit failure), this produced TWO
    `episodes.jsonl` rows at `episode_index: 0`, only one backed by a real
    parquet file, with `meta/info.json`'s `total_episodes` claiming 2."""
    from data.writers import stats

    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 2)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")

    real_write_partial_file = stats.write_partial_file
    call_count = {"n": 0}

    def _fail_on_first_call(*args: object, **kwargs: object) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated transient EFS I/O fault")
        real_write_partial_file(*args, **kwargs)

    monkeypatch.setattr(stats, "write_partial_file", _fail_on_first_call)

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

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["ledger"]) == 2
    failing_entry, converted_entry = report["ledger"]

    assert failing_entry["bucket"] == "newly_rejected"
    assert failing_entry["episode_index"] is None
    check = failing_entry["checks"][0]
    assert check["check_name"] == WRITE_UNEXPECTED_FAILURE_CHECK_NAME
    assert check["evidence"]["exception_type"] == "OSError"

    assert converted_entry["bucket"] == "converted"
    # The failed call's `_episodes_meta`/`_tasks_meta` appends must have been
    # rolled back, so the writer's counter never advanced past 0 and this
    # second, successful call reuses the SAME index.
    assert converted_entry["episode_index"] == 0

    episodes_rows = [
        json.loads(line)
        for line in (out_dir / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    # The core CR-01 assertion: exactly ONE row, and its index is unique.
    assert len(episodes_rows) == 1
    episode_indices = [row["episode_index"] for row in episodes_rows]
    assert len(set(episode_indices)) == len(episode_indices)

    import pyarrow.parquet as pq

    parquet_files = sorted((out_dir / "data" / "chunk-000").glob("episode_*.parquet"))
    assert len(parquet_files) == 1
    surviving_index = episode_indices[0]
    assert parquet_files[0].name == f"episode_{surviving_index:06d}.parquet"
    table = pq.read_table(parquet_files[0])
    assert table.num_rows == episodes_rows[0]["length"]

    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == len(parquet_files) == 1

    # Exactly one stats-partial file exists, for the surviving episode only
    # — the failed first call never got far enough to create one (the
    # patched `write_partial_file` raises before its own tmp-file write),
    # so there is nothing orphaned to have rolled back here, but this
    # confirms the directory still ends up in the expected one-file steady
    # state rather than accumulating debris from the failed attempt.
    stats_partials_dir = out_dir / "meta" / "stats_partials"
    stats_partial_files = list(stats_partials_dir.glob("*.json"))
    assert [p.name for p in stats_partial_files] == [f"episode_{surviving_index:06d}.json"]


def test_gap_orphaned_manifest_report_path_is_detected_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-01: a manifest `runs` entry whose recorded `report_path` no
    longer exists on disk (simulating the crash window between
    `append_run_to_manifest` and `write_run_report`) is detected and
    named — on stderr and in the next run's own report — rather than
    silently skipped forever. The run that detects it still completes and
    still writes its own report at the next ordinal."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    base_args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]

    assert main(base_args) == EXIT_CLEAN
    first_report_path = out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json"
    assert first_report_path.exists()
    first_report_path.unlink()

    capsys.readouterr()
    second_exit = main(base_args)
    assert second_exit == EXIT_CLEAN
    captured = capsys.readouterr()
    assert str(first_report_path) in captured.err

    second_report_path = out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json"
    assert second_report_path.exists()
    second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
    assert second_report["orphaned_report_paths"] == [str(first_report_path)]


def test_gap_write_temp_debris_swept_on_resume_and_named_in_report(tmp_path: Path) -> None:
    """WR-04: dot-prefixed write-in-progress `.tmp` debris from an
    interrupted encode or parquet/record write is swept once per RESUME
    and every deleted path is named in the run report — while the
    dataset-wide `.info.json.tmp` metadata temp (a negative control for a
    temp this sweep must NOT touch) survives untouched."""
    input_dir = tmp_path / "inputs"
    _build_n_recordings_in_one_dir(input_dir, 1)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, input_dir=str(input_dir), input_glob="*.mcap")
    args = [
        str(out_dir),
        "--profile",
        str(PROFILE_PATH),
        "--run-config",
        str(config_path),
        "--no-validate",
    ]
    assert main(args) == EXIT_CLEAN

    debris_index = 9
    video_key = _EMITTED_VIDEO_KEYS[0]
    video_tmp_path = (
        out_dir / "videos" / "chunk-000" / video_key / f".episode_{debris_index:06d}.mp4.tmp"
    )
    video_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    video_tmp_path.write_bytes(b"partial mp4")

    parquet_tmp_path = out_dir / "data" / "chunk-000" / f".episode_{debris_index:06d}.parquet.tmp"
    parquet_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp_path.write_bytes(b"partial parquet")

    manifest_tmp_path = (
        out_dir / "meta" / "encode_manifests" / f".episode_{debris_index:06d}.json.tmp"
    )
    manifest_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp_path.write_text("{}", encoding="utf-8")

    record_tmp_path = out_dir / "meta" / "data_engine" / f".episode_{debris_index:06d}.json.tmp"
    record_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    record_tmp_path.write_text("{}", encoding="utf-8")

    # Negative control: a dataset-wide metadata temp this sweep must NEVER
    # touch, per its own docstring's exclusion list — planted in the SAME
    # directory the sweep DOES clean (meta/data_engine/), to prove the
    # exclusion is the `episode_*` glob pattern itself, not merely "wrong
    # directory". `--no-validate` (below) means validate_dataset never
    # runs this run, so nothing else in this run's own write path renames
    # or touches this file either — it is a true negative control, not an
    # artifact of unrelated finalize()/manifest bookkeeping that would
    # rename it away regardless of the sweep.
    validation_report_tmp_path = (
        out_dir / "meta" / "data_engine" / ".validation_report.json.tmp"
    )
    validation_report_tmp_path.write_text("{}", encoding="utf-8")

    before = _file_snapshot_excluding_run_metadata(out_dir)

    exit_code = main(args)
    assert exit_code == EXIT_CLEAN

    assert not video_tmp_path.exists()
    assert not parquet_tmp_path.exists()
    assert not manifest_tmp_path.exists()
    assert not record_tmp_path.exists()
    assert validation_report_tmp_path.exists()

    after = _file_snapshot_excluding_run_metadata(out_dir)
    # Every completed episode's parquet/MP4 stays byte-identical (same
    # size + mtime); the four swept temps are gone, so after is a strict
    # subset of before.
    assert after < before

    report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(report["swept_temp_paths"]) == {
        str(video_tmp_path),
        str(parquet_tmp_path),
        str(manifest_tmp_path),
        str(record_tmp_path),
    }
    assert len(report["swept_temp_paths"]) == 4


def test_gap_prefix_basename_revision_divergence_reported_without_changing_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WINDOWS.md id 12 (narrowed): a prefix directory whose basename looks
    like a `rev-` revision token but disagrees with the revision this run
    actually computed is reported on stderr and in the run report, but
    never changes the exit code and never refuses the run — proven by
    comparing exit codes against the identical conversion into a
    differently-named prefix, never by asserting a hardcoded exit-code
    literal. A basename that does not even look like a `rev-` token
    records `prefix_name_matches_revision` as `null`, not `false`."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)

    mismatched_out_dir = tmp_path / "rev-deadbeef"
    cfg_dir_1 = tmp_path / "cfg1"
    cfg_dir_1.mkdir()
    config_path_1 = _write_run_config(cfg_dir_1, inputs=[str(source_mcap)])
    capsys.readouterr()
    exit_code_mismatched = main(
        [
            str(mismatched_out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path_1),
        ]
    )
    captured = capsys.readouterr()
    assert "rev-deadbeef" in captured.err

    report_mismatched = json.loads(
        (mismatched_out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert report_mismatched["prefix_name_matches_revision"] is False
    assert report_mismatched["dataset_revision"]["short"] in captured.err

    plain_out_dir = tmp_path / "plain_output"
    cfg_dir_2 = tmp_path / "cfg2"
    cfg_dir_2.mkdir()
    config_path_2 = _write_run_config(cfg_dir_2, inputs=[str(source_mcap)])
    exit_code_plain = main(
        [
            str(plain_out_dir),
            "--profile",
            str(PROFILE_PATH),
            "--run-config",
            str(config_path_2),
        ]
    )
    assert exit_code_mismatched == exit_code_plain

    report_plain = json.loads(
        (plain_out_dir / "meta" / "data_engine" / "run_reports" / "run_000.json").read_text(
            encoding="utf-8"
        )
    )
    assert report_plain["prefix_name_matches_revision"] is None


# ---------------------------------------------------------------------------
# 04-12 gap closure (terminal plan of the batch): the two real,
# validation-ENABLED `main()` invocations 04-VERIFICATION.md named as a
# required deliverable (Gap "CLI-02", `missing` list), plus the composition
# proof over the standing-fatal-finding shape every real dataset this
# project has produced actually has. Test A closes WINDOWS.md id 9's
# through-`main()` regression coverage; Test B proves 04-09's written-ness
# fix and 04-10's rehydration fix compose correctly under validation.
# ---------------------------------------------------------------------------


def test_gap_two_validated_invocations_over_one_prefix_are_idempotent(tmp_path: Path) -> None:
    """The end-to-end proof 04-VERIFICATION.md's CLI-02 gap named as
    required: `main()` run TWICE with validation ENABLED against the same
    prefix must agree with itself on every measured/recorded quantity, not
    merely exit 0.

    `camera_frames=6`: at this row count `consistency.action_state_lag`
    abstains (`max_lag = min(MAX_LAG_SEARCH, row_count - 3)` — see the
    04-09 gap section's comment above), so no fatal check fails. The
    shipped Galaxea profile declares no physics limits, so
    `kinematics.velocity_acceleration_bounds` is always unevaluated and
    `ValidationOutcome.fully_validated` is always `False` for a real
    conversion under it — `EXIT_COMPLETED_WITH_FINDINGS`, never
    `EXIT_CLEAN`, is the correct outcome here, exactly as
    `test_tracer_one_recording_end_to_end` already established for a
    single validated run at this same `camera_frames` value
    (04-01-SUMMARY.md recorded that an earlier `EXIT_CLEAN` expectation
    was wrong)."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=6)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    args = [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]

    first_exit = main(args)
    assert first_exit != EXIT_CLEAN

    first_validation_report = json.loads(
        (out_dir / "meta" / "data_engine" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    first_info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    # `meta/data_engine/episode_` excluded: `amend_checks` legitimately
    # APPENDS to the episode's `checks` list on every validated run
    # (pre-existing D-02/D-03 behavior, invisible to every resume decision
    # per this helper's own docstring) — unrelated to what this snapshot
    # comparison exists to catch.
    before_second_run = _file_snapshot_excluding_run_metadata(
        out_dir, extra_excluded_prefixes=("meta/data_engine/episode_",)
    )

    second_exit = main(args)
    assert second_exit == first_exit

    second_validation_report = json.loads(
        (out_dir / "meta" / "data_engine" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    second_info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    after_second_run = _file_snapshot_excluding_run_metadata(
        out_dir, extra_excluded_prefixes=("meta/data_engine/episode_",)
    )

    for validation_report in (first_validation_report, second_validation_report):
        histogram_result = next(
            r
            for r in validation_report["dataset_results"]
            if r["check_name"] == HISTOGRAM_CHECK_NAME
        )
        assert histogram_result["passed"] is True
        for finding in histogram_result["evidence"].get("findings", []):
            assert finding.get("invariant") != "mass_identity"
            assert finding.get("expected_mass") != 0

    for video_key in _EMITTED_VIDEO_KEYS:
        first_video_info = first_info["features"][video_key]["video_info"]
        second_video_info = second_info["features"][video_key]["video_info"]
        assert second_video_info["video.height"] == first_video_info["video.height"]
        assert second_video_info["video.width"] == first_video_info["video.width"]
        assert first_video_info["video.height"] > 0
        assert first_video_info["video.width"] > 0

    assert second_info["total_episodes"] == first_info["total_episodes"]
    assert second_info["total_frames"] == first_info["total_frames"]

    second_run_report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["bucket"] for entry in second_run_report["ledger"]] == ["resumed_skip"]

    assert after_second_run == before_second_run


def test_gap_two_validated_invocations_over_a_standing_fatal_finding_resume_not_reject(
    tmp_path: Path,
) -> None:
    """The composition proof: `camera_frames=12` reproduces the state of
    every real Galaxea dataset this project has produced (see the 04-09
    gap section's comment above) — a written episode whose on-disk
    `disposition` is amended to `REJECT` by a standing, MEASURED (not
    abstained) fatal `consistency.action_state_lag` finding. A second
    validated `main()` invocation against the same prefix must still
    resume it as `resumed_skip`, never `rejected_skip` — proving 04-09's
    written-ness fix and 04-10's rehydration fix compose under validation,
    not merely in isolation."""
    source_mcap = build_fixture_source_mcap(tmp_path / "sources", camera_frames=12)
    out_dir = tmp_path / "output"
    config_path = _write_run_config(tmp_path, inputs=[str(source_mcap)])
    args = [str(out_dir), "--profile", str(PROFILE_PATH), "--run-config", str(config_path)]

    first_exit = main(args)
    assert first_exit == EXIT_VALIDATION_FAILED

    first_validation_report = json.loads(
        (out_dir / "meta" / "data_engine" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    lag_result = next(
        r
        for r in first_validation_report["episode_results"]["0"]
        if r["check_name"] == LAG_CHECK_NAME
    )
    # This test's own precondition, asserted rather than assumed: a
    # genuinely MEASURED fatal finding, never an abstention — without this,
    # the test could pass vacuously against an abstaining check.
    assert lag_result["severity"] == "fatal"
    assert lag_result["passed"] is False
    assert not lag_result["evidence"].get(UNEVALUATED_KEY, False)

    record_path = out_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    # The exact contradiction that broke all four consumers before 04-09.
    assert record["disposition"] == "REJECT"
    assert record["ingest_outcome"] == IngestOutcome.WRITTEN.value

    first_manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    first_revision_full = first_manifest["dataset_revision"]["full"]

    second_exit = main(args)
    assert second_exit == first_exit

    second_report = json.loads(
        (out_dir / "meta" / "data_engine" / "run_reports" / "run_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["bucket"] for entry in second_report["ledger"]] == ["resumed_skip"]

    second_validation_report = json.loads(
        (out_dir / "meta" / "data_engine" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    histogram_result = next(
        r
        for r in second_validation_report["dataset_results"]
        if r["check_name"] == HISTOGRAM_CHECK_NAME
    )
    assert histogram_result["passed"] is True

    second_manifest = json.loads(
        (out_dir / "meta" / "data_engine" / "run_manifest.json").read_text(encoding="utf-8")
    )
    # The written-hash set did not change (no new write, no new rejection),
    # so the composite digest must not move either.
    assert second_manifest["dataset_revision"]["full"] == first_revision_full
