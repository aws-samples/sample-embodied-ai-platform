"""Tests for `data/provenance/recorder.py` — `ProvenanceRecorder`'s per-episode
provenance record, exercised entirely from synthetic `EpisodeHeader`/
`EpisodeWriteResult` containers, with no adapter and no real recording
present (PROV-01/PROV-02/PROV-03/PROV-04).
"""

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from data.contracts.containers import (
    RECORD_SCHEMA_VERSION,
    CheckResult,
    ClockReport,
    Disposition,
    EpisodeHeader,
    EpisodeRecord,
    EpisodeWriteResult,
    IngestOutcome,
    StreamInfo,
)
from data.contracts.profile import load_profile, profile_hash
from data.adapters.galaxea import GalaxeaAdapterError, GalaxeaInputAdapter
from data.provenance.recorder import (
    GIT_COMMIT_UNAVAILABLE_MARKER,
    PIPELINE_VERSION,
    ProvenanceRecorder,
    _serialize_record,
)
from data.writers.lerobot_v21 import LeRobotV21Writer
from data.tests.frame_fixtures import build_synthetic_episode
import data.tests.mcap_fixture as fixture_mod

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"


@pytest.fixture()
def profile():
    return load_profile(PROFILE_PATH)


def _source_file(tmp_path: Path, content: bytes = b"synthetic mcap bytes") -> Path:
    path = tmp_path / "source.mcap"
    path.write_bytes(content)
    return path


def _stream_inventory() -> tuple[StreamInfo, ...]:
    return (
        StreamInfo("/hdas/feedback_arm_left", "sensor_msgs/msg/JointState", 100, 30.0, True),
        StreamInfo("/vr_pose", "teleoperation_msg_ros2/msg/VrPose", 0, None, False),
    )


def _checks() -> tuple[CheckResult, ...]:
    return (
        CheckResult("metadata.mcap_crc", "fatal", True, {}),
        CheckResult(
            "kinematics.velocity_acceleration_bounds",
            "advisory",
            False,
            {"per_field": {"left_arm": {"measured_max_velocity": 1.0}}},
        ),
    )


def _clock_report() -> ClockReport:
    return ClockReport(
        grid_hz=25.0,
        max_skew_ns=1_000_000,
        mean_skew_ns=200_000,
        per_field_source_age_ns={"state.left_arm": 100, "action.left_arm": 50},
        resolved_action_source={"left_arm": "/motion_control/control_arm_left"},
    )


def _make_header(
    tmp_path: Path,
    *,
    episode_id: str | None = None,
    stream_inventory: tuple[StreamInfo, ...] | None = None,
    checks: tuple[CheckResult, ...] | None = None,
    clock_report: ClockReport | None = None,
    disposition: Disposition = Disposition.PASS,
) -> EpisodeHeader:
    source_path = str(_source_file(tmp_path)) if episode_id is None else episode_id
    global_meta: dict[str, object] = {
        "stream_inventory": stream_inventory if stream_inventory is not None else _stream_inventory(),
        "checks": checks if checks is not None else _checks(),
        "clock_report": clock_report if clock_report is not None else _clock_report(),
        "disposition": disposition,
    }
    return EpisodeHeader(
        episode_id=source_path,
        fps=25.0,
        robot_type="r1pro",
        task_prompt="t",
        frame_count=10,
        global_meta=global_meta,
    )


def _make_write_result(episode_index: int = 0) -> EpisodeWriteResult:
    return EpisodeWriteResult(episode_index=episode_index, frame_count=10, output_paths={})


# ---------------------------------------------------------------------------
# Test 1: emitted path, six-digit zero padding.
# ---------------------------------------------------------------------------


def test_record_writes_zero_padded_episode_path(tmp_path: Path, profile) -> None:
    header = _make_header(tmp_path)
    write_result = _make_write_result(episode_index=7)
    recorder = ProvenanceRecorder(tmp_path, profile)

    path = recorder.record(header, write_result)

    assert path == tmp_path / "meta" / "data_engine" / "episode_000007.json"
    assert path.exists()


# ---------------------------------------------------------------------------
# Test 2: record_schema_version pinned to the contract's value.
# ---------------------------------------------------------------------------


def test_record_schema_version_matches_contract(tmp_path: Path, profile) -> None:
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())
    assert payload["record_schema_version"] == RECORD_SCHEMA_VERSION
    assert isinstance(payload["record_schema_version"], int)


# ---------------------------------------------------------------------------
# Test 3: source_uri / source_sha256.
# ---------------------------------------------------------------------------


def test_source_uri_and_sha256(tmp_path: Path, profile) -> None:
    source = _source_file(tmp_path, content=b"hello world")
    header = _make_header(tmp_path, episode_id=str(source))
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    assert payload["source_uri"] == str(source)
    import hashlib

    assert payload["source_sha256"] == hashlib.sha256(b"hello world").hexdigest()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_source = _source_file(other_dir, content=b"different bytes")
    header2 = _make_header(tmp_path, episode_id=str(other_source))
    path2 = recorder.record(header2, _make_write_result(episode_index=1))
    payload2 = json.loads(path2.read_text())
    assert payload2["source_sha256"] != payload["source_sha256"]


# ---------------------------------------------------------------------------
# Test 4: pipeline_version / profile_hash / git_commit.
# ---------------------------------------------------------------------------


def test_pipeline_version_profile_hash_git_commit_present(tmp_path: Path, profile) -> None:
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    assert payload["pipeline_version"]
    assert payload["profile_hash"] == profile_hash(profile)
    assert payload["git_commit"]
    assert payload["git_commit"] != ""


def test_git_commit_unavailable_marker_when_not_git_checkout(
    tmp_path: Path, profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())
    assert payload["git_commit"] == GIT_COMMIT_UNAVAILABLE_MARKER


# ---------------------------------------------------------------------------
# Test 5: stream inventory order/unemitted flag preserved.
# ---------------------------------------------------------------------------


def test_stream_inventory_order_and_unemitted_flag_preserved(tmp_path: Path, profile) -> None:
    inventory = _stream_inventory()
    header = _make_header(tmp_path, stream_inventory=inventory)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    serialized = payload["stream_inventory"]
    assert [row["topic"] for row in serialized] == [s.topic for s in inventory]
    assert serialized[0]["emitted"] is True
    assert serialized[1]["emitted"] is False


# ---------------------------------------------------------------------------
# Test 6: clock report round-trips fully, sorted keys.
# ---------------------------------------------------------------------------


def test_clock_report_round_trips(tmp_path: Path, profile) -> None:
    report = _clock_report()
    header = _make_header(tmp_path, clock_report=report)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    cr = payload["clock_report"]
    assert cr["grid_hz"] == report.grid_hz
    assert cr["max_skew_ns"] == report.max_skew_ns
    assert cr["mean_skew_ns"] == report.mean_skew_ns
    assert cr["per_field_source_age_ns"] == dict(report.per_field_source_age_ns)
    assert cr["resolved_action_source"] == dict(report.resolved_action_source)
    assert list(cr.keys()) == sorted(cr.keys())
    assert list(cr["per_field_source_age_ns"].keys()) == sorted(
        cr["per_field_source_age_ns"].keys()
    )


# ---------------------------------------------------------------------------
# Test 7: check list preserves registry order, full evidence.
# ---------------------------------------------------------------------------


def test_check_list_preserves_order_with_full_evidence(tmp_path: Path, profile) -> None:
    checks = _checks()
    header = _make_header(tmp_path, checks=checks)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    serialized = payload["checks"]
    assert [c["check_name"] for c in serialized] == [c.check_name for c in checks]
    for expected, actual in zip(checks, serialized):
        assert actual["severity"] == expected.severity
        assert actual["passed"] == expected.passed
        assert actual["evidence"] == dict(expected.evidence)


# ---------------------------------------------------------------------------
# Test 8: byte-identical across two calls over identical inputs.
# ---------------------------------------------------------------------------


def test_two_record_calls_over_identical_inputs_are_byte_identical(
    tmp_path: Path, profile
) -> None:
    source = _source_file(tmp_path)
    header = _make_header(tmp_path, episode_id=str(source))
    recorder = ProvenanceRecorder(tmp_path, profile)

    path1 = recorder.record(header, _make_write_result(episode_index=0))
    bytes1 = path1.read_bytes()

    other_dir = tmp_path / "run2"
    other_dir.mkdir()
    recorder2 = ProvenanceRecorder(other_dir, profile)
    path2 = recorder2.record(header, _make_write_result(episode_index=0))
    bytes2 = path2.read_bytes()

    assert bytes1 == bytes2


# ---------------------------------------------------------------------------
# Test 9: no parquet/video import, no file-format access.
# ---------------------------------------------------------------------------


def test_recorder_module_imports_no_parquet_or_video_library() -> None:
    import data.provenance.recorder as recorder_module

    source = recorder_module.__file__
    assert source is not None
    with open(source) as f:
        text = f.read()
    assert "import pyarrow" not in text
    assert "import av" not in text


# ---------------------------------------------------------------------------
# Test 10: meta/data_engine/ subdirectory is inert to the 4 LeRobot metadata
# files.
# ---------------------------------------------------------------------------


def test_data_engine_subdirectory_does_not_break_lerobot_metadata_parsing(
    tmp_path: Path, profile
) -> None:
    episode = build_synthetic_episode(frame_count=2, fps=10.0, profile=profile, task_prompt="t")
    writer = LeRobotV21Writer(tmp_path, profile, stdin_isatty=lambda: False)
    result = writer.write_episode(episode)
    writer.finalize()

    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    recorder.record(header, result)

    meta_dir = tmp_path / "meta"
    assert (meta_dir / "data_engine").is_dir()

    with open(meta_dir / "info.json") as f:
        json.load(f)
    with open(meta_dir / "tasks.jsonl") as f:
        for line in f:
            if line.strip():
                json.loads(line)
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            if line.strip():
                json.loads(line)
    with open(meta_dir / "stats.json") as f:
        json.load(f)


# ---------------------------------------------------------------------------
# Test 11: exercised entirely from synthetic containers — this whole module
# never imports data.adapters, and no test above touches a real recording.
# ---------------------------------------------------------------------------


def test_module_never_imports_data_adapters() -> None:
    import ast

    import data.provenance.recorder as recorder_module

    source = recorder_module.__file__
    assert source is not None
    with open(source) as f:
        tree = ast.parse(f.read(), filename=source)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(name.startswith("data.adapters") for name in imported_modules)


# ---------------------------------------------------------------------------
# Task 3 — PROV-04's three separate axes, the reserved fields, and the
# record a fatal abort still emits.
# ---------------------------------------------------------------------------


def _make_full_episode_record(
    *,
    outcome: str = "unknown",
    disposition: Disposition = Disposition.PASS,
    outcome_labeled_by: str = "unset",
    failure_modes: tuple[str, ...] = (),
) -> EpisodeRecord:
    """Builds an `EpisodeRecord` directly (bypassing `ProvenanceRecorder`,
    which never sets `outcome`) so the serializer itself can be tested
    against a record whose outcome/disposition are set independently."""
    return EpisodeRecord(
        RECORD_SCHEMA_VERSION,
        "e",
        "u",
        "h" * 64,
        PIPELINE_VERSION,
        "p",
        "g",
        (),
        _clock_report(),
        (),
        disposition,
        outcome,  # type: ignore[arg-type]
        failure_modes,
        outcome_labeled_by,  # type: ignore[arg-type]
    )


def test_outcome_failure_modes_labeled_by_disposition_are_four_distinct_keys(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path, disposition=Disposition.REJECT)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    assert {"outcome", "failure_modes", "outcome_labeled_by", "disposition"} <= set(
        payload.keys()
    )
    # A REJECT-disposition record still makes no claim about task success —
    # the unknown outcome, empty failure-mode list, and unset labeller are
    # untouched by `disposition`.
    assert payload["outcome"] == "unknown"
    assert payload["failure_modes"] == []
    assert payload["outcome_labeled_by"] == "unset"
    assert payload["disposition"] == "REJECT"


def test_failure_outcome_and_pass_disposition_serialize_faithfully() -> None:
    """A failed demonstration can be technically perfect and deliberately
    retained for training — the serializer must not conflate the two."""
    record = _make_full_episode_record(
        outcome="failure",
        disposition=Disposition.PASS,
        outcome_labeled_by="human",
        failure_modes=("dropped_object",),
    )
    payload = _serialize_record(record)
    assert payload["outcome"] == "failure"
    assert payload["disposition"] == "PASS"
    assert payload["outcome_labeled_by"] == "human"
    assert payload["failure_modes"] == ["dropped_object"]


def test_reserved_fields_present_at_documented_defaults(tmp_path: Path, profile) -> None:
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())

    assert payload["data_origin"] == "real"
    assert payload["parent_episode_ids"] == []
    assert payload["action_source"] == "measured"


def test_record_rejected_emits_reject_disposition_with_failing_check_evidence(
    tmp_path: Path, profile
) -> None:
    failing_check = CheckResult(
        "videos.master_camera_frames", "fatal", False, {"topic": "/master/cam"}
    )
    header = _make_header(
        tmp_path,
        stream_inventory=_stream_inventory(),
        checks=(failing_check,),
    )
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record_rejected(header, header.episode_id, 3)

    assert path == tmp_path / "meta" / "data_engine" / "episode_000003.json"
    payload = json.loads(path.read_text())
    assert payload["disposition"] == "REJECT"
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["check_name"] == "videos.master_camera_frames"
    assert payload["checks"][0]["passed"] is False
    assert payload["checks"][0]["evidence"] == {"topic": "/master/cam"}
    assert payload["data_origin"] == "real"
    assert payload["parent_episode_ids"] == []
    assert payload["action_source"] == "measured"
    assert payload["outcome"] == "unknown"


def test_record_rejected_channel_less_file_yields_empty_inventory(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path, stream_inventory=())
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record_rejected(header, header.episode_id, 0)
    payload = json.loads(path.read_text())
    assert payload["stream_inventory"] == []


def test_record_rejected_file_with_channels_yields_populated_inventory(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path, stream_inventory=_stream_inventory())
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record_rejected(header, header.episode_id, 0)
    payload = json.loads(path.read_text())
    assert len(payload["stream_inventory"]) == len(_stream_inventory())
    assert payload["stream_inventory"]  # non-empty, never standing in for "unexamined"


def test_documentation_field_and_module_docstring_state_not_evidence(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())
    assert "not evidence" in payload["documentation"]

    import data.provenance.recorder as recorder_module

    assert "not evidence" in (recorder_module.__doc__ or "")


def test_no_code_path_in_provenance_or_adapters_sets_outcome_off_default() -> None:
    """A grep-level guard: neither `data/provenance/` nor `data/adapters/`
    ever assigns `outcome=` to anything other than leaving it at the
    contract's own `"unknown"` default (by never passing it at all)."""
    import re

    pattern = re.compile(r"\boutcome\s*=\s*")
    for directory in ("data/provenance", "data/adapters"):
        for path in Path(directory).rglob("*.py"):
            text = path.read_text()
            for match in pattern.finditer(text):
                # Allow only `outcome=` appearing inside a dict-literal key
                # (e.g. a test's own evidence payload) — never a keyword
                # argument feeding an EpisodeRecord/ProvenanceRecorder call.
                snippet = text[max(0, match.start() - 40) : match.start() + 40]
                assert False, f"unexpected outcome= assignment in {path}: {snippet!r}"


def test_rejected_episode_record_is_byte_identical_across_two_runs(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path, stream_inventory=_stream_inventory())
    recorder1 = ProvenanceRecorder(tmp_path, profile)
    path1 = recorder1.record_rejected(header, header.episode_id, 0)
    bytes1 = path1.read_bytes()

    other_dir = tmp_path / "run2"
    other_dir.mkdir()
    recorder2 = ProvenanceRecorder(other_dir, profile)
    path2 = recorder2.record_rejected(header, header.episode_id, 0)
    bytes2 = path2.read_bytes()

    assert bytes1 == bytes2


def test_galaxea_fatal_abort_attaches_rejected_header_usable_by_record_rejected(
    tmp_path: Path, profile
) -> None:
    """Wires PROV-04's fatal-abort hook end to end: a real
    `GalaxeaInputAdapter().parse()` fatal-check failure raises
    `GalaxeaAdapterError` carrying `rejected_header`/`source_uri`, and that
    pair is directly usable by `ProvenanceRecorder.record_rejected` — the
    wiring Task 3 adds to `data/adapters/galaxea.py`."""
    fixture_path = tmp_path / "corrupt.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=5, joint_samples=5)
    fixture_mod.corrupt_chunk_crc(str(fixture_path))

    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), profile)

    error = excinfo.value
    assert error.rejected_header is not None
    assert error.source_uri == str(fixture_path)
    assert error.rejected_header.global_meta["disposition"] == Disposition.REJECT
    # Corrupted-CRC abort: the scan fails before any channel is registered —
    # genuinely channel-less, not standing in for "we did not look".
    assert error.rejected_header.global_meta["stream_inventory"] == ()

    out_dir = tmp_path / "output"
    recorder = ProvenanceRecorder(out_dir, profile)
    record_path = recorder.record_rejected(error.rejected_header, error.source_uri, 0)
    payload = json.loads(record_path.read_text())
    assert payload["disposition"] == "REJECT"
    assert any(not c["passed"] for c in payload["checks"])


def test_galaxea_fatal_abort_on_file_with_channels_yields_populated_inventory(
    tmp_path: Path, profile
) -> None:
    """A different fatal check (zero master-camera frames, with other
    channels present) proves the populated-inventory half of the same
    contract."""
    fixture_path = tmp_path / "zero_master_frames.mcap"
    fixture_mod.build_fixture_mcap(str(fixture_path), camera_frames=0, joint_samples=5)

    with pytest.raises(GalaxeaAdapterError) as excinfo:
        GalaxeaInputAdapter().parse(str(fixture_path), profile)

    error = excinfo.value
    assert error.rejected_header is not None
    assert len(error.rejected_header.global_meta["stream_inventory"]) > 0


# ---------------------------------------------------------------------------
# 04-09 — ingest_outcome: recorded at write time, carried through unchanged
# by amend_checks, and never re-derived or defaulted for a v1 record.
# ---------------------------------------------------------------------------


def test_record_stamps_ingest_outcome_written(tmp_path: Path, profile) -> None:
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record(header, _make_write_result())
    payload = json.loads(path.read_text())
    assert payload["ingest_outcome"] == "WRITTEN"


def test_record_rejected_stamps_ingest_outcome_rejected_before_write(
    tmp_path: Path, profile
) -> None:
    header = _make_header(tmp_path, stream_inventory=_stream_inventory())
    recorder = ProvenanceRecorder(tmp_path, profile)
    path = recorder.record_rejected(header, header.episode_id, 0)
    payload = json.loads(path.read_text())
    assert payload["ingest_outcome"] == "REJECTED_BEFORE_WRITE"


def test_amend_checks_carries_written_ingest_outcome_through_unchanged(
    tmp_path: Path, profile
) -> None:
    """A written episode that a later fatal check flags to `REJECT` must
    still read `ingest_outcome == "WRITTEN"` after the amend — this is the
    exact distinction 04-09 exists to make."""
    header = _make_header(tmp_path, disposition=Disposition.PASS)
    recorder = ProvenanceRecorder(tmp_path, profile)
    recorder.record(header, _make_write_result())

    new_check = CheckResult("videos.frame_count_agreement", "fatal", False, {})
    recorder.amend_checks(0, (new_check,), disposition=Disposition.REJECT)

    record_path = tmp_path / "meta" / "data_engine" / "episode_000000.json"
    payload = json.loads(record_path.read_text())
    assert payload["disposition"] == "REJECT"
    assert payload["ingest_outcome"] == "WRITTEN"


def test_amend_checks_carries_rejected_before_write_ingest_outcome_through_unchanged(
    tmp_path: Path, profile
) -> None:
    """A never-written episode's `ingest_outcome` must survive an amend
    unchanged too — `validate_dataset` amends records at ingest-rejected
    indices as well (`ValidationContext.episode_indices` scans provenance
    records, not parquet files), and defaulting an amended record to
    `WRITTEN` would flip a never-written record's durable marker."""
    header = _make_header(tmp_path, stream_inventory=_stream_inventory())
    recorder = ProvenanceRecorder(tmp_path, profile)
    recorder.record_rejected(header, header.episode_id, 0)

    new_check = CheckResult("some.check", "advisory", False, {})
    recorder.amend_checks(0, (new_check,), disposition=Disposition.REJECT)

    record_path = tmp_path / "meta" / "data_engine" / "episode_000000.json"
    payload = json.loads(record_path.read_text())
    assert payload["ingest_outcome"] == "REJECTED_BEFORE_WRITE"


def test_amend_checks_on_v1_shaped_record_leaves_ingest_outcome_none_not_written(
    tmp_path: Path, profile
) -> None:
    """A v1 record (no `ingest_outcome` key at all, `record_schema_version
    == 1`) must amend to `ingest_outcome is None` — never defaulted to
    `WRITTEN` — reproducing the exact on-disk shape of an existing EFS
    prefix predating this field."""
    header = _make_header(tmp_path)
    recorder = ProvenanceRecorder(tmp_path, profile)
    recorder.record(header, _make_write_result())

    record_path = tmp_path / "meta" / "data_engine" / "episode_000000.json"
    payload = json.loads(record_path.read_text())
    del payload["ingest_outcome"]
    payload["record_schema_version"] = 1
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    new_check = CheckResult("some.check", "advisory", True, {})
    recorder.amend_checks(0, (new_check,), disposition=Disposition.PASS)

    after = json.loads(record_path.read_text())
    assert after["ingest_outcome"] is None
    # record_schema_version is carried through unchanged by amend_checks —
    # this amend never migrates a v1 record to v2.
    assert after["record_schema_version"] == 1


def test_serialize_record_ingest_outcome_key_always_present() -> None:
    """`ingest_outcome` is always a present key — `None` when unset, never
    conditionally omitted — so a reader never has to distinguish "absent"
    from "present but null"."""
    record = _make_full_episode_record()
    payload = _serialize_record(record)
    assert "ingest_outcome" in payload
    assert payload["ingest_outcome"] is None

    record_with_outcome = dataclasses.replace(record, ingest_outcome=IngestOutcome.WRITTEN)
    payload_with_outcome = _serialize_record(record_with_outcome)
    assert payload_with_outcome["ingest_outcome"] == "WRITTEN"
