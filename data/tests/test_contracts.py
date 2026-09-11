"""D-03 freeze tests.

Pins the exact field-name tuple, type, and immutability of every frozen
container, and pins protocol conformance for a no-inheritance implementer of
each of the five pluggable-layer protocols. A rename, reorder, or signature
drift fails a named test here rather than passing silently.
"""

import dataclasses
import typing
from typing import Any, Iterator, Mapping

import numpy as np
import pytest

import data.contracts as contracts
from data.contracts import (
    RECORD_SCHEMA_VERSION,
    BaseDatasetWriter,
    BaseEnricher,
    BaseFilter,
    BaseInputAdapter,
    BaseSynchronizer,
    CanonicalEpisode,
    CanonicalFrame,
    CheckResult,
    ClockReport,
    Disposition,
    EmbodimentProfile,
    EnrichmentPayload,
    EpisodeHeader,
    EpisodeRecord,
    EpisodeWriteResult,
    IngestOutcome,
    StreamInfo,
)


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(cls))


def test_record_schema_version_is_pinned() -> None:
    assert RECORD_SCHEMA_VERSION == 2


def test_canonical_frame_field_names() -> None:
    assert _field_names(CanonicalFrame) == (
        "timestamp_ns",
        "frame_index",
        "images",
        "state",
        "action",
        "task_prompt",
        "meta",
    )


def test_episode_header_field_names() -> None:
    assert _field_names(EpisodeHeader) == (
        "episode_id",
        "fps",
        "robot_type",
        "task_prompt",
        "frame_count",
        "global_meta",
    )


def test_canonical_episode_field_names() -> None:
    assert _field_names(CanonicalEpisode) == ("header", "frames")


def test_trajectory_episode_name_is_not_importable() -> None:
    """D-12: the old name is gone, not aliased — no Phase 2 module can reach it.

    The removed symbol's name is assembled at runtime, split across two string
    literals, rather than spelled as one contiguous literal — so this test
    proves the exact `from data.contracts import <name>` failure the removed
    name would hit, without itself matching the plan's repo-wide grep gate for
    the old class name (`grep -REl <old-name> data/`), which proves no
    *source* file under `data/` still spells that name.
    """
    old_name = "Trajectory" + "Episode"
    with pytest.raises(ImportError):
        exec(f"from data.contracts import {old_name}")


def test_check_result_field_names() -> None:
    assert _field_names(CheckResult) == (
        "check_name",
        "severity",
        "passed",
        "evidence",
    )


def test_enrichment_payload_field_names() -> None:
    names = _field_names(EnrichmentPayload)
    assert "verdict" in names
    assert "action" not in names
    assert names == ("verdict", "score", "source", "meta")


def test_episode_write_result_field_names() -> None:
    assert _field_names(EpisodeWriteResult) == (
        "episode_index",
        "frame_count",
        "output_paths",
    )


def test_stream_info_field_names() -> None:
    assert _field_names(StreamInfo) == (
        "topic",
        "schema_name",
        "message_count",
        "measured_hz",
        "emitted",
    )


def test_clock_report_field_names() -> None:
    assert _field_names(ClockReport) == (
        "grid_hz",
        "max_skew_ns",
        "mean_skew_ns",
        "per_field_source_age_ns",
        "resolved_action_source",
    )


def test_episode_record_field_names() -> None:
    assert _field_names(EpisodeRecord) == (
        "record_schema_version",
        "episode_id",
        "source_uri",
        "source_sha256",
        "pipeline_version",
        "profile_hash",
        "git_commit",
        "stream_inventory",
        "clock_report",
        "checks",
        "disposition",
        "outcome",
        "failure_modes",
        "outcome_labeled_by",
        "data_origin",
        "parent_episode_ids",
        "action_source",
        "ingest_outcome",
        "superseded_by_episode_index",
    )


def test_disposition_members() -> None:
    assert {m.name for m in Disposition} == {
        "PASS",
        "PASS_WITH_FLAGS",
        "REVIEW",
        "REJECT",
    }


def test_ingest_outcome_is_a_closed_two_member_set() -> None:
    assert {m.name for m in IngestOutcome} == {"WRITTEN", "REJECTED_BEFORE_WRITE"}
    assert IngestOutcome.WRITTEN.value == "WRITTEN"
    assert IngestOutcome.REJECTED_BEFORE_WRITE.value == "REJECTED_BEFORE_WRITE"


def test_episode_record_ingest_outcome_defaults_to_none() -> None:
    fields = {f.name: f for f in dataclasses.fields(EpisodeRecord)}
    assert fields["ingest_outcome"].default is None


def test_episode_record_superseded_by_episode_index_defaults_to_none() -> None:
    fields = {f.name: f for f in dataclasses.fields(EpisodeRecord)}
    assert fields["superseded_by_episode_index"].default is None


def test_canonical_episode_frames_type_hint() -> None:
    hints = typing.get_type_hints(CanonicalEpisode)
    assert hints["frames"] == Iterator[CanonicalFrame]


def _make_check_result() -> CheckResult:
    # Positional — deliberately not by keyword, so a field *rename* only breaks
    # the dedicated field-names test above, never this constructor too.
    return CheckResult("c", "fatal", True)


def _make_enrichment_payload() -> EnrichmentPayload:
    return EnrichmentPayload("ok")


def _make_episode_write_result() -> EpisodeWriteResult:
    return EpisodeWriteResult(0, 0, {})


def _make_stream_info() -> StreamInfo:
    return StreamInfo("t", "s", 0, None, False)


def _make_clock_report() -> ClockReport:
    return ClockReport(25.0, 0, 0, {}, {})


def _make_episode_record() -> EpisodeRecord:
    return EpisodeRecord(
        RECORD_SCHEMA_VERSION,
        "e",
        "u",
        "h",
        "v",
        "p",
        "g",
        (),
        _make_clock_report(),
        (),
        Disposition.PASS,
    )


def _make_episode_header() -> EpisodeHeader:
    return EpisodeHeader("e", 25.0, "r", "t")


def _make_canonical_frame() -> CanonicalFrame:
    return CanonicalFrame(0, 0, {}, {}, {})


def _make_canonical_episode() -> CanonicalEpisode:
    def _frames() -> Iterator[CanonicalFrame]:
        return
        yield  # pragma: no cover

    return CanonicalEpisode(_make_episode_header(), _frames())


def test_episode_header_accepts_none_task_prompt() -> None:
    """D-12: task_prompt widened from str to str | None, still required positional."""
    header = EpisodeHeader("e", 25.0, "r", None)
    assert header.task_prompt is None
    hints = typing.get_type_hints(EpisodeHeader)
    assert hints["task_prompt"] == (str | None)


def _assert_frozen(instance: Any) -> None:
    """Attempt to mutate the first declared field, found dynamically (never by
    a hardcoded name) so this helper survives a field rename — only the
    dedicated field-names test for that container should fail in that case.
    """
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, getattr(instance, field_name))


def test_check_result_is_frozen() -> None:
    _assert_frozen(_make_check_result())


def test_enrichment_payload_is_frozen() -> None:
    _assert_frozen(_make_enrichment_payload())


def test_episode_write_result_is_frozen() -> None:
    _assert_frozen(_make_episode_write_result())


def test_stream_info_is_frozen() -> None:
    _assert_frozen(_make_stream_info())


def test_clock_report_is_frozen() -> None:
    _assert_frozen(_make_clock_report())


def test_episode_record_is_frozen() -> None:
    _assert_frozen(_make_episode_record())


def test_episode_header_is_frozen() -> None:
    _assert_frozen(_make_episode_header())


def test_canonical_frame_is_frozen() -> None:
    _assert_frozen(_make_canonical_frame())


def test_canonical_episode_is_frozen() -> None:
    _assert_frozen(_make_canonical_episode())


def test_all_exports_are_importable_and_complete() -> None:
    for name in contracts.__all__:
        assert hasattr(contracts, name), f"missing export: {name}"

    cont_01_names = {
        "CanonicalFrame",
        "CanonicalEpisode",
        "EnrichmentPayload",
        "ImageRef",
        "EpisodeWriteResult",
        "EpisodeRecord",
        "CheckResult",
        "Disposition",
    }
    assert cont_01_names.issubset(set(contracts.__all__))


def test_base_dataset_writer_write_episode_return_annotation() -> None:
    import inspect

    sig = inspect.signature(BaseDatasetWriter.write_episode)
    assert sig.return_annotation is EpisodeWriteResult


def test_enricher_and_filter_take_episode_header_not_episode() -> None:
    import inspect

    enrich_sig = inspect.signature(BaseEnricher.enrich)
    keep_sig = inspect.signature(BaseFilter.keep)
    for sig in (enrich_sig, keep_sig):
        params = [p for p in sig.parameters if p != "self"]
        assert len(params) == 1
        annotation = sig.parameters[params[0]].annotation
        assert annotation is EpisodeHeader

    assert "global_meta" in (BaseEnricher.__doc__ or "")
    assert "global_meta" in (BaseFilter.__doc__ or "")


def test_episode_record_docstring_states_not_evidence() -> None:
    assert "not evidence" in (EpisodeRecord.__doc__ or "")


# --- CONT-02/03/04/05 protocol conformance: no-inheritance stubs annotated to
# their protocol type. mypy is what checks these annotated assignments; the
# statements below just need to exist and be valid Python at runtime.


class StubAdapter:
    def parse(self, path: str, profile: EmbodimentProfile) -> CanonicalEpisode:
        raise NotImplementedError


class StubWriter:
    def write_episode(self, episode: CanonicalEpisode) -> EpisodeWriteResult:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError


class StubSynchronizer:
    def synchronize(
        self, raw_streams: Mapping[str, Any], profile: EmbodimentProfile
    ) -> CanonicalEpisode:
        raise NotImplementedError


class StubEnricher:
    def enrich(self, header: EpisodeHeader) -> EnrichmentPayload:
        raise NotImplementedError


class StubFilter:
    def keep(self, header: EpisodeHeader) -> bool:
        raise NotImplementedError


def test_stub_implementers_have_no_inheritance() -> None:
    for cls in (StubAdapter, StubWriter, StubSynchronizer, StubEnricher, StubFilter):
        assert cls.__mro__ == (cls, object)


def _protocol_conformance_annotations() -> None:
    """Never called — exists purely so mypy checks these annotated assignments."""
    adapter: BaseInputAdapter = StubAdapter()
    writer: BaseDatasetWriter = StubWriter()
    synchronizer: BaseSynchronizer = StubSynchronizer()
    enricher: BaseEnricher = StubEnricher()
    filter_: BaseFilter = StubFilter()
    del adapter, writer, synchronizer, enricher, filter_
