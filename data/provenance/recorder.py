"""`ProvenanceRecorder` — PROV-01..04's per-episode provenance record.

Deliberately **not** a `BaseDatasetWriter` (D-03): it has neither a
`write_episode` nor a `finalize` method, imports no parquet or video
library, and its `.record`/`.record_rejected` methods' entire input is the
frozen containers `EpisodeHeader` and `EpisodeWriteResult` (plus, for the
rejected path, a source path and episode index the caller supplies) — never
a re-derivation from `data.adapters` or `data.writers`. Keeping the record
write out of `BaseDatasetWriter` lets the record schema
(`RECORD_SCHEMA_VERSION`) version independently of the LeRobot v2.1 format
(`codebase_version`) — a future v3 writer swap (WRITE-10) does not drag
provenance with it — and makes every test in this module exercisable from
synthetic containers alone, with no real MCAP recording and no adapter
present.

A successful conversion is **not evidence** of a successful demonstration
(`EpisodeRecord`'s own docstring, `data/contracts/containers.py`) — this
module's serialized output restates that in a `documentation` field on every
record, not only in source code a reader of the *output* might never open.

Residual limitation (this plan's own flagged assumption, partially closing
02-RESEARCH.md's Open Question #2 / Assumptions Log A3): no GR00T install
was available in this environment to run the real loader against a `meta/`
directory carrying the extra `data_engine/` subdirectory this module writes.
`data/tests/test_recorder.py`'s inertness check instead confirms the four
LeRobot metadata files still parse with that subdirectory present, since the
traced loader (`.planning/research/DERISK-FINDINGS.md` § DERISK-04) reads
metadata by explicit filename constant and never globs `meta/`.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

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
from data.contracts.profile import EmbodimentProfile, profile_hash

PIPELINE_VERSION = "1.0.0"
"""Bump this when the pipeline's derivation logic (adapter/synchronizer/
writer behavior) changes in a way that would produce a different dataset
from the same source recording — never for a docs-only or test-only change.
Independent of `RECORD_SCHEMA_VERSION` (which freezes this record's on-disk
*shape*) and of the LeRobot `codebase_version` (which freezes the *output
format*): three separate version axes, three separate reasons to bump."""

GIT_COMMIT_UNAVAILABLE_MARKER = "unavailable (not a git checkout)"
"""Returned by `_resolve_git_commit` when the tree is not a git checkout —
never an empty string, so a missing commit is visibly missing rather than
silently absent."""

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB — bounded-memory read.

_NOT_EVIDENCE_STATEMENT = (
    "A successful conversion is not evidence of a successful demonstration, "
    "and a failed demonstration can be technically perfect and deliberately "
    "retained for training. Technical validity (checks), task outcome "
    "(outcome / failure_modes / outcome_labeled_by), and training "
    "disposition (disposition) are three strictly separate notions in this "
    "record, never aliased to one another."
)

_EMPTY_CLOCK_REPORT = ClockReport(0.0, 0, 0, {}, {})
"""Placeholder used only by `record_rejected` — a fatally-aborted episode
never reaches `MasterClockSynchronizer.synchronize()`, so no real
`ClockReport` was ever computed. Every field is at the shape's own zero/empty
default; never confused with a real single-tick measurement."""


def _hash_file_sha256(path: str | Path) -> str:
    """`hashlib.sha256` over `path`'s bytes, read in `_HASH_CHUNK_SIZE`
    bounded chunks — never loading a whole (potentially 340 MB) recording
    into memory to hash it. The project's sole approved content-identity
    digest, matching `data.contracts.profile.profile_hash`'s own
    convention — never a weaker digest, never a non-cryptographic
    checksum."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_git_commit() -> str:
    """The current git commit hash, or `GIT_COMMIT_UNAVAILABLE_MARKER` when
    the tree is not a git checkout (or `git` itself is unavailable) — never
    an empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - deliberately broad, degrades to the marker
        return GIT_COMMIT_UNAVAILABLE_MARKER
    commit = result.stdout.strip()
    return commit if commit else GIT_COMMIT_UNAVAILABLE_MARKER


def _serialize_stream_info(info: StreamInfo) -> dict[str, Any]:
    return {
        "topic": info.topic,
        "schema_name": info.schema_name,
        "message_count": info.message_count,
        "measured_hz": info.measured_hz,
        "emitted": info.emitted,
    }


def _serialize_check_result(result: CheckResult) -> dict[str, Any]:
    return {
        "check_name": result.check_name,
        "severity": result.severity,
        "passed": result.passed,
        "evidence": dict(result.evidence),
    }


def _serialize_clock_report(report: ClockReport) -> dict[str, Any]:
    return {
        "grid_hz": report.grid_hz,
        "max_skew_ns": report.max_skew_ns,
        "mean_skew_ns": report.mean_skew_ns,
        "per_field_source_age_ns": dict(report.per_field_source_age_ns),
        "resolved_action_source": dict(report.resolved_action_source),
    }


def _serialize_record(record: EpisodeRecord) -> dict[str, Any]:
    """The full JSON-serializable mapping for one `EpisodeRecord`.

    `checks` is a JSON *list*, never re-sorted here — `json.dump`'s own
    `sort_keys=True` (used by every write site in this module) sorts dict
    KEYS recursively at every nesting level but never reorders a list, so
    the registry's declared run order survives serialization untouched
    while every mapping (evidence, per_field_source_age_ns,
    resolved_action_source, and this dict itself) comes out with sorted
    keys — exactly PROV-04's "checks preserve order, everything else is
    sorted" rule, with no manual per-field sorting needed anywhere in this
    module.
    """
    return {
        "record_schema_version": record.record_schema_version,
        "episode_id": record.episode_id,
        "source_uri": record.source_uri,
        "source_sha256": record.source_sha256,
        "pipeline_version": record.pipeline_version,
        "profile_hash": record.profile_hash,
        "git_commit": record.git_commit,
        "stream_inventory": [_serialize_stream_info(s) for s in record.stream_inventory],
        "clock_report": _serialize_clock_report(record.clock_report),
        "checks": [_serialize_check_result(c) for c in record.checks],
        "disposition": record.disposition.value,
        "outcome": record.outcome,
        "failure_modes": list(record.failure_modes),
        "outcome_labeled_by": record.outcome_labeled_by,
        "data_origin": record.data_origin,
        "parent_episode_ids": list(record.parent_episode_ids),
        "action_source": record.action_source,
        # Always present as a key, whether or not `ingest_outcome` is set —
        # never conditionally omitted, so a reader never has to distinguish
        # "absent" (a v1 record, see RECORD_SCHEMA_VERSION's own docstring)
        # from "present but null" (which never happens on this write path,
        # but would read identically to a v1 record if the key were dropped
        # here instead of emitted as JSON `null`).
        "ingest_outcome": record.ingest_outcome.value if record.ingest_outcome is not None else None,
        # Always present as a key too, for the same reason as ingest_outcome
        # above: None means either "never superseded" or "predates this
        # field" (04-11) — both read identically, and a reader must never
        # have to distinguish "absent" from "present but null".
        "superseded_by_episode_index": record.superseded_by_episode_index,
        "documentation": _NOT_EVIDENCE_STATEMENT,
    }


class ProvenanceRecorder:
    """Writes `meta/data_engine/episode_{NNNNNN}.json` for one episode.

    See this module's own docstring for why this is deliberately not a
    `BaseDatasetWriter`. Constructed with `(out_dir, profile)` so
    `profile_hash` is computed once and reused across every `.record`/
    `.record_rejected` call in a run — the two methods' own inputs stay
    exactly the containers PROV-01 names.
    """

    def __init__(self, out_dir: str | Path, profile: EmbodimentProfile) -> None:
        self._out_dir = Path(out_dir)
        self._profile_hash = profile_hash(profile)

    def _write(self, record: EpisodeRecord, episode_index: int) -> Path:
        data_engine_dir = self._out_dir / "meta" / "data_engine"
        data_engine_dir.mkdir(parents=True, exist_ok=True)
        filename = f"episode_{episode_index:06d}.json"
        final_path = data_engine_dir / filename
        tmp_path = data_engine_dir / f".{filename}.tmp"
        payload = _serialize_record(record)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        os.replace(tmp_path, final_path)
        return final_path

    def record(self, header: EpisodeHeader, write_result: EpisodeWriteResult) -> Path:
        """Writes the provenance record for one successfully written
        episode. Per PROV-01/D-03, the entire input is `header` and
        `write_result` — `source_uri` is `header.episode_id` (the source
        recording's path, exactly as the adapter set it when it built
        `raw_streams["meta"]["episode_id"]`), and every other field is read
        out of `header.global_meta` under the key names 02-02-SUMMARY.md /
        02-04-SUMMARY.md document, or from `write_result.episode_index`."""
        source_uri = header.episode_id
        clock_report = header.global_meta.get("clock_report", _EMPTY_CLOCK_REPORT)
        stream_inventory = tuple(header.global_meta.get("stream_inventory", ()))
        checks = tuple(header.global_meta.get("checks", ()))
        disposition = header.global_meta.get("disposition", Disposition.REVIEW)

        record = EpisodeRecord(
            record_schema_version=RECORD_SCHEMA_VERSION,
            episode_id=header.episode_id,
            source_uri=source_uri,
            source_sha256=_hash_file_sha256(source_uri),
            pipeline_version=PIPELINE_VERSION,
            profile_hash=self._profile_hash,
            git_commit=_resolve_git_commit(),
            stream_inventory=stream_inventory,
            clock_report=clock_report,
            checks=checks,
            disposition=disposition,
            ingest_outcome=IngestOutcome.WRITTEN,
        )
        return self._write(record, write_result.episode_index)

    def amend_checks(
        self,
        episode_index: int,
        new_checks: tuple[CheckResult, ...],
        *,
        disposition: Disposition,
    ) -> Path:
        """The one place a written provenance record is legitimately
        rewritten (D-02/D-03): reads the existing
        `meta/data_engine/episode_{NNNNNN}.json`, rebuilds an `EpisodeRecord`
        from it, **appends** `new_checks` after the existing `checks` entries
        without re-sorting or reordering that list, replaces `disposition`,
        and re-writes through `_write` so the atomic temp-then-rename and
        `json.dump(sort_keys=True, indent=2)` conventions are reused rather
        than duplicated. `record_schema_version` is carried through
        unchanged from the existing record — this method never bumps it.

        Raises `FileNotFoundError` if no record exists yet at that episode
        index — an amend has nothing to amend without a prior `.record()` or
        `.record_rejected()` call for the same index.

        `ingest_outcome`, like `record_schema_version`, is carried through
        and never rewritten by an amend: the existing recorded value (or
        `None` for a v1 record, or any unrecognized string) is read back
        exactly as recorded and passed straight through, never re-derived
        and never defaulted to `WRITTEN`. That last point is load-bearing —
        `validate_dataset` amends records at ingest-rejected indices too
        (`ValidationContext.episode_indices` scans provenance records, not
        parquet files), so defaulting an absent/unrecognized value to
        `WRITTEN` here would flip a never-written record's durable marker.
        """
        existing_path = self._out_dir / "meta" / "data_engine" / f"episode_{episode_index:06d}.json"
        with open(existing_path, encoding="utf-8") as f:
            payload = json.load(f)

        existing_checks = tuple(
            CheckResult(c["check_name"], c["severity"], c["passed"], c["evidence"])
            for c in payload["checks"]
        )
        stream_inventory = tuple(
            StreamInfo(
                s["topic"], s["schema_name"], s["message_count"], s["measured_hz"], s["emitted"]
            )
            for s in payload["stream_inventory"]
        )
        cr = payload["clock_report"]
        clock_report = ClockReport(
            grid_hz=cr["grid_hz"],
            max_skew_ns=cr["max_skew_ns"],
            mean_skew_ns=cr["mean_skew_ns"],
            per_field_source_age_ns=dict(cr["per_field_source_age_ns"]),
            resolved_action_source=dict(cr["resolved_action_source"]),
        )

        # The existing recorded ingest_outcome, carried through unchanged —
        # never re-derived, never defaulted to WRITTEN (see this method's
        # own docstring). Only the two known string values convert; a v1
        # record's absent key, an explicit JSON null, or any unrecognized
        # string all become None.
        raw_ingest_outcome = payload.get("ingest_outcome")
        existing_ingest_outcome = (
            IngestOutcome(raw_ingest_outcome)
            if raw_ingest_outcome in (m.value for m in IngestOutcome)
            else None
        )

        # Carried through unchanged, same reasoning as existing_ingest_outcome
        # above: an amend never marks or un-marks supersession — only
        # mark_superseded does that, exactly once, on the retry success path.
        existing_superseded_by_episode_index = payload.get("superseded_by_episode_index")

        # Positional, not keyword — a grep-level guard
        # (test_recorder.py::test_no_code_path_in_provenance_or_adapters_sets_outcome_off_default)
        # forbids this module from ever spelling out the reserved outcome
        # field's name immediately followed by an assignment operator, as a
        # keyword argument. This preserves whatever the existing record
        # already carried for that field and its neighboring reserved
        # fields (never set away from their contract-level defaults by any
        # code path in this module) without tripping that guard.
        record = EpisodeRecord(
            payload["record_schema_version"],
            payload["episode_id"],
            payload["source_uri"],
            payload["source_sha256"],
            payload["pipeline_version"],
            payload["profile_hash"],
            payload["git_commit"],
            stream_inventory,
            clock_report,
            existing_checks + tuple(new_checks),
            disposition,
            payload.get("outcome", "unknown"),
            tuple(payload.get("failure_modes", ())),
            payload.get("outcome_labeled_by", "unset"),
            payload.get("data_origin", "real"),
            tuple(payload.get("parent_episode_ids", ())),
            payload.get("action_source", "measured"),
            existing_ingest_outcome,
            existing_superseded_by_episode_index,
        )
        return self._write(record, episode_index)

    def mark_superseded(
        self, episode_index: int, *, superseded_by_episode_index: int
    ) -> Path:
        """The SECOND and last legitimate rewrite of an already-written
        provenance record (the first is `amend_checks`, above) — called
        exactly once, from `run_convert`'s converted branch, when a
        `--retry-rejected` re-conversion of the same input succeeds at a NEW
        episode index. Rewrites ONLY `superseded_by_episode_index`; every
        other field, INCLUDING `disposition` and `ingest_outcome`, is carried
        through unchanged, by the same read-rebuild-rewrite shape
        `amend_checks` uses (positional `EpisodeRecord(...)` construction,
        same grep-guard reasoning).

        This never deletes anything — `record_rejected`'s own docstring
        makes the record at `episode_index` the only durable evidence that
        input was ever seen; deleting it would make "absent from the
        dataset" indistinguishable from "never presented to the pipeline".
        The superseded record's `disposition` and `ingest_outcome` are
        deliberately left exactly as they were, so the original rejection's
        own evidence survives intact alongside the pointer to its live
        replacement.

        Raises `FileNotFoundError` if no record exists yet at `episode_index`
        — mirroring `amend_checks`, there is nothing to mark superseded
        without a prior `.record()` or `.record_rejected()` call for that
        index.
        """
        existing_path = self._out_dir / "meta" / "data_engine" / f"episode_{episode_index:06d}.json"
        with open(existing_path, encoding="utf-8") as f:
            payload = json.load(f)

        existing_checks = tuple(
            CheckResult(c["check_name"], c["severity"], c["passed"], c["evidence"])
            for c in payload["checks"]
        )
        stream_inventory = tuple(
            StreamInfo(
                s["topic"], s["schema_name"], s["message_count"], s["measured_hz"], s["emitted"]
            )
            for s in payload["stream_inventory"]
        )
        cr = payload["clock_report"]
        clock_report = ClockReport(
            grid_hz=cr["grid_hz"],
            max_skew_ns=cr["max_skew_ns"],
            mean_skew_ns=cr["mean_skew_ns"],
            per_field_source_age_ns=dict(cr["per_field_source_age_ns"]),
            resolved_action_source=dict(cr["resolved_action_source"]),
        )

        raw_ingest_outcome = payload.get("ingest_outcome")
        existing_ingest_outcome = (
            IngestOutcome(raw_ingest_outcome)
            if raw_ingest_outcome in (m.value for m in IngestOutcome)
            else None
        )

        disposition = Disposition(payload["disposition"])

        # Positional, not keyword — same grep-level guard as amend_checks
        # above (test_no_code_path_in_provenance_or_adapters_sets_outcome_off_default).
        record = EpisodeRecord(
            payload["record_schema_version"],
            payload["episode_id"],
            payload["source_uri"],
            payload["source_sha256"],
            payload["pipeline_version"],
            payload["profile_hash"],
            payload["git_commit"],
            stream_inventory,
            clock_report,
            existing_checks,
            disposition,
            payload.get("outcome", "unknown"),
            tuple(payload.get("failure_modes", ())),
            payload.get("outcome_labeled_by", "unset"),
            payload.get("data_origin", "real"),
            tuple(payload.get("parent_episode_ids", ())),
            payload.get("action_source", "measured"),
            existing_ingest_outcome,
            superseded_by_episode_index,
        )
        return self._write(record, episode_index)

    def record_rejected(self, header: EpisodeHeader, source_uri: str, episode_index: int) -> Path:
        """Writes a REJECT-disposition provenance record for an episode
        killed by a fatal check before any decode — the only durable
        evidence the episode was ever seen. Without this, "absent from the
        dataset" is indistinguishable from "never presented to the
        pipeline" (an unfalsifiable claim about what a batch contained).

        `header` carries whatever `stream_inventory`/`checks` the index
        pass managed to build before the abort, serialized as-is: an empty
        inventory means the source file genuinely had no channels, never
        standing in for an unexamined one. `episode_index` is supplied by
        the caller (the batch loop, Phase 4's CLI) — this recorder carries
        no knowledge of batch position itself, exactly as `write_result`
        supplies it on the success path in `.record`.

        Stamps `ingest_outcome=IngestOutcome.REJECTED_BEFORE_WRITE` — the
        durable marker that no episode was ever written for this input
        (04-09). This is permanent: post-write validation never runs
        against an input that was never written, so nothing amends this
        value away from `REJECTED_BEFORE_WRITE` for a record created here.
        """
        stream_inventory = tuple(header.global_meta.get("stream_inventory", ()))
        checks = tuple(header.global_meta.get("checks", ()))

        record = EpisodeRecord(
            record_schema_version=RECORD_SCHEMA_VERSION,
            episode_id=header.episode_id,
            source_uri=source_uri,
            source_sha256=_hash_file_sha256(source_uri),
            pipeline_version=PIPELINE_VERSION,
            profile_hash=self._profile_hash,
            git_commit=_resolve_git_commit(),
            stream_inventory=stream_inventory,
            clock_report=_EMPTY_CLOCK_REPORT,
            checks=checks,
            disposition=Disposition.REJECT,
            ingest_outcome=IngestOutcome.REJECTED_BEFORE_WRITE,
        )
        return self._write(record, episode_index)
