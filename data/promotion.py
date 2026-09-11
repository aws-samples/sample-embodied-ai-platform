"""CLI-08's never-write-in-place promotion guard, CLI-06's run-level
provenance manifest, and CLI-09's dataset revision identifier.

`run_manifest.json` (`RUN_MANIFEST_FILENAME`) is the single durable,
per-prefix, cumulatively-rewritten provenance file `evaluate_prefix` reads:
`dataset_revision_id`'s composite hash names every run's converted content,
and `append_run_to_manifest` folds one more run's entry into its `runs`
list on every invocation — never overwriting an earlier run's entry.
`run_reports/run_{NNN:03d}.json` (`RUN_REPORTS_DIRNAME`) is one immutable
report per run, written once by `write_run_report` and never overwritten —
`NNN` is the run's ordinal position in the manifest's `runs` list.

Rewriting the manifest additively on every run is deliberate and does not
violate CLI-08's append-only rule: that rule governs an already-written
episode's parquet/video/metadata, not this run-scoped file. The manifest
carries `manifest_schema_version` (`MANIFEST_SCHEMA_VERSION`), mirroring
`RECORD_SCHEMA_VERSION` (`data/contracts/containers.py`), so a future shape
change is a versioned migration, never a silent break.

`evaluate_prefix` is this module's read-only guard: called before any
write, it decides fresh-vs-resume-vs-refuse by comparing this run's
`(pipeline_version, profile_hash, format_key)` against whatever is already
recorded under a prefix, and never mutates anything it reads. This module's
entire job is exact-string comparison plus canonical serialization, so it
deliberately contains no case-folding, whitespace-trimming, or
Unicode-normalization call anywhere — a recorded value differing from this
run's only in letter case or padding must never compare equal.

Deliberately NOT wired into this guard: REGR-01 and REGR-02. Every call
into `validate_dataset` from `data/cli.py` passes `baseline=None`, and
`use_local_golden` is forwarded only from the explicit `--use-local-golden`
flag — honouring Phase 3 D-08's rule that a local golden is never enabled
implicitly. `diff_datasets` (REGR-02) stays a separate, manually-invoked
tool; nothing in this module or in `data/cli.py` calls it.
"""

import enum
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_REPORTS_DIRNAME = "run_reports"

_EPISODE_RECORD_FILENAME_RE = re.compile(r"episode_\d{6}\.json")


class RunManifestError(Exception):
    """Raised when an existing run manifest is unreadable or carries an
    unrecognized `manifest_schema_version`, or when `write_run_report` finds
    a report already at the target path (an immutable per-run report is
    never overwritten)."""


def dataset_revision_id(
    source_sha256s: list[str],
    profile_hash_value: str,
    git_commit: str,
    format_key: str,
) -> tuple[str, str]:
    """CLI-09's content-derived dataset revision identifier.

    Sha256 over a canonicalized composite of (every converted source
    recording's sha256, the profile hash, the pipeline's git commit, the
    output format key) — deterministic and order-independent: sorting
    `source_sha256s` before hashing means two calls supplying the same hash
    set in a different order return identical values. No non-content input
    (wall-clock time, hostname, run ordinal, filesystem path, environment
    variable) is ever folded into this composite — every element hashed
    here is recomputable from the four recorded inputs alone.

    Returns:
        `(short, full)`. `full` is the 64-hex-character sha256 digest —
        every equality comparison and every recorded attribution must use
        this value. `short` is `"rev-"` plus `full`'s first eight hex
        characters, display-only.
    """
    composite = {
        "source_sha256s": sorted(source_sha256s),
        "profile_hash": profile_hash_value,
        "git_commit": git_commit,
        "format_key": format_key,
    }
    full = hashlib.sha256(json.dumps(composite, sort_keys=True).encode("utf-8")).hexdigest()
    return f"rev-{full[:8]}", full


def read_run_manifest(out_dir: str | Path) -> dict[str, Any] | None:
    """Returns the parsed `run_manifest.json` under `out_dir`, or `None`
    when no manifest file exists yet (a fresh prefix, or a prefix that
    predates this manifest).

    Raises:
        RunManifestError: naming the path, when the file exists but is
            unparseable, does not parse to a JSON mapping, or carries an
            unrecognized `manifest_schema_version`.
    """
    manifest_path = Path(out_dir) / "meta" / "data_engine" / RUN_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RunManifestError(f"run manifest at {manifest_path} could not be read: {e}") from e
    if not isinstance(manifest, dict):
        raise RunManifestError(
            f"run manifest at {manifest_path} did not parse to a JSON mapping"
        )
    version = manifest.get("manifest_schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise RunManifestError(
            f"run manifest at {manifest_path} carries manifest_schema_version "
            f"{version!r}, expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    return manifest


def append_run_to_manifest(
    out_dir: str | Path,
    *,
    run_entry: dict[str, Any],
    pipeline_version: str,
    profile_hash_value: str,
    format_key: str,
    dataset_revision: dict[str, Any],
) -> Path:
    """Appends `run_entry` to the manifest's `runs` list — with
    `run_ordinal` set to the pre-append length of that list — and replaces
    the top-level `pipeline_version`/`profile_hash`/`format_key`/
    `dataset_revision` fields with this call's values. Starts a fresh
    manifest (an empty `runs` list) when none exists yet under `out_dir`.

    Writes through the temp-then-atomic-rename plus
    `json.dump(sort_keys=True, indent=2)` convention
    `ProvenanceRecorder._write` (`data/provenance/recorder.py`) established.
    """
    out_dir = Path(out_dir)
    data_engine_dir = out_dir / "meta" / "data_engine"
    data_engine_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_engine_dir / RUN_MANIFEST_FILENAME

    existing = read_run_manifest(out_dir)
    runs: list[dict[str, Any]] = list(existing["runs"]) if existing is not None else []
    runs.append({**run_entry, "run_ordinal": len(runs)})

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_version": pipeline_version,
        "profile_hash": profile_hash_value,
        "format_key": format_key,
        "dataset_revision": dataset_revision,
        "runs": runs,
    }

    tmp_path = manifest_path.parent / f".{manifest_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def write_run_report(out_dir: str | Path, *, run_ordinal: int, payload: dict[str, Any]) -> Path:
    """Writes `meta/data_engine/run_reports/run_{run_ordinal:03d}.json` —
    one immutable report per run, following the same atomic-write
    convention as `append_run_to_manifest`.

    Raises:
        RunManifestError: naming the path, rather than overwriting an
            existing report already at that ordinal.
    """
    reports_dir = Path(out_dir) / "meta" / "data_engine" / RUN_REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"run_{run_ordinal:03d}.json"
    if report_path.exists():
        raise RunManifestError(
            f"run report already exists at {report_path} — refusing to "
            "overwrite an immutable per-run report"
        )
    tmp_path = report_path.parent / f".{report_path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, report_path)
    return report_path


class PrefixVerdict(enum.Enum):
    """`evaluate_prefix`'s four possible verdicts (CLI-08)."""

    FRESH = "FRESH"
    RESUME = "RESUME"
    REFUSE = "REFUSE"
    STATE_ERROR = "STATE_ERROR"


@dataclass(frozen=True)
class PrefixDecision:
    """`evaluate_prefix`'s return value. `mismatched_field`/`expected`/
    `found` are populated only for `REFUSE` and `STATE_ERROR`;
    `recorded_source_sha256s` is the sorted, deduplicated set of every
    `source_sha256` already recorded under the prefix (empty for `FRESH`),
    so the caller does not need to re-read the records it already
    implicitly scanned."""

    verdict: PrefixVerdict
    reason: str
    mismatched_field: str | None
    expected: str | None
    found: str | None
    recorded_source_sha256s: tuple[str, ...]


def _read_recorded_episode_fields(out_dir: Path) -> dict[int, dict[str, Any]]:
    """Every per-episode provenance record's `(pipeline_version,
    profile_hash, source_sha256, disposition, ingest_outcome,
    superseded_by_episode_index)`, keyed by episode index — selected by the
    exact `episode_{NNNNNN}.json` filename pattern so `run_manifest.json`
    and `run_reports/` (both also under `meta/data_engine/`) are excluded by
    construction rather than by a blanket directory glob.

    Widened to `dict[int, dict[str, Any]]` (04-11) because the mapping now
    carries an integer-or-null field (`superseded_by_episode_index`)
    alongside strings — narrower than `_serialize_record`'s full payload,
    still not `dict[int, str]`. `ingest_outcome` and
    `superseded_by_episode_index` are the only fields whose value may
    genuinely be `None` (a v1 record predating `RECORD_SCHEMA_VERSION` 2, or
    an explicit JSON `null`, or — for the latter — simply never
    superseded) — every other field is a required part of the record shape
    at every schema version and is asserted non-`None` by this function's
    own callers before being narrowed to `dict[int, str]`."""
    data_engine_dir = out_dir / "meta" / "data_engine"
    if not data_engine_dir.exists():
        return {}
    fields: dict[int, dict[str, Any]] = {}
    for path in sorted(data_engine_dir.glob("episode_*.json")):
        if not _EPISODE_RECORD_FILENAME_RE.fullmatch(path.name):
            continue
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        episode_index = int(path.stem.split("_")[1])
        fields[episode_index] = {
            "pipeline_version": record["pipeline_version"],
            "profile_hash": record["profile_hash"],
            "source_sha256": record["source_sha256"],
            "disposition": record["disposition"],
            "ingest_outcome": record.get("ingest_outcome"),
            "superseded_by_episode_index": record.get("superseded_by_episode_index"),
        }
    return fields


def read_recorded_source_hashes(out_dir: str | Path) -> dict[int, str]:
    """`{episode_index: source_sha256}` for every per-episode provenance
    record already recorded under `out_dir` — selected by the exact
    `episode_{NNNNNN}.json` filename pattern (see
    `_read_recorded_episode_fields`). This is the CLI-02 resume skip-set's
    sole data source — never a listing of `data/chunk-000/` or
    `videos/chunk-000/`, which cannot distinguish a completed episode from
    crash debris (D-01, RESEARCH.md Pitfall 6).

    `source_sha256` is a required field at every schema version — a record
    missing it is corrupt, so this asserts it non-`None` (naming the
    episode index) rather than silently narrowing `str | None` to `str`."""
    fields = _read_recorded_episode_fields(Path(out_dir))
    result: dict[int, str] = {}
    for index, entry in fields.items():
        source_sha256 = entry["source_sha256"]
        assert source_sha256 is not None, (
            f"episode {index}'s provenance record is missing source_sha256 — corrupt record"
        )
        result[index] = source_sha256
    return result


def read_recorded_dispositions(out_dir: str | Path) -> dict[int, str]:
    """`{episode_index: disposition}` (the raw `Disposition` string value,
    e.g. `"PASS"`/`"PASS_WITH_FLAGS"`/`"REVIEW"`/`"REJECT"`) for every
    per-episode provenance record already recorded under `out_dir` — same
    selection rule as `read_recorded_source_hashes`.

    `disposition` is a required field at every schema version — a record
    missing it is corrupt, so this asserts it non-`None` (naming the
    episode index) rather than silently narrowing `str | None` to `str`.

    NOT the resume/retry written-ness signal as of 04-09 — see
    `data/cli.py::_episode_written_map`'s own docstring for why written-ness
    is decided from `ingest_outcome` (via `read_recorded_ingest_outcomes`),
    never from this function's return value."""
    fields = _read_recorded_episode_fields(Path(out_dir))
    result: dict[int, str] = {}
    for index, entry in fields.items():
        disposition = entry["disposition"]
        assert disposition is not None, (
            f"episode {index}'s provenance record is missing disposition — corrupt record"
        )
        result[index] = disposition
    return result


def read_recorded_ingest_outcomes(out_dir: str | Path) -> dict[int, str | None]:
    """`{episode_index: ingest_outcome}` (the raw `IngestOutcome` string
    value — `"WRITTEN"` or `"REJECTED_BEFORE_WRITE"` — or `None`) for every
    per-episode provenance record already recorded under `out_dir` — same
    selection rule as `read_recorded_source_hashes`.

    A `None` value means the record predates `RECORD_SCHEMA_VERSION` 2 (see
    that constant's own docstring in `data/contracts/containers.py`); the
    caller must derive written-ness from artifact presence for that index
    rather than treating the absence as either outcome."""
    fields = _read_recorded_episode_fields(Path(out_dir))
    return {index: entry["ingest_outcome"] for index, entry in fields.items()}


def read_recorded_supersessions(out_dir: str | Path) -> dict[int, int | None]:
    """`{episode_index: superseded_by_episode_index}` for every per-episode
    provenance record already recorded under `out_dir` — same selection
    rule as `read_recorded_source_hashes` (04-11).

    `None` means the record is either live (never superseded) or predates
    this field's introduction — both read identically as "not superseded".
    A non-`None` value names the LIVE episode index a successful
    `--retry-rejected` re-conversion of the same input wrote its record at;
    the index this function's key names is dead evidence at that point —
    see `data/cli.py::_episode_written_map`'s own docstring for how that
    omission propagates."""
    fields = _read_recorded_episode_fields(Path(out_dir))
    return {index: entry["superseded_by_episode_index"] for index, entry in fields.items()}


def evaluate_prefix(
    out_dir: str | Path,
    *,
    pipeline_version: str,
    profile_hash_value: str,
    format_key: str,
) -> PrefixDecision:
    """CLI-08's never-write-in-place promotion guard.

    Performs no filesystem write of any kind — no directory creation, no
    temp file, no rename — proven by observation in
    `data/tests/test_cli.py::test_evaluate_prefix_writes_nothing` rather
    than by reading this function's body alone.

    A prefix with no `meta/data_engine/` directory (including one that does
    not exist at all) is `FRESH`. Otherwise, this run's own
    `(pipeline_version, profile_hash_value, format_key)` is compared
    against whatever is already recorded: `format_key` against the
    manifest alone (a per-episode record carries no `format_key` field);
    `pipeline_version` and `profile_hash` are cross-checked between the
    manifest (when present) and every per-episode record first — a
    disagreement between those two sources on a field both carry is
    `STATE_ERROR`, naming the field, both values, and the episode index the
    record came from, never silently resolved in favour of either source.
    When no manifest exists but per-episode records do (a Phase-3 fixture
    prefix, for instance), the records alone are the comparison basis.
    Every comparison is `==` on the exact string as read from disk — no
    case-folding, whitespace-trimming, or Unicode-normalizing transform is
    ever applied to either side.
    """
    out_dir = Path(out_dir)
    data_engine_dir = out_dir / "meta" / "data_engine"
    if not data_engine_dir.exists():
        return PrefixDecision(
            verdict=PrefixVerdict.FRESH,
            reason=f"no meta/data_engine/ directory under {out_dir} — treated as fresh",
            mismatched_field=None,
            expected=None,
            found=None,
            recorded_source_sha256s=(),
        )

    manifest = read_run_manifest(out_dir)
    records = _read_recorded_episode_fields(out_dir)
    recorded_source_sha256s = tuple(
        sorted(
            {
                sha256
                for entry in records.values()
                if (sha256 := entry["source_sha256"]) is not None
            }
        )
    )

    if manifest is not None:
        for field in ("pipeline_version", "profile_hash"):
            manifest_value = manifest[field]
            for episode_index in sorted(records):
                record_value = records[episode_index][field]
                if record_value != manifest_value:
                    return PrefixDecision(
                        verdict=PrefixVerdict.STATE_ERROR,
                        reason=(
                            f"manifest's recorded {field} ({manifest_value!r}) "
                            f"disagrees with episode {episode_index}'s recorded "
                            f"{field} ({record_value!r})"
                        ),
                        mismatched_field=field,
                        expected=manifest_value,
                        found=record_value,
                        recorded_source_sha256s=recorded_source_sha256s,
                    )

    comparisons: list[tuple[str, str, str]] = []
    if manifest is not None:
        comparisons.append(("format_key", manifest["format_key"], format_key))
        comparisons.append(("pipeline_version", manifest["pipeline_version"], pipeline_version))
        comparisons.append(("profile_hash", manifest["profile_hash"], profile_hash_value))
    elif records:
        any_record = records[next(iter(sorted(records)))]
        # pipeline_version/profile_hash are required fields at every schema
        # version (unlike ingest_outcome) — a record missing either is
        # corrupt, so this asserts non-None rather than silently narrowing.
        any_record_pipeline_version = any_record["pipeline_version"]
        any_record_profile_hash = any_record["profile_hash"]
        assert any_record_pipeline_version is not None, "corrupt record: missing pipeline_version"
        assert any_record_profile_hash is not None, "corrupt record: missing profile_hash"
        comparisons.append(
            ("pipeline_version", any_record_pipeline_version, pipeline_version)
        )
        comparisons.append(("profile_hash", any_record_profile_hash, profile_hash_value))

    for field, recorded_value, this_run_value in comparisons:
        if recorded_value != this_run_value:
            return PrefixDecision(
                verdict=PrefixVerdict.REFUSE,
                reason=(
                    f"prefix {out_dir} carries a recorded {field} "
                    f"({recorded_value!r}) that differs from this run's "
                    f"({this_run_value!r})"
                ),
                mismatched_field=field,
                expected=recorded_value,
                found=this_run_value,
                recorded_source_sha256s=recorded_source_sha256s,
            )

    return PrefixDecision(
        verdict=PrefixVerdict.RESUME,
        reason="every recorded field this run could compare against matches",
        mismatched_field=None,
        expected=None,
        found=None,
        recorded_source_sha256s=recorded_source_sha256s,
    )
