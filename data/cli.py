"""CLI-01's batch-conversion entry point: `python -m data.cli <output-prefix>
--profile <path> --run-config <path>` converts a directory of Galaxea R1 Pro
MCAP recordings into one LeRobot v2.1 dataset with no manual step in
between.

Wires the loop `data/tests/mutators.py::build_varying_baseline_dataset`
already proves end to end — `GalaxeaInputAdapter.parse` ->
`LeRobotV21Writer.write_episode` -> `ProvenanceRecorder.record` ->
`writer.finalize()` -> `validate_dataset` — productionized here with the
CLI-08 never-write-in-place prefix guard (run first, before any write), a
CLI-06 run-level provenance manifest, a CLI-09 content-derived dataset
revision identifier, and a per-run report (all `data/promotion.py`).

Two concurrent runs into one prefix are out of scope (single-process
sequential per PROJECT.md) — this exact sentence is also `build_parser`'s
`epilog`, since argparse never surfaces this module docstring to `--help`.
"""

import argparse
import dataclasses
import datetime as dt
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from data.adapters.checks import CheckRegistry, FatalCheckContext
from data.adapters.galaxea import (
    GalaxeaAdapterError,
    GalaxeaInputAdapter,
    _find_master_clock_camera,
    _plan_decode_requests,
    _run_index_pass,
)
from data.contracts.containers import CanonicalEpisode, CheckResult, Disposition, IngestOutcome
from data.contracts.profile import EmbodimentProfile, load_profile, profile_hash
from data.promotion import (
    RUN_REPORTS_DIRNAME,
    PrefixDecision,
    PrefixVerdict,
    RunManifestError,
    append_run_to_manifest,
    dataset_revision_id,
    evaluate_prefix,
    read_recorded_ingest_outcomes,
    read_recorded_source_hashes,
    read_recorded_supersessions,
    read_run_manifest,
    write_run_report,
)
from data.provenance.recorder import PIPELINE_VERSION, ProvenanceRecorder, _hash_file_sha256, _resolve_git_commit
from data.run_config import RunConfig, load_run_config
from data.validation.constant_channels import CHECK_NAME as CONSTANT_CHANNEL_CHECK_NAME
from data.validation.registry import UNEVALUATED_KEY
from data.validation.validate import ValidationOutcome, validate_dataset
from data.writers.lerobot_v21 import (
    _CHUNK_INDEX,
    _DATA_PATH_TEMPLATE,
    _VIDEO_PATH_TEMPLATE,
    ENCODE_MANIFEST_DIRNAME,
    get_writer,
)

EXIT_CLEAN = 0
"""Everything discovered converted (or was legitimately resumed-skipped) and
validation, if run, fully passed."""
EXIT_VALIDATION_FAILED = 1
"""`validate_dataset` reported a fatal failure (`ValidationOutcome.exit_code
!= 0`)."""
EXIT_PREFIX_REFUSED = 2
"""The CLI-08 promotion guard refused a mismatched prefix — nothing was
written (Task 3)."""
EXIT_PREFIX_STATE_ERROR = 3
"""The run manifest disagrees with its own per-episode provenance records —
nothing was written (Task 3)."""
EXIT_COMPLETED_WITH_FINDINGS = 4
"""Completed, but `ValidationOutcome.fully_validated` is `False` and/or one
or more inputs were rejected — never conflated with `EXIT_CLEAN`."""
EXIT_CONSTANT_CHANNELS_UNACCEPTED = 5
"""G-03: the run completed and wrote its artifacts, but
`statistics.constant_channel` (advisory as of G-03) flagged a finding that
was neither accepted by `--accept-constant-channels` nor accepted at an
interactive prompt — so the run must not be read as clean or even as a
routine completed-with-findings run. Outranked only by a non-zero
`validate_dataset` exit code (`EXIT_VALIDATION_FAILED`); outranks
`EXIT_COMPLETED_WITH_FINDINGS`."""

LEDGER_BUCKETS = (
    "converted",
    "resumed_skip",
    "rejected_skip",
    "reclaimed",
    "duplicate_source",
    "newly_rejected",
)
"""The six dispositions one discovered input can land in across this plan
and 04-02. Task 1 exercises only `converted` and `newly_rejected`; the
remaining four are reserved for Task 2 (resumability, duplicate collapsing)
and Task 3 (the promotion guard)."""

WRITE_UNEXPECTED_FAILURE_CHECK_NAME = "writer.unexpected_write_failure"
"""The one place this `CheckResult.check_name` literal is spelled (WR-02,
04-11): an unexpected exception out of `writer.write_episode` (an encoder
fault, a corrupt JPEG, an unsupported frame shape) degrades that ONE input
to a `newly_rejected` ledger entry carrying a `CheckResult` under this
name — never a bare exception propagating out and losing the rest of the
batch's report."""

_REVISION_PREFIX_NAME_RE = re.compile(r"rev-[0-9a-f]{8}")
"""Matches a prefix directory basename shaped exactly like
`dataset_revision_id`'s own short form (`data/promotion.py`) — `"rev-"`
plus 8 lowercase hex characters, fullmatch only. WINDOWS.md id 12: the
documented workstation procedure names a prefix from a dry run's
PREDICTED revision before the real content exists, so this is reported
(never refused, never used to rename anything) when the basename disagrees
with the revision this run actually computed."""

_CONCURRENCY_SCOPE_SENTENCE = (
    "two concurrent runs into one prefix are out of scope (single-process "
    "sequential per PROJECT.md)"
)

CONSTANT_CHANNEL_ACCEPTANCE_FLAG = "--accept-constant-channels"
"""The flag's literal spelling, held once so it is never re-spelled in help
text, the stderr message, or a test."""

ACCEPTANCE_MODES = (
    "not-evaluated",
    "no-finding",
    "flag",
    "prompt-accepted",
    "prompt-declined",
    "unattended-no-flag",
)
"""Every situation `resolve_constant_channel_acceptance` can produce,
inspectable in one place the way this project's check registries are:
`not-evaluated` (validation did not run at all, e.g. `--no-validate`),
`no-finding` (validation ran and the check reported nothing to consent to —
either it passed, or it carried the unevaluated marker), `flag` (an
explicit `--accept-constant-channels` short-circuited the prompt),
`prompt-accepted`/`prompt-declined` (an interactive tty answered), and
`unattended-no-flag` (no tty, no flag — declined without ever prompting)."""

_CONSTANT_CHANNEL_TRAINING_RISK = (
    "GR00T's default min-max normalization divides observed values by "
    "(max - min); a channel whose merged min equals its max divides by "
    "zero. Accepting this finding authorizes proceeding with that hazard "
    "rather than removing it — the fix is editing the profile key named in "
    "each flagged channel's evidence (state_vector_order / action_primary)."
)


@dataclasses.dataclass(frozen=True)
class ConstantChannelAcceptance:
    """G-03: the recorded outcome of `resolve_constant_channel_acceptance`,
    written into every run's report on EVERY run — including a run with no
    finding and a run that skipped validation — so a reader never has to
    infer acceptance from an absent key.

    `accepted` is `None` — never `False` — when there was nothing to
    accept (`not-evaluated`/`no-finding`), mirroring the `match: bool |
    None` convention `data/tools/verify_gr00t_run.py`'s width report
    already established for an unrun comparison."""

    finding_present: bool
    accepted: bool | None
    mode: str
    channels: tuple[str, ...]
    flag: str
    training_risk: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "finding_present": self.finding_present,
            "accepted": self.accepted,
            "mode": self.mode,
            "channels": list(self.channels),
            "flag": self.flag,
            "training_risk": self.training_risk,
        }


def _no_consent_needed_acceptance(mode: str) -> ConstantChannelAcceptance:
    """Shared by the `not-evaluated` and `no-finding` modes — both carry
    `finding_present=False`/`accepted=None`/empty `channels`, differing only
    in which situation produced them."""
    return ConstantChannelAcceptance(
        finding_present=False,
        accepted=None,
        mode=mode,
        channels=(),
        flag=CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
        training_risk=_CONSTANT_CHANNEL_TRAINING_RISK,
    )


def constant_channel_finding(validation: ValidationOutcome | None) -> CheckResult | None:
    """The single dataset-scoped `CheckResult` whose name matches
    `data.validation.constant_channels.CHECK_NAME` (imported, never a
    copied string), or `None` when validation did not run or no such
    result is present. A result carrying the unevaluated marker counts as
    no finding: there is nothing to consent to when the check had no basis
    for a verdict, and the existing `fully_validated` path already
    surfaces that abstention."""
    if validation is None:
        return None
    for result in validation.dataset_results:
        if result.check_name != CONSTANT_CHANNEL_CHECK_NAME:
            continue
        if result.evidence.get(UNEVALUATED_KEY, False):
            return None
        return result
    return None


def resolve_constant_channel_acceptance(
    finding: CheckResult | None,
    *,
    invoked: bool,
    accept_flag: bool,
    stdin_isatty: bool,
    prompt: Callable[[str], str] | None,
) -> ConstantChannelAcceptance:
    """Mirrors `data/writers/lerobot_v21.py::resolve_task_string`'s seam
    exactly: `stdin_isatty` is a plain boolean the caller resolves, and
    `prompt` is an optional callable taking the message and returning the
    answer, so both branches are directly testable with no terminal.

    The flag short-circuits the prompt — an operator who already typed it
    is not asked again. Under a tty with no flag, prompts once with a
    message naming every flagged channel and accepts only an explicit
    affirmative answer; anything else, including an empty answer, is
    declined, because a bare Enter must not authorize training on a
    zero-variance channel. With no tty and no flag, never prompts and
    never raises: returns declined with the `unattended-no-flag` mode, so
    the exit code carries the refusal through the same structured path as
    every other finding.
    """
    if not invoked:
        return _no_consent_needed_acceptance("not-evaluated")
    if finding is None or finding.passed:
        return _no_consent_needed_acceptance("no-finding")

    channels = tuple(
        sorted(
            f"{fc['feature']}.{fc['joint_name']}"
            for fc in finding.evidence.get("flagged_channels", [])
        )
    )

    if accept_flag:
        return ConstantChannelAcceptance(
            finding_present=True,
            accepted=True,
            mode="flag",
            channels=channels,
            flag=CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
            training_risk=_CONSTANT_CHANNEL_TRAINING_RISK,
        )

    if stdin_isatty:
        if prompt is None:
            raise ValueError(
                "stdin is a TTY but no interactive prompt callable was supplied"
            )
        message = (
            "The following channels are constant across this dataset: "
            f"{', '.join(channels)}. {_CONSTANT_CHANNEL_TRAINING_RISK} "
            "Accept and proceed? [y/N] "
        )
        answer = prompt(message)
        accepted = answer.strip().lower() in ("y", "yes")
        return ConstantChannelAcceptance(
            finding_present=True,
            accepted=accepted,
            mode="prompt-accepted" if accepted else "prompt-declined",
            channels=channels,
            flag=CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
            training_risk=_CONSTANT_CHANNEL_TRAINING_RISK,
        )

    return ConstantChannelAcceptance(
        finding_present=True,
        accepted=False,
        mode="unattended-no-flag",
        channels=channels,
        flag=CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
        training_risk=_CONSTANT_CHANNEL_TRAINING_RISK,
    )


@dataclasses.dataclass(frozen=True)
class InputLedgerEntry:
    """One discovered input's disposition in this run's every-input-
    accounted ledger (CLI-05)."""

    source_path: str
    source_sha256: str
    bucket: str
    episode_index: int | None
    frame_count: int | None
    disposition: str | None
    checks: tuple[CheckResult, ...]
    elapsed_seconds: float
    deleted_paths: tuple[str, ...] = ()
    """Populated only for `bucket == "reclaimed"` — every path deleted
    while reclaiming that crash-debris episode index (D-03). Empty for
    every other bucket."""
    decode_seconds: float = 0.0
    encode_seconds: float = 0.0
    """Populated only for `bucket == "converted"` — the DELTA of the
    writer's cumulative `decode_seconds`/`encode_seconds` properties across
    exactly this input's own `write_episode` call (never the raw cumulative
    reading — see `LeRobotV21Writer.decode_seconds`'s own docstring). `0.0`
    for every other bucket, since none of them ever calls `write_episode`."""


@dataclasses.dataclass(frozen=True)
class ConvertOutcome:
    """Everything one `run_convert` call produced. `prefix_decision` is
    `None` only when `evaluate_prefix` was never reached (it always is, in
    the current implementation) — populated for every real verdict,
    including `FRESH`/`RESUME` runs that go on to convert."""

    exit_code: int
    ledger: tuple[InputLedgerEntry, ...]
    dataset_revision: tuple[str, str] | None
    manifest_path: Path | None
    report_path: Path | None
    validation: ValidationOutcome | None
    prefix_decision: PrefixDecision | None
    constant_channel_acceptance: ConstantChannelAcceptance | None = None
    """`None` only for the pre-write REFUSE/STATE_ERROR returns, where
    `validate_dataset` is never reached. Populated for every real
    conversion attempt, including a `--no-validate` run (mode
    `not-evaluated`)."""


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_pinned_package_versions() -> dict[str, str]:
    """Parses `data/requirements.txt`'s pinned `name==version` lines into a
    `{name: version}` mapping for the run manifest — never re-derived from
    `pip freeze` or an installed-distribution query, so the recorded
    versions always match this repo's own pinned list."""
    req_path = Path(__file__).resolve().parent / "requirements.txt"
    versions: dict[str, str] = {}
    for line in req_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, version = stripped.partition("==")
        versions[name] = version
    return versions


def _scan_recorded_source_hashes(out_dir: Path, *, emitted_video_keys: list[str]) -> set[str]:
    """Every `source_sha256` `dataset_revision_id` should hash: exactly the
    hashes of episodes `_episode_written_map` reports as written (04-09) —
    quoting that function's own contract, "every converted source
    recording's sha256" (`data/promotion.py::dataset_revision_id`'s own
    docstring). Derived from `_episode_written_map`'s written entries only —
    never by filtering `read_recorded_source_hashes`' full mapping through a
    defaulted lookup, which would silently admit an omitted (04-11
    superseded) index once that state exists."""
    written = _episode_written_map(out_dir, emitted_video_keys=emitted_video_keys)
    hashes_by_index = read_recorded_source_hashes(out_dir)
    return {
        hashes_by_index[index]
        for index, is_written in written.items()
        if is_written and index in hashes_by_index
    }


def _orphaned_report_paths(out_dir: Path) -> list[str]:
    """WR-01 (04-11): every recorded `runs[].report_path` in the manifest
    whose file does not exist on disk — the crash window between
    `append_run_to_manifest` and `write_run_report` leaves exactly this:
    the manifest entry survives, its report does not. Returns the sorted
    list of missing paths — empty for a fresh prefix or a prefix with no
    manifest yet. Never repairs or backfills anything; detection only.

    A recorded path that no longer exists at its own location is checked a
    SECOND way before being reported: by its own basename, resolved under
    THIS prefix's `out_dir / "meta" / "data_engine" / RUN_REPORTS_DIRNAME`
    directory. WINDOWS.md id 12 records a real prefix that was renamed out
    of band — its manifest still carries the OLD absolute path, which is
    merely STALE (the report file moved along with its directory), never
    orphaned (the file itself was never lost) — and must not be reported
    as missing."""
    manifest = read_run_manifest(out_dir)
    if manifest is None:
        return []
    reports_dir = out_dir / "meta" / "data_engine" / RUN_REPORTS_DIRNAME
    missing: list[str] = []
    for run in manifest["runs"]:
        recorded_path = Path(run["report_path"])
        if recorded_path.exists():
            continue
        if recorded_path.is_absolute() and (reports_dir / recorded_path.name).exists():
            continue
        missing.append(run["report_path"])
    return sorted(missing)


def _serialize_check(check: CheckResult) -> dict[str, Any]:
    return {
        "check_name": check.check_name,
        "severity": check.severity,
        "passed": check.passed,
        "evidence": dict(check.evidence),
    }


def _count_unevaluated(validation: ValidationOutcome) -> int:
    """Mirrors `data/validation/validate.py::_is_unevaluated`'s own
    convention: a `CheckResult` whose evidence carries `UNEVALUATED_KEY`
    true had no basis for a verdict — never counted as a pass, never as a
    fatal failure, and here counted explicitly so the run report surfaces
    the "checked and passed" vs. "not checked" distinction rather than
    leaving a reader to re-derive it by scanning every check's evidence."""
    count = 0
    for c in validation.dataset_results:
        if c.evidence.get(UNEVALUATED_KEY, False):
            count += 1
    for results in validation.episode_results.values():
        for c in results:
            if c.evidence.get(UNEVALUATED_KEY, False):
                count += 1
    return count


def _serialize_validation_outcome(
    validation: ValidationOutcome | None, *, invoked: bool
) -> dict[str, Any]:
    """The run report's `validation` block (D-09/D-12): `invoked` records
    whether `validate_dataset` ran at all (`--no-validate` skips it, but
    that must be visible in the report, never indistinguishable from "ran
    and found nothing to flag"). When `validation` is `None` (not invoked),
    every other field is `None`/empty rather than a fabricated passing
    value."""
    if validation is None:
        return {
            "invoked": invoked,
            "fully_validated": None,
            "exit_code": None,
            "unevaluated_count": None,
            "dataset_results": [],
            "episode_results": {},
        }
    return {
        "invoked": invoked,
        "fully_validated": validation.fully_validated,
        "exit_code": validation.exit_code,
        "unevaluated_count": _count_unevaluated(validation),
        "dataset_results": [_serialize_check(c) for c in validation.dataset_results],
        "episode_results": {
            str(idx): [_serialize_check(c) for c in results]
            for idx, results in sorted(validation.episode_results.items())
        },
    }


def _derive_exit_code(
    ledger: tuple[InputLedgerEntry, ...],
    validation: ValidationOutcome | None,
    acceptance: ConstantChannelAcceptance,
) -> int:
    """The documented exit-code precedence (Claude's Discretion item 5,
    04-01-PLAN.md, extended by G-03): the two pre-write codes (2, 3)
    short-circuit before this function is ever called (Task 3 wires
    those); here the order is `1` beats `5` beats `4` beats `0`.

    A non-zero `validate_dataset` exit code wins outright — a fatal
    validation failure outranks an unaccepted constant-channel finding.
    Next, an unaccepted finding (`acceptance.finding_present and not
    acceptance.accepted`) drives `EXIT_CONSTANT_CHANNELS_UNACCEPTED`. Next,
    completed-with-findings covers a rejected input, an unevaluated
    validation result, OR an ACCEPTED constant-channel finding — an
    accepted finding never returns clean either, because a run that
    trained-risk-accepted a zero-variance channel is not a clean run. A
    `PASS_WITH_FLAGS` disposition alone never changes this — only
    `ValidationOutcome.fully_validated`, per-input rejection, and the
    acceptance state do."""
    if validation is not None and validation.exit_code != 0:
        return EXIT_VALIDATION_FAILED
    if acceptance.finding_present and not acceptance.accepted:
        return EXIT_CONSTANT_CHANNELS_UNACCEPTED
    any_rejected = any(e.bucket in ("rejected_skip", "newly_rejected") for e in ledger)
    if (
        any_rejected
        or (validation is not None and not validation.fully_validated)
        or acceptance.finding_present
    ):
        return EXIT_COMPLETED_WITH_FINDINGS
    return EXIT_CLEAN


def _assert_ledger_complete(ledger: tuple[InputLedgerEntry, ...], discovered_count: int) -> None:
    """CLI-05's every-input-accounted rule: the ledger must carry exactly
    one entry per discovered input, checked immediately before the run
    report is written. Raises `ValueError` naming both numbers — an
    incomplete ledger must never reach disk, because that is what CLI-05's
    guarantee buys.

    `reclaimed` entries are excluded from this count — each accounts for a
    crash-debris episode INDEX (D-03), not a discovered INPUT, so they never
    count toward or against per-input completeness."""
    per_input_count = sum(1 for e in ledger if e.bucket != "reclaimed")
    if per_input_count != discovered_count:
        raise ValueError(
            f"ledger accounts for {per_input_count} input(s) but {discovered_count} "
            "were discovered — refusing to write an incomplete run report"
        )


def _discover_artifact_indices(
    out_dir: Path, emitted_video_keys: list[str]
) -> set[int]:
    """Every episode index with at least one on-disk artifact under
    `data/chunk-000/`, any emitted camera's `videos/chunk-000/<video_key>/`,
    `meta/{ENCODE_MANIFEST_DIRNAME}/`, or `meta/stats_partials/` — the
    candidate set crash-debris reclamation (D-03) checks against the
    provenance-record index set. Directories that do not exist yet
    contribute nothing rather than raising."""
    indices: set[int] = set()

    parquet_dir = (
        out_dir / _DATA_PATH_TEMPLATE.format(episode_chunk=_CHUNK_INDEX, episode_index=0)
    ).parent
    if parquet_dir.exists():
        for path in parquet_dir.glob("episode_*.parquet"):
            try:
                indices.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue

    for video_key in emitted_video_keys:
        video_dir = (
            out_dir
            / _VIDEO_PATH_TEMPLATE.format(
                episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=0
            )
        ).parent
        if not video_dir.exists():
            continue
        for path in video_dir.glob("episode_*.mp4"):
            try:
                indices.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue

    for dirname in (ENCODE_MANIFEST_DIRNAME, "stats_partials"):
        candidate_dir = out_dir / "meta" / dirname
        if not candidate_dir.exists():
            continue
        for path in candidate_dir.glob("episode_*.json"):
            try:
                indices.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue

    return indices


def _reclaim_crash_debris(
    out_dir: Path, *, emitted_video_keys: list[str], recorded_indices: set[int]
) -> list[InputLedgerEntry]:
    """D-03: for every episode index carrying at least one on-disk artifact
    but no `meta/data_engine/episode_{NNNNNN}.json` provenance record,
    deletes that index's parquet, every emitted camera's MP4, its encode
    manifest and its stats partial — never touching an index that DOES have
    a record — then returns one `reclaimed` ledger entry per reclaimed
    index, naming every path deleted (never silent). Every artifact path is
    derived from the writer module's own `_DATA_PATH_TEMPLATE`/
    `_VIDEO_PATH_TEMPLATE`/`ENCODE_MANIFEST_DIRNAME` constants, never
    re-spelled."""
    debris_indices = sorted(_discover_artifact_indices(out_dir, emitted_video_keys) - recorded_indices)

    entries: list[InputLedgerEntry] = []
    for idx in debris_indices:
        deleted: list[str] = []

        parquet_path = out_dir / _DATA_PATH_TEMPLATE.format(
            episode_chunk=_CHUNK_INDEX, episode_index=idx
        )
        if parquet_path.exists():
            parquet_path.unlink()
            deleted.append(str(parquet_path))

        for video_key in emitted_video_keys:
            video_path = out_dir / _VIDEO_PATH_TEMPLATE.format(
                episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=idx
            )
            if video_path.exists():
                video_path.unlink()
                deleted.append(str(video_path))

        manifest_path = out_dir / "meta" / ENCODE_MANIFEST_DIRNAME / f"episode_{idx:06d}.json"
        if manifest_path.exists():
            manifest_path.unlink()
            deleted.append(str(manifest_path))

        partial_path = out_dir / "meta" / "stats_partials" / f"episode_{idx:06d}.json"
        if partial_path.exists():
            partial_path.unlink()
            deleted.append(str(partial_path))

        entries.append(
            InputLedgerEntry(
                source_path=f"<crash-debris episode_index={idx}>",
                source_sha256="",
                bucket="reclaimed",
                episode_index=idx,
                frame_count=None,
                disposition=None,
                checks=(),
                elapsed_seconds=0.0,
                deleted_paths=tuple(deleted),
            )
        )
    return entries


def _sweep_write_temp_debris(out_dir: Path, *, emitted_video_keys: list[str]) -> list[str]:
    """WR-04 (04-11): deletes dot-prefixed write-in-progress temp files an
    interrupted encode or parquet/record write can leave behind —
    `.episode_*.parquet.tmp` under `data/chunk-000/`, `.episode_*.mp4.tmp`
    under each emitted camera's video directory, and `.episode_*.json.tmp`
    under `meta/{ENCODE_MANIFEST_DIRNAME}/`, `meta/stats_partials/`, and
    `meta/data_engine/` — every directory derived from the writer module's
    own `_DATA_PATH_TEMPLATE`/`_VIDEO_PATH_TEMPLATE`/`ENCODE_MANIFEST_DIRNAME`
    constants exactly as `_reclaim_crash_debris` above already does, never
    re-spelled. Returns the sorted list of deleted paths — empty when
    nothing matched. A directory that does not exist yet contributes
    nothing rather than raising.

    Restricted to the `episode_*` prefix, deliberately: the dataset-wide
    metadata temps (`.info.json.tmp`, `.stats.json.tmp`, `.tasks.jsonl.tmp`,
    `.episodes.jsonl.tmp`, `.validation_report.json.tmp`,
    `.run_manifest.json.tmp`, `.run_NNN.json.tmp`) are each written and
    replaced within a single `finalize()`/manifest call — sweeping them
    buys nothing and risks deleting a temp another process is mid-write.

    Deliberately NOT folded into `_discover_artifact_indices`: 04-09 made
    that function the v1 back-compat written-ness derivation, so counting
    a `.tmp` file as an artifact there would make a pre-schema-2
    ingest-rejected record derive as WRITTEN — reintroducing the exact
    class of defect this batch exists to close.
    """
    deleted: list[str] = []

    parquet_dir = (
        out_dir / _DATA_PATH_TEMPLATE.format(episode_chunk=_CHUNK_INDEX, episode_index=0)
    ).parent
    if parquet_dir.exists():
        for path in parquet_dir.glob(".episode_*.parquet.tmp"):
            path.unlink()
            deleted.append(str(path))

    for video_key in emitted_video_keys:
        video_dir = (
            out_dir
            / _VIDEO_PATH_TEMPLATE.format(
                episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=0
            )
        ).parent
        if not video_dir.exists():
            continue
        for path in video_dir.glob(".episode_*.mp4.tmp"):
            path.unlink()
            deleted.append(str(path))

    for dirname in (ENCODE_MANIFEST_DIRNAME, "stats_partials", "data_engine"):
        candidate_dir = out_dir / "meta" / dirname
        if not candidate_dir.exists():
            continue
        for path in candidate_dir.glob(".episode_*.json.tmp"):
            path.unlink()
            deleted.append(str(path))

    return sorted(deleted)


def _episode_written_map(
    out_dir: Path, *, emitted_video_keys: list[str]
) -> dict[int, bool]:
    """The ONLY place written-ness is decided (04-09): `{episode_index:
    True-if-written}` for every episode index `read_recorded_source_hashes`
    reports — EXCEPT an index `read_recorded_supersessions` reports as
    superseded (04-11), which is OMITTED from the returned map entirely.

    Maps to `True` when the recorded `ingest_outcome` is exactly
    `IngestOutcome.WRITTEN.value`, `False` when it is exactly
    `IngestOutcome.REJECTED_BEFORE_WRITE.value`, and otherwise (absent,
    JSON `null`, or any unrecognized string — the v1 back-compat case, see
    `RECORD_SCHEMA_VERSION`'s own docstring) falls back to `index in
    _discover_artifact_indices(out_dir, emitted_video_keys)` — reusing that
    existing crash-debris helper, never re-implementing an artifact scan.

    The recorded-disposition reader is deliberately not on this path: a
    written episode that post-write validation later amended to
    `Disposition.REJECT` must still read `True` here, because that amend
    (`ProvenanceRecorder.amend_checks`) never touches `ingest_outcome`.

    A superseded index is dead evidence (04-11): the input it recorded was
    a rejection that a later `--retry-rejected` run converted successfully
    at ANOTHER index, so this dead index is neither converted (its
    artifacts never existed) nor retryable (its input already converted
    elsewhere) — it belongs in NEITHER of `_build_resume_maps`' two
    buckets. Omitting it here, rather than mapping it to `True` or `False`,
    is what makes that omission propagate by construction (see
    `_build_resume_maps`' own docstring for why the omission must never be
    papered over with a defaulted lookup downstream).
    """
    ingest_outcomes_by_index = read_recorded_ingest_outcomes(out_dir)
    supersessions_by_index = read_recorded_supersessions(out_dir)
    artifact_indices = _discover_artifact_indices(out_dir, emitted_video_keys)
    written: dict[int, bool] = {}
    for index, raw_outcome in ingest_outcomes_by_index.items():
        if supersessions_by_index.get(index) is not None:
            continue
        if raw_outcome == IngestOutcome.WRITTEN.value:
            written[index] = True
        elif raw_outcome == IngestOutcome.REJECTED_BEFORE_WRITE.value:
            written[index] = False
        else:
            written[index] = index in artifact_indices
    return written


def _build_resume_maps(
    out_dir: Path, *, emitted_video_keys: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """`(converted_hash_to_index, rejected_hash_to_index)` — the resume
    skip set (D-01/D-04), built by iterating `_episode_written_map`'s items
    and looking each index's hash up in `read_recorded_source_hashes`'
    mapping — never the reverse, and never a directory listing.

    **Iteration source is load-bearing for 04-11.** Before 04-11, the two
    key sets (`_episode_written_map`'s and `read_recorded_source_hashes`')
    were always identical, so iterating either side was behaviorally a
    no-op. 04-11 makes `_episode_written_map` OMIT superseded indices,
    which means the two key sets diverge whenever any record is
    superseded (exactly the scenario `mark_superseded` exists to create) —
    which is why iteration order is load-bearing starting now, not
    eventually. This introduces a third state (absent) that two-way
    bucketing has no rule for. Iterating the
    written map means an omitted index lands in neither bucket by
    construction, with no default value deciding it — never iterate
    `read_recorded_source_hashes` and look up written-ness with a defaulted
    `.get(index, False)` (which would publish a dead, superseded index in
    `rejected_hash_to_index` as though it were a live retry target), and
    never index `written_by_index[index]` directly off that loop either
    (which would `KeyError` on an omitted index).

    `rejected_hash_to_index` now means "recorded but never written" — which
    is exactly what `rejected_skip` and `--retry-rejected` have always
    meant. Shared by `run_convert` and `run_dry_run` (D-10) so the dry run's
    reported skip list is computed by the exact same logic a real run would
    use to skip."""
    written_by_index = _episode_written_map(out_dir, emitted_video_keys=emitted_video_keys)
    recorded_source_hashes_by_index = read_recorded_source_hashes(out_dir)
    converted_hash_to_index: dict[str, int] = {}
    rejected_hash_to_index: dict[str, int] = {}
    for index, is_written in written_by_index.items():
        sha256 = recorded_source_hashes_by_index.get(index)
        if sha256 is None:
            continue
        if is_written:
            converted_hash_to_index[sha256] = index
        else:
            rejected_hash_to_index[sha256] = index
    return converted_hash_to_index, rejected_hash_to_index


def _assert_manifest_revision_recomputes(out_dir: Path) -> None:
    """The promotion invariant this plan's assumption-delta decision
    accepted: after every `append_run_to_manifest` call, the manifest's
    `dataset_revision.full` must recompute from its own recorded
    `dataset_revision.inputs`. Raises `RunManifestError` naming both
    digests when it does not — this is what goes red the instant a future
    change reintroduces a per-episode-only identity."""
    manifest = read_run_manifest(out_dir)
    if manifest is None:
        raise RunManifestError(
            f"no run manifest found at {out_dir} immediately after "
            "append_run_to_manifest — this should be unreachable"
        )
    inputs = manifest["dataset_revision"]["inputs"]
    _, recomputed_full = dataset_revision_id(
        inputs["source_sha256s"],
        inputs["profile_hash"],
        inputs["git_commit"],
        inputs["format_key"],
    )
    recorded_full = manifest["dataset_revision"]["full"]
    if recomputed_full != recorded_full:
        raise RunManifestError(
            f"manifest at {out_dir} records dataset_revision.full "
            f"{recorded_full!r}, which does not recompute from its own "
            f"recorded dataset_revision.inputs (recomputed {recomputed_full!r})"
        )


ETA_UNAVAILABLE = "n/a (< 2 episodes completed)"
"""D-08's explicit estimate-unavailable token: printed instead of a number
until at least two episodes have completed, since a single-episode
extrapolation over a batch of unknown per-episode cost is not a
measurement (RESEARCH.md, the ETA prohibition in this plan's
must-have prohibitions)."""


class ProgressReporter:
    """D-08's tty-aware stderr progress reporter — stdlib only
    (`sys.stderr`, `.isatty()`, `time.monotonic()`). One in-place-redrawn
    line per episode when `stream` is a tty; one appended, newline-terminated
    line per episode otherwise. Stdout is never touched — progress is
    exclusively a stderr concern, so machine-readable stdout output (the
    report path, the revision short form) is never interleaved with it.

    The tty-vs-not decision is made ONCE, at construction, from an
    injectable `isatty` callable (defaulting to `stream.isatty`) — never
    re-checked per line — so both branches are directly testable without
    manipulating a real terminal.
    """

    def __init__(
        self,
        total: int,
        *,
        stream: Any = None,
        isatty: Any = None,
        now: Any = None,
    ) -> None:
        self._total = total
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = (isatty if isatty is not None else self._stream.isatty)()
        self._now = now if now is not None else time.monotonic
        self._start = self._now()
        self._completed = 0
        self._frames = 0

    def update(self, *, frames_this_episode: int) -> None:
        """Call once per COMPLETED (real, non-zero-frame) episode — never
        per discovered input, since a resumed-skip/rejected-skip/duplicate
        input never touches the encoder and has nothing to report progress
        on."""
        self._completed += 1
        self._frames += frames_this_episode
        elapsed = self._now() - self._start
        fps = self._frames / elapsed if elapsed > 0 else 0.0

        if self._completed >= 2:
            per_episode_seconds = elapsed / self._completed
            remaining = max(self._total - self._completed, 0)
            eta = f"{per_episode_seconds * remaining:.1f}s"
        else:
            eta = ETA_UNAVAILABLE

        line = (
            f"episode {self._completed}/{self._total} | "
            f"frames {self._frames} | {fps:.1f} fps | eta {eta}"
        )
        if self._is_tty:
            self._stream.write(f"\r{line}")
        else:
            self._stream.write(line + "\n")
        self._stream.flush()

    def finish(self) -> None:
        """Emits the single final newline the tty branch owes (it never
        wrote one per line, only `\\r`-redraws) — a no-op for the non-tty
        branch, which already ended its last line with `\\n`."""
        if self._is_tty:
            self._stream.write("\n")
            self._stream.flush()


def discover_inputs(run_cfg: RunConfig) -> list[Path]:
    """Resolves this run's ordered list of candidate input paths — CLI-03's
    deterministic input discovery, which is what fixes `episode_index`
    assignment (CLI-08).

    When `run_cfg.inputs` is non-empty, every entry is resolved with
    `Path(...).resolve()` and the listed order is preserved verbatim —
    `input_dir`/`input_glob` are ignored entirely in that case. Otherwise,
    `run_cfg.input_dir` is resolved and every `run_cfg.input_glob` match is
    resolved and sorted by its resolved absolute POSIX path string.

    Does NOT deduplicate resolved-identical entries (e.g. a symlink beside
    its target, or a `..`-segment path resolving to an already-listed
    file) — `run_convert`'s loop collapses those into a single conversion
    plus a `duplicate_source` ledger entry, so the collapse is visible in
    the per-input accounting rather than silently absorbed here.

    Raises:
        ValueError: naming the resolved input directory and the glob
            string, when neither `inputs` nor a matching `input_dir` glob
            result names anything to convert. Also raised, naming the
            offending path and the resolved input directory, when a
            glob-discovered candidate's resolved form falls outside the
            resolved input directory — so a crafted `input_glob` (e.g. one
            containing a `..` segment) cannot reach outside the declared
            tree. This containment check applies only to the `input_dir`/
            `input_glob` branch: an explicit `inputs` list is used exactly
            as given, with no directory to be "inside" of.
    """
    if run_cfg.inputs:
        return [Path(p).resolve() for p in run_cfg.inputs]

    if run_cfg.input_dir is None:
        raise ValueError(
            "run config supplies neither 'inputs' nor 'input_dir' — nothing to convert"
        )

    input_dir_resolved = Path(run_cfg.input_dir).resolve()
    candidates = sorted(
        (p.resolve() for p in input_dir_resolved.glob(run_cfg.input_glob)),
        key=lambda p: p.as_posix(),
    )
    if not candidates:
        raise ValueError(
            f"no inputs discovered under {input_dir_resolved} matching "
            f"glob {run_cfg.input_glob!r}"
        )
    for candidate in candidates:
        if not candidate.is_relative_to(input_dir_resolved):
            raise ValueError(
                f"discovered input {candidate} resolves outside the resolved "
                f"input directory {input_dir_resolved} — refusing to convert "
                "a path a crafted glob could smuggle out of the declared tree"
            )
    return candidates


def _dry_run_one_input(path: Path, profile: EmbodimentProfile) -> dict[str, Any]:
    """D-10: reproduces `GalaxeaInputAdapter.parse()`'s own ordering —
    index pass, fatal `CheckRegistry`, then `_plan_decode_requests` — with
    NO payload decode and NO writer/recorder ever constructed. Reuses the
    adapter's own `_run_index_pass`/`_plan_decode_requests` rather than a
    second validation path, exactly as `parse()` itself does."""
    index = _run_index_pass(str(path))
    master_name, master_cam = _find_master_clock_camera(profile)
    predicted_frame_count = index.message_counts.get(master_cam.topic, 0)

    registry = CheckRegistry.default()
    fatal_ctx = FatalCheckContext(
        scan_error=index.scan_error,
        non_monotonic_topics=index.non_monotonic_topics,
        schema_encodings=tuple((s.name, s.encoding) for s in index.schemas_by_id.values()),
        master_camera_topic=master_cam.topic,
        master_camera_message_count=index.message_counts.get(master_cam.topic, 0),
    )
    fatal_results = registry.run_fatal(fatal_ctx)
    failing_fatal = [r for r in fatal_results if not r.passed]
    if failing_fatal:
        return {
            "source_path": str(path),
            "would_convert": False,
            "predicted_frame_count": predicted_frame_count,
            "resolved_action_sources": {},
            "abort_reason": ", ".join(r.check_name for r in failing_fatal),
            "failing_checks": [_serialize_check(r) for r in failing_fatal],
        }

    try:
        _state_fields, action_resolutions = _plan_decode_requests(profile, index)
    except GalaxeaAdapterError as e:
        return {
            "source_path": str(path),
            "would_convert": False,
            "predicted_frame_count": predicted_frame_count,
            "resolved_action_sources": {},
            "abort_reason": str(e),
            "failing_checks": [],
        }

    resolved_action_sources = {
        action_key: source.topic for action_key, (_field, source) in action_resolutions.items()
    }
    return {
        "source_path": str(path),
        "would_convert": True,
        "predicted_frame_count": predicted_frame_count,
        "resolved_action_sources": resolved_action_sources,
        "abort_reason": None,
        "failing_checks": [],
    }


def run_dry_run(
    out_dir: Path,
    *,
    profile: EmbodimentProfile,
    run_cfg: RunConfig,
) -> int:
    """D-10/CLI-07: a full index pass with no payload decode, no encode,
    and no filesystem write anywhere under `out_dir` — not even creating
    the output prefix directory. Constructs no writer, no recorder.
    Prints a human-readable plan to stdout (this is the dry run's entire
    purpose — informing the operator, not machine consumption) and always
    returns `EXIT_CLEAN`: a dry run reports what would happen, it never
    itself fails.

    The predicted dataset revision (04-09) now excludes any discovered
    input the index pass itself predicts will abort, so it can still
    diverge from the real revision only for an input a real run rejects or
    truncates via a check this index pass structurally cannot see — the
    G-01 camera-coverage fatal check, per 04-REVIEW.md WR-03 — meaning the
    prediction is narrowed, not exact, and WINDOWS.md id 12 stays open with
    that reduced scope."""
    out_dir = out_dir.resolve()
    profile_hash_value = profile_hash(profile)

    prefix_decision = evaluate_prefix(
        out_dir,
        pipeline_version=PIPELINE_VERSION,
        profile_hash_value=profile_hash_value,
        format_key=run_cfg.format_key,
    )
    print(f"prefix verdict: {prefix_decision.verdict.value} — {prefix_decision.reason}")

    discovered = discover_inputs(run_cfg)
    hashes_by_path: dict[Path, str] = {}
    for candidate in discovered:
        if candidate not in hashes_by_path:
            hashes_by_path[candidate] = _hash_file_sha256(candidate)

    emitted_video_keys = [
        cam.video_key for cam in profile.cameras.values() if cam.emit and cam.video_key is not None
    ]
    converted_hash_to_index, rejected_hash_to_index = _build_resume_maps(
        out_dir, emitted_video_keys=emitted_video_keys
    )

    print("resume skip list:")
    any_skip = False
    for input_path in discovered:
        sha256 = hashes_by_path[input_path]
        if sha256 in converted_hash_to_index:
            any_skip = True
            print(f"  {input_path}: resumed_skip (episode_index={converted_hash_to_index[sha256]})")
        elif sha256 in rejected_hash_to_index:
            any_skip = True
            print(f"  {input_path}: rejected_skip (episode_index={rejected_hash_to_index[sha256]})")
    if not any_skip:
        print("  (none)")

    print("per-input plan:")
    # One index pass per input — its result is reused below for the
    # predicted revision, never re-run. would_convert_hashes collects the
    # hashes of discovered inputs whose predicted verdict is
    # would_convert=True AND that are not already recorded as rejected;
    # everything else this loop computes is print-only.
    would_convert_hashes: set[str] = set()
    for input_path in discovered:
        sha256 = hashes_by_path[input_path]
        info = _dry_run_one_input(input_path, profile)
        if info["would_convert"]:
            # WR-A (04-REVIEW.md): this index-pass verdict cannot see the
            # reason a real (non-`--retry-rejected`) run already rejected
            # this exact hash (e.g. G-01 camera coverage — the same blind
            # spot WR-03 documents below for the never-before-seen case).
            # Without this check, a real run buckets it `rejected_skip` and
            # never writes it, while this hash would still get unioned into
            # `predicted_source_hashes`, diverging from what that real run
            # actually produces.
            if sha256 in rejected_hash_to_index:
                print(
                    f"  {input_path}: would convert per this index pass, but "
                    f"already recorded rejected (episode_index="
                    f"{rejected_hash_to_index[sha256]}) — a real run without "
                    f"--retry-rejected will not write it"
                )
            else:
                would_convert_hashes.add(sha256)
                print(
                    f"  {input_path}: would convert — predicted_frame_count="
                    f"{info['predicted_frame_count']}, resolved_action_sources="
                    f"{info['resolved_action_sources']}"
                )
        else:
            print(f"  {input_path}: would ABORT — {info['abort_reason']}")

    # The predicted revision unions (a) discovered inputs the index pass
    # itself predicts will convert AND are not already recorded rejected,
    # with (b) recorded hashes _scan_recorded_source_hashes reports as
    # WRITTEN — never every discovered/recorded hash disposition-blind
    # (CR-01, 04-VERIFICATION.md). This narrows the prediction, it does not
    # make it exact: WR-03 (04-REVIEW.md) — an index pass structurally
    # cannot see the G-01 camera-coverage fatal check without a real decode
    # pass, so an input this loop predicts will convert can still be
    # rejected by a real run, and vice versa is not possible here (a
    # predicted-abort input is never unioned in). WR-A (04-REVIEW.md) closes
    # the mirror-image gap: an input already recorded rejected by a check
    # this index pass cannot see is excluded above even when this pass's
    # own narrower checks would say `would_convert=True`, since a real run
    # without `--retry-rejected` will bucket it `rejected_skip` and never
    # write it. WINDOWS.md id 12 stays open with that reduced scope.
    predicted_source_hashes = sorted(
        would_convert_hashes
        | _scan_recorded_source_hashes(out_dir, emitted_video_keys=emitted_video_keys)
    )
    git_commit = _resolve_git_commit()
    revision = dataset_revision_id(
        predicted_source_hashes, profile_hash_value, git_commit, run_cfg.format_key
    )
    print(f"predicted dataset revision: {revision[0]}")

    return EXIT_CLEAN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data.cli",
        description=(
            "Converts a directory of Galaxea R1 Pro MCAP recordings into "
            "one LeRobot v2.1 dataset. See --run-config for batch knobs "
            "(input discovery, output format, codec, task string) and "
            "--profile for the embodiment mapping."
        ),
        epilog=_CONCURRENCY_SCOPE_SENTENCE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "out_dir",
        metavar="output-prefix",
        nargs="?",
        default=None,
        help=(
            "output dataset prefix; falls back to the run config's "
            "output_prefix when omitted. An error naming both is raised "
            "when neither is set."
        ),
    )
    parser.add_argument("--profile", required=True, metavar="profile-path")
    parser.add_argument("--run-config", required=True, metavar="run-config-path")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "skip the post-finalize validate_dataset call (D-12's default "
            "is to run it)"
        ),
    )
    parser.add_argument(
        "--use-local-golden",
        action="store_true",
        help=(
            "forwarded to validate_dataset's use_local_golden — never "
            "enabled implicitly (Phase 3 D-08)"
        ),
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help=(
            "re-parse an input whose source_sha256 already carries a "
            "REJECT-disposition provenance record, at full decode cost, "
            "instead of skipping it (D-04's transient-failure escape hatch)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report what a real conversion would do — resolved action "
            "sources, predicted frame counts, which inputs would abort and "
            "why, the resume skip list, the CLI-08 prefix verdict, and the "
            "predicted dataset revision — without decoding, encoding, or "
            "creating anything under the output prefix (D-10)"
        ),
    )
    parser.add_argument(
        CONSTANT_CHANNEL_ACCEPTANCE_FLAG,
        action="store_true",
        help=(
            "accept statistics.constant_channel's finding (advisory as of "
            "G-03) and proceed. Names, at exit time, the exact channels "
            "accepted and the training risk (GR00T's min-max normalization "
            "dividing by zero on a zero-variance channel) — recorded in "
            "every run's report regardless. No run-config key mirrors this "
            "flag: a standing always-accept value hidden in a config file "
            "would become an invisible waiver, which is exactly what this "
            "gate exists to prevent (deliberate omission, not oversight)."
        ),
    )
    return parser


def run_convert(
    out_dir: Path,
    *,
    profile: EmbodimentProfile,
    run_cfg: RunConfig,
    cli_output_prefix: str | None,
    no_validate: bool,
    use_local_golden: bool,
    retry_rejected: bool = False,
    accept_constant_channels: bool = False,
    stdin_isatty: Callable[[], bool] | None = None,
    prompt: Callable[[str], str] | None = None,
) -> ConvertOutcome:
    """Discovers every input `discover_inputs` names, converts each in
    order, and writes the manifest/report regardless of outcome — even a
    batch with zero converted episodes (every input rejected, resumed-skip,
    or duplicate-collapsed) still appends a `runs` entry and a report
    (CLI-05/CLI-06's every-input-accounted, never-a-silent-empty-conversion
    rule).

    Resumability (CLI-02/D-01/D-03/D-04): the skip set is built ONLY from
    `meta/data_engine/episode_{NNNNNN}.json` contents via
    `read_recorded_source_hashes`/`read_recorded_dispositions` — never from
    a directory listing of `data/` or `videos/`, which cannot distinguish a
    completed episode from crash debris. An input whose hash already
    carries a non-REJECT record is bucketed `resumed_skip`; one whose hash
    carries a REJECT record is bucketed `rejected_skip` unless
    `retry_rejected` is set, in which case it is converted normally and its
    prior rejection record is superseded. Crash-debris reclamation runs once
    the promotion guard has decided `RESUME`, before the writer is
    constructed, so `resume=True` never rehydrates over artifacts with no
    provenance record.
    """
    out_dir = out_dir.resolve()
    profile_hash_value = profile_hash(profile)
    resolved_stdin_isatty = stdin_isatty if stdin_isatty is not None else sys.stdin.isatty
    resolved_prompt: Callable[[str], str] = prompt if prompt is not None else input

    # CLI-08: the promotion guard runs before anything else touches
    # out_dir — before discover_inputs (which never touches out_dir, but
    # keeping the guard textually first matches "the first thing run_convert
    # does") and, critically, before the writer or the recorder are ever
    # constructed (Pattern 3, 04-RESEARCH.md).
    prefix_decision = evaluate_prefix(
        out_dir,
        pipeline_version=PIPELINE_VERSION,
        profile_hash_value=profile_hash_value,
        format_key=run_cfg.format_key,
    )
    if prefix_decision.verdict is PrefixVerdict.REFUSE:
        print(prefix_decision.reason, file=sys.stderr)
        return ConvertOutcome(
            exit_code=EXIT_PREFIX_REFUSED,
            ledger=(),
            dataset_revision=None,
            manifest_path=None,
            report_path=None,
            validation=None,
            prefix_decision=prefix_decision,
        )
    if prefix_decision.verdict is PrefixVerdict.STATE_ERROR:
        print(prefix_decision.reason, file=sys.stderr)
        return ConvertOutcome(
            exit_code=EXIT_PREFIX_STATE_ERROR,
            ledger=(),
            dataset_revision=None,
            manifest_path=None,
            report_path=None,
            validation=None,
            prefix_decision=prefix_decision,
        )

    discovered = discover_inputs(run_cfg)

    hashes_by_path: dict[Path, str] = {}
    for candidate in discovered:
        if candidate not in hashes_by_path:
            hashes_by_path[candidate] = _hash_file_sha256(candidate)

    # D-01/D-03/Pitfall 6: the skip set and the crash-debris candidate set
    # are BOTH built exclusively from meta/data_engine/episode_*.json
    # contents — never from a data/ or videos/ directory listing, which
    # cannot tell a completed episode apart from crash debris.
    #
    # emitted_video_keys is computed here, BEFORE _build_resume_maps, so it
    # can be passed to it (04-09) — moved up from its former position after
    # this block, which is why this comment now sits above both.
    emitted_video_keys = [
        cam.video_key for cam in profile.cameras.values() if cam.emit and cam.video_key is not None
    ]
    recorded_source_hashes_by_index = read_recorded_source_hashes(out_dir)
    converted_hash_to_index, rejected_hash_to_index = _build_resume_maps(
        out_dir, emitted_video_keys=emitted_video_keys
    )

    ledger: list[InputLedgerEntry] = []
    swept_temp_paths: list[str] = []
    if prefix_decision.verdict is PrefixVerdict.RESUME:
        # WR-04 (04-11): interrupted-write .tmp debris is swept once per
        # RESUME, before crash-debris reclamation below — a .tmp file is
        # never a completed artifact, so it can never be what reclamation's
        # own recorded/artifact-index comparison is deciding between.
        swept_temp_paths = _sweep_write_temp_debris(
            out_dir, emitted_video_keys=emitted_video_keys
        )
        # D-03: crash debris — an artifact with no provenance record — is
        # reclaimed before the writer is constructed, so resume=True never
        # rehydrates state that includes a never-committed episode. Always
        # reported, never silent.
        ledger.extend(
            _reclaim_crash_debris(
                out_dir,
                emitted_video_keys=emitted_video_keys,
                recorded_indices=set(recorded_source_hashes_by_index),
            )
        )

    writer = get_writer(
        run_cfg.format_key,
        out_dir,
        profile,
        codec=run_cfg.codec,
        emit_episodes_stats=run_cfg.emit_episodes_stats,
        stdin_isatty=lambda: False,
        resume=(prefix_decision.verdict is PrefixVerdict.RESUME),
    )
    recorder = ProvenanceRecorder(out_dir, profile)

    started_at = _utc_now_iso()
    seen_resolved: set[Path] = set()
    seen_hashes: set[str] = set()
    # Mirrors the writer's own `_next_episode_index` (never read directly —
    # the writer stays the sole authority on that counter, per its own
    # docstring): incremented only after a real, non-zero-frame write, so a
    # rejected or zero-frame input's record_rejected() call always names
    # the same index the next REAL write will consume. A rejected input
    # immediately followed by a converted one therefore shares that index —
    # the run report (canonical per D-09) is what preserves the rejected
    # input's disposition either way. On a RESUME, this mirrors the writer's
    # own rehydration rule exactly: max(recorded episode_index) + 1, never a
    # row count — reclamation above never adds a provenance record, so it
    # cannot move this value.
    next_episode_index = (
        max(recorded_source_hashes_by_index) + 1 if recorded_source_hashes_by_index else 0
    )

    # D-08: progress is stderr-only, tty-aware, and reports real conversion
    # work only — resumed/rejected/duplicate inputs never touch the encoder
    # and have nothing to report progress on.
    progress = ProgressReporter(len(discovered))

    for input_path in discovered:
        entry_start = time.monotonic()
        source_sha256 = hashes_by_path[input_path]

        if input_path in seen_resolved:
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="duplicate_source",
                    episode_index=None,
                    frame_count=None,
                    disposition=None,
                    checks=(),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
            continue
        seen_resolved.add(input_path)

        if source_sha256 in seen_hashes:
            # CLI-05: a distinct path whose CONTENT hashes the same as an
            # input already processed earlier in this same run collapses to
            # duplicate_source too — exactly one conversion per distinct
            # source_sha256 per run, not one per distinct path.
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="duplicate_source",
                    episode_index=None,
                    frame_count=None,
                    disposition=None,
                    checks=(),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
            continue
        seen_hashes.add(source_sha256)

        if source_sha256 in converted_hash_to_index:
            # D-01: already converted in a prior run — never opened for
            # decode.
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="resumed_skip",
                    episode_index=converted_hash_to_index[source_sha256],
                    frame_count=None,
                    disposition=None,
                    checks=(),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
            continue

        retry_target_index: int | None = None
        if source_sha256 in rejected_hash_to_index:
            retry_target_index = rejected_hash_to_index[source_sha256]
            if not retry_rejected:
                # D-04: skipped and re-accounted from the existing rejection
                # record — never opened for decode unless --retry-rejected.
                ledger.append(
                    InputLedgerEntry(
                        source_path=str(input_path),
                        source_sha256=source_sha256,
                        bucket="rejected_skip",
                        episode_index=retry_target_index,
                        frame_count=None,
                        disposition=Disposition.REJECT.value,
                        checks=(),
                        elapsed_seconds=time.monotonic() - entry_start,
                    )
                )
                continue

        try:
            episode = GalaxeaInputAdapter().parse(str(input_path), profile)
        except GalaxeaAdapterError as e:
            if e.rejected_header is None:
                raise
            # D-04: a --retry-rejected re-attempt that still fails supersedes
            # the existing rejection record at its OWN index (an overwrite,
            # not a new index) rather than consuming a fresh one.
            reject_index = (
                retry_target_index if retry_target_index is not None else next_episode_index
            )
            recorder.record_rejected(
                e.rejected_header, e.source_uri or str(input_path), reject_index
            )
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="newly_rejected",
                    episode_index=reject_index,
                    frame_count=None,
                    disposition=Disposition.REJECT.value,
                    checks=tuple(e.rejected_header.global_meta.get("checks", ())),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
            continue

        rebuilt_header = dataclasses.replace(episode.header, task_prompt=run_cfg.task_string)
        rebuilt_episode = CanonicalEpisode(header=rebuilt_header, frames=episode.frames)
        # RESEARCH.md Open Question 7 / D-04's timing split: capture the
        # writer's CUMULATIVE decode/encode properties immediately before
        # and immediately after this one write_episode call and DIFFERENCE
        # them — never record the raw post-call reading, which is the
        # whole batch's running total and would inflate monotonically once
        # recorded once per episode (LeRobotV21Writer.decode_seconds's own
        # docstring).
        #
        # WR-02 (04-11): an unexpected exception out of write_episode (an
        # encoder fault, a corrupt JPEG, an unsupported frame shape) must
        # cost this ONE input, never the whole batch's run report. Catches
        # the narrow builtin `Exception`, never the two-word base class
        # that would also swallow a KeyboardInterrupt and make the batch
        # unstoppable. No provenance record is written here, deliberately.
        #
        # CR-01 (04-REVIEW.md) correction: write_episode's non-zero-frame
        # branch does NOT commit its parquet as the last word. Its actual
        # order is camera finalize() -> encode-manifest os.replace ->
        # parquet os.replace -> in-memory _tasks_meta/_episodes_meta append
        # -> stats.write_partial_file/merge_partials -> only THEN
        # `_next_episode_index += 1`. A raise in the stats step therefore
        # lands strictly after the parquet is durably committed. Correctness
        # here now rests on `write_episode` itself (data/writers/lerobot_v21.py)
        # rolling back its own `_tasks_meta`/`_episodes_meta` appends and
        # deleting the artifacts it committed for this call before
        # re-raising, so that by the time control reaches this `except`
        # block the writer's counters and lists are exactly as if this
        # call had never happened — which is what makes it safe to leave no
        # provenance record and no claimed episode index here: the input is
        # re-attempted on the next resume (correct for a transient I/O
        # fault) and any orphan on-disk artifact is either already cleaned
        # up by the writer's own rollback or reclaimed as crash debris —
        # mirroring the zero-frame branch just below, which also appends a
        # newly_rejected entry with episode_index=None and writes no
        # record. This input's cross-run accounting therefore rests on the
        # run report (canonical per D-09) rather than a provenance record.
        try:
            decode_before, encode_before = writer.decode_seconds, writer.encode_seconds
            write_result = writer.write_episode(rebuilt_episode)
            entry_decode_seconds = writer.decode_seconds - decode_before
            entry_encode_seconds = writer.encode_seconds - encode_before
        except Exception as e:
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="newly_rejected",
                    episode_index=None,
                    frame_count=None,
                    disposition=Disposition.REJECT.value,
                    checks=(
                        CheckResult(
                            WRITE_UNEXPECTED_FAILURE_CHECK_NAME,
                            "fatal",
                            False,
                            {
                                "exception_type": type(e).__name__,
                                "exception_message": str(e),
                                "source_path": str(input_path),
                            },
                        ),
                    ),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
            # CR-01 backstop (04-REVIEW.md fix option (c)): this whole
            # guard's safety depends on write_episode rolling back its own
            # `_tasks_meta`/`_episodes_meta` appends on failure so the
            # writer's episode-index counter and its metadata lists never
            # disagree. If that invariant is ever violated (a future writer
            # change re-introduces the CR-01 corruption), fail loudly here
            # — immediately, at the point of the violation — rather than
            # silently baking a duplicate `episode_index` row into
            # `episodes.jsonl` at `finalize()` time, far from the cause.
            assert writer.episodes_meta_count == writer.next_episode_index, (
                "writer.write_episode left _episodes_meta/_next_episode_index "
                "out of sync after raising — this would corrupt episodes.jsonl "
                "with a duplicate episode_index on the next successful write "
                "(see 04-REVIEW.md CR-01)"
            )
            continue

        if write_result.frame_count == 0:
            # Pitfall 3 (04-RESEARCH.md): a zero-frame write does not
            # advance the writer's episode index and produces no encode
            # manifest / stats partial — per D-03 it is not "done", so no
            # provenance record is written at the returned index, which the
            # next real episode will reuse.
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="newly_rejected",
                    episode_index=None,
                    frame_count=0,
                    disposition=None,
                    checks=tuple(rebuilt_header.global_meta.get("checks", ())),
                    elapsed_seconds=time.monotonic() - entry_start,
                )
            )
        else:
            recorder.record(rebuilt_header, write_result)
            if (
                retry_target_index is not None
                and retry_target_index != write_result.episode_index
            ):
                # WR-05: a --retry-rejected re-conversion that succeeds at a
                # NEW index supersedes the stale rejection record at
                # retry_target_index rather than leaving two live records
                # for the same input. Never called on the retry-failed-again
                # path (data/cli.py's record_rejected call above), which
                # overwrites at retry_target_index and supersedes nothing.
                recorder.mark_superseded(
                    retry_target_index,
                    superseded_by_episode_index=write_result.episode_index,
                )
            recorded_disposition = rebuilt_header.global_meta.get("disposition")
            ledger.append(
                InputLedgerEntry(
                    source_path=str(input_path),
                    source_sha256=source_sha256,
                    bucket="converted",
                    episode_index=write_result.episode_index,
                    frame_count=write_result.frame_count,
                    disposition=(
                        recorded_disposition.value
                        if isinstance(recorded_disposition, Disposition)
                        else None
                    ),
                    checks=tuple(rebuilt_header.global_meta.get("checks", ())),
                    elapsed_seconds=time.monotonic() - entry_start,
                    decode_seconds=entry_decode_seconds,
                    encode_seconds=entry_encode_seconds,
                )
            )
            next_episode_index += 1
            progress.update(frames_this_episode=write_result.frame_count)

    progress.finish()

    _assert_ledger_complete(tuple(ledger), len(discovered))

    final_decode_seconds = writer.decode_seconds
    final_encode_seconds = writer.encode_seconds

    writer.finalize()

    validation: ValidationOutcome | None = None
    if not no_validate:
        validation = validate_dataset(out_dir, profile, use_local_golden=use_local_golden)

    # G-03: resolved immediately after validate_dataset and before the
    # report payload is built, so `constant_channel_acceptance` is present
    # in the report on EVERY run — including a run with no finding and a
    # run that skipped validation entirely.
    constant_channel_acceptance = resolve_constant_channel_acceptance(
        constant_channel_finding(validation),
        invoked=not no_validate,
        accept_flag=accept_constant_channels,
        stdin_isatty=resolved_stdin_isatty(),
        prompt=resolved_prompt,
    )
    if constant_channel_acceptance.finding_present and not constant_channel_acceptance.accepted:
        print(
            "constant-channel finding not accepted — channels: "
            f"{', '.join(constant_channel_acceptance.channels)}. Re-run with "
            f"{CONSTANT_CHANNEL_ACCEPTANCE_FLAG} to accept and proceed.",
            file=sys.stderr,
        )

    finished_at = _utc_now_iso()

    recorded_source_hashes = _scan_recorded_source_hashes(
        out_dir, emitted_video_keys=emitted_video_keys
    )
    git_commit = _resolve_git_commit()
    revision = dataset_revision_id(
        sorted(recorded_source_hashes), profile_hash_value, git_commit, run_cfg.format_key
    )

    # WINDOWS.md id 12 (narrowed, 04-11): the documented workstation
    # procedure names a prefix from a dry run's PREDICTED revision before
    # the real content exists, so a divergence between that name and this
    # run's ACTUAL computed revision is a normal, expected occurrence —
    # reported, never refused (refusing would block every legitimate run
    # whose prefix is named before its content exists) and never used to
    # rename the directory out from under a running conversion (not this
    # function's business). `None`, never `False`, when the basename does
    # not even look like a `rev-` token at all — there is nothing to
    # compare in that case, mirroring `ConstantChannelAcceptance.accepted`'s
    # own `bool | None` convention for "nothing to compare".
    prefix_name_matches_revision: bool | None
    if not _REVISION_PREFIX_NAME_RE.fullmatch(out_dir.name):
        prefix_name_matches_revision = None
    else:
        prefix_name_matches_revision = out_dir.name == revision[0]
        if not prefix_name_matches_revision:
            print(
                f"prefix directory name ({out_dir.name}) no longer describes "
                f"the content it holds — this run's computed revision is "
                f"{revision[0]}",
                file=sys.stderr,
            )

    # WR-01 (04-11): detects — never repairs — a manifest `runs` entry from
    # a PRIOR run whose recorded report_path no longer exists on disk (the
    # crash window between append_run_to_manifest and write_run_report,
    # below, that this plan's scope_decisions explicitly leaves open).
    # Never changes this run's exit code and never fabricates the missing
    # report — a report describes a run this process did not perform.
    orphaned_report_paths = _orphaned_report_paths(out_dir)
    if orphaned_report_paths:
        print(
            "orphaned run report(s) — the manifest entry survived but its "
            f"report did not: {', '.join(orphaned_report_paths)}",
            file=sys.stderr,
        )

    effective_config = run_cfg.model_dump(mode="json")
    effective_config["output_prefix"] = str(out_dir)
    overridden_config: dict[str, Any] = {}
    if (
        cli_output_prefix is not None
        and run_cfg.output_prefix is not None
        and str(out_dir) != run_cfg.output_prefix
    ):
        overridden_config["output_prefix"] = run_cfg.output_prefix

    existing_manifest = read_run_manifest(out_dir)
    run_ordinal = len(existing_manifest["runs"]) if existing_manifest is not None else 0
    report_path = (
        out_dir / "meta" / "data_engine" / RUN_REPORTS_DIRNAME / f"run_{run_ordinal:03d}.json"
    )

    run_entry: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": git_commit,
        "package_versions": _read_pinned_package_versions(),
        "effective_config": effective_config,
        "overridden_config": overridden_config,
        "inputs": [
            {
                "source_path": e.source_path,
                "source_sha256": e.source_sha256,
                "bucket": e.bucket,
                "episode_index": e.episode_index,
                "frame_count": e.frame_count,
                "disposition": e.disposition,
            }
            for e in ledger
        ],
        "report_path": str(report_path),
    }

    # WR-01 (04-11): a crash between THIS append and the write_run_report
    # call below would leave THIS run's own manifest entry orphaned —
    # detected on the NEXT run into this prefix by _orphaned_report_paths
    # above, never here. The write order stays exactly as it was
    # (append-then-report): swapping it only moves the window rather than
    # closing it, because write_run_report refuses to overwrite an
    # existing report at an ordinal, and run_ordinal is derived from the
    # manifest's own run count — a crash after a report-first write would
    # leave an orphan report AND make the next run recompute the same
    # ordinal and abort outright on that refusal, trading a silent orphan
    # for a hard block on every future run into this prefix.
    manifest_path = append_run_to_manifest(
        out_dir,
        run_entry=run_entry,
        pipeline_version=PIPELINE_VERSION,
        profile_hash_value=profile_hash_value,
        format_key=run_cfg.format_key,
        dataset_revision={
            "short": revision[0],
            "full": revision[1],
            # Every input dataset_revision_id hashed, nested so the digest
            # is recomputable from the manifest alone — the promotion
            # invariant below re-derives `full` from exactly this block.
            "inputs": {
                "source_sha256s": sorted(recorded_source_hashes),
                "profile_hash": profile_hash_value,
                "git_commit": git_commit,
                "format_key": run_cfg.format_key,
            },
        },
    )
    _assert_manifest_revision_recomputes(out_dir)

    bucket_counts: dict[str, int] = {}
    for entry in ledger:
        bucket_counts[entry.bucket] = bucket_counts.get(entry.bucket, 0) + 1

    report_payload: dict[str, Any] = {
        "run_ordinal": run_ordinal,
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset_revision": {"short": revision[0], "full": revision[1]},
        "inputs_discovered": len(discovered),
        "bucket_counts": bucket_counts,
        "ledger": [
            {
                "source_path": e.source_path,
                "source_sha256": e.source_sha256,
                "bucket": e.bucket,
                "episode_index": e.episode_index,
                "frame_count": e.frame_count,
                "disposition": e.disposition,
                "checks": [_serialize_check(c) for c in e.checks],
                "elapsed_seconds": e.elapsed_seconds,
                "decode_seconds": e.decode_seconds,
                "encode_seconds": e.encode_seconds,
                "deleted_paths": list(e.deleted_paths),
            }
            for e in ledger
        ],
        "cumulative_decode_seconds": final_decode_seconds,
        "cumulative_encode_seconds": final_encode_seconds,
        "validation": _serialize_validation_outcome(validation, invoked=not no_validate),
        "constant_channel_acceptance": constant_channel_acceptance.as_dict(),
        "orphaned_report_paths": orphaned_report_paths,
        "swept_temp_paths": swept_temp_paths,
        "prefix_name_matches_revision": prefix_name_matches_revision,
    }
    written_report_path = write_run_report(
        out_dir, run_ordinal=run_ordinal, payload=report_payload
    )

    exit_code = _derive_exit_code(tuple(ledger), validation, constant_channel_acceptance)

    return ConvertOutcome(
        exit_code=exit_code,
        ledger=tuple(ledger),
        dataset_revision=revision,
        manifest_path=manifest_path,
        report_path=written_report_path,
        validation=validation,
        constant_channel_acceptance=constant_channel_acceptance,
        prefix_decision=prefix_decision,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    run_cfg = load_run_config(args.run_config)

    resolved_out_dir = args.out_dir if args.out_dir is not None else run_cfg.output_prefix
    if resolved_out_dir is None:
        parser.error(
            "no output prefix given: pass it positionally (out_dir) or set "
            "output_prefix in the run config file"
        )

    if args.dry_run:
        return run_dry_run(Path(resolved_out_dir), profile=profile, run_cfg=run_cfg)

    outcome = run_convert(
        Path(resolved_out_dir),
        profile=profile,
        run_cfg=run_cfg,
        cli_output_prefix=args.out_dir,
        no_validate=args.no_validate,
        use_local_golden=args.use_local_golden,
        retry_rejected=args.retry_rejected,
        accept_constant_channels=args.accept_constant_channels,
    )

    if outcome.report_path is not None:
        print(str(outcome.report_path))
    if outcome.dataset_revision is not None:
        print(outcome.dataset_revision[0])

    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
