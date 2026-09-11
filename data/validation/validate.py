"""D-04: `validate_dataset` — the importable entry point Phase 4's CLI (and
`data/validate.py`'s argv shim) calls. Builds the default registry, runs
every registered check in declared order, amends each affected episode
record, writes the dataset-wide report, and computes the exit code.

Every registered check runs regardless of earlier failures — the report is
always complete and there is no short-circuit — because D-02 makes this
stage a reporting stage whose gate is Phase 4's promotion rule, and a
partial report would make "not reported" indistinguishable from "passed",
the exact silent class this phase exists to close.
"""

import dataclasses
from pathlib import Path

from data.contracts.containers import RECORD_SCHEMA_VERSION, CheckResult, Disposition
from data.contracts.profile import EmbodimentProfile
from data.provenance.recorder import ProvenanceRecorder
from data.validation.context import BaselineConfig, ValidationContext
from data.validation.registry import UNEVALUATED_KEY, ValidationRegistry
from data.validation.report import amend_episode_record, write_validation_report


@dataclasses.dataclass(frozen=True)
class ValidationOutcome:
    """Everything one `validate_dataset` run produced: the episode-scoped
    results per episode index, the dataset-scoped results, whether every
    check had a basis for a verdict, and the process exit code."""

    episode_results: dict[int, tuple[CheckResult, ...]]
    dataset_results: tuple[CheckResult, ...]
    fully_validated: bool
    exit_code: int


def _is_unevaluated(result: CheckResult) -> bool:
    return bool(result.evidence.get(UNEVALUATED_KEY, False))


def _disposition_for(results: tuple[CheckResult, ...]) -> Disposition:
    """`REJECT` when any fatal-severity result failed WITHOUT the
    unevaluated marker — an unevaluated result never counts toward a
    disposition downgrade (registry.py's own documented severity
    semantics)."""
    for result in results:
        if result.severity == "fatal" and not result.passed and not _is_unevaluated(result):
            return Disposition.REJECT
    return Disposition.PASS


def validate_dataset(
    out_dir: str | Path,
    profile: EmbodimentProfile,
    *,
    baseline: BaselineConfig | None = None,
    use_local_golden: bool = False,
) -> ValidationOutcome:
    ctx = ValidationContext(
        out_dir=Path(out_dir),
        profile=profile,
        baseline=baseline,
        use_local_golden=use_local_golden,
    )
    registry = ValidationRegistry.default()
    recorder = ProvenanceRecorder(ctx.out_dir, profile)

    fully_validated = True
    any_fatal_failure = False
    episode_results: dict[int, tuple[CheckResult, ...]] = {}

    for episode_index in ctx.episode_indices:
        results: list[CheckResult] = []
        for check in registry.episode_scoped:
            results.extend(check.fn(ctx, episode_index))
        results_tuple = tuple(results)
        episode_results[episode_index] = results_tuple

        if any(_is_unevaluated(r) for r in results_tuple):
            fully_validated = False

        episode_disposition = _disposition_for(results_tuple)
        if episode_disposition is Disposition.REJECT:
            any_fatal_failure = True
        amend_episode_record(
            recorder, episode_index, results_tuple, disposition=episode_disposition
        )

    dataset_results: list[CheckResult] = []
    for check in registry.dataset_scoped:
        dataset_results.extend(check.fn(ctx))
    dataset_results_tuple = tuple(dataset_results)

    if any(_is_unevaluated(r) for r in dataset_results_tuple):
        fully_validated = False
    if _disposition_for(dataset_results_tuple) is Disposition.REJECT:
        any_fatal_failure = True

    overall_disposition = Disposition.REJECT if any_fatal_failure else Disposition.PASS
    write_validation_report(
        ctx.out_dir,
        record_schema_version=RECORD_SCHEMA_VERSION,
        dataset_results=dataset_results_tuple,
        episode_results=episode_results,
        fully_validated=fully_validated,
        disposition=overall_disposition,
    )

    return ValidationOutcome(
        episode_results=episode_results,
        dataset_results=dataset_results_tuple,
        fully_validated=fully_validated,
        exit_code=1 if any_fatal_failure else 0,
    )
