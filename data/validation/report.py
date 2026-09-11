"""D-02/D-03: the report layer. `amend_episode_record` delegates to
`ProvenanceRecorder.amend_checks` — the one place a written provenance
record is legitimately rewritten. `write_validation_report` writes the
dataset-wide `meta/data_engine/validation_report.json` atomically with
sorted keys. Nothing in this module deletes, moves, or renumbers anything on
disk.
"""

import json
import os
from pathlib import Path
from typing import Any

from data.contracts.containers import CheckResult, Disposition
from data.provenance.recorder import ProvenanceRecorder

VALIDATION_REPORT_FILENAME = "validation_report.json"


def amend_episode_record(
    recorder: ProvenanceRecorder,
    episode_index: int,
    new_checks: tuple[CheckResult, ...],
    *,
    disposition: Disposition,
) -> Path:
    """Thin delegation to `ProvenanceRecorder.amend_checks` — kept as a
    module-level function in the report layer so `validate.py` never
    reaches directly into `data/provenance/` for the amend call."""
    return recorder.amend_checks(episode_index, new_checks, disposition=disposition)


def _serialize_results(results: tuple[CheckResult, ...]) -> list[dict[str, Any]]:
    return [
        {
            "check_name": r.check_name,
            "severity": r.severity,
            "passed": r.passed,
            "evidence": dict(r.evidence),
        }
        for r in results
    ]


def write_validation_report(
    out_dir: str | Path,
    *,
    record_schema_version: int,
    dataset_results: tuple[CheckResult, ...],
    episode_results: dict[int, tuple[CheckResult, ...]],
    fully_validated: bool,
    disposition: Disposition,
) -> Path:
    """Writes `meta/data_engine/validation_report.json` atomically
    (temp-then-`os.replace`) with `json.dump(sort_keys=True, indent=2)`,
    carrying: `record_schema_version` echoed from the contract, the ordered
    list of dataset-scoped results, the per-episode result summary, the
    `fully_validated` boolean, and the computed overall `Disposition`.
    """
    meta_dir = Path(out_dir) / "meta" / "data_engine"
    meta_dir.mkdir(parents=True, exist_ok=True)
    final_path = meta_dir / VALIDATION_REPORT_FILENAME
    tmp_path = meta_dir / f".{VALIDATION_REPORT_FILENAME}.tmp"

    payload: dict[str, Any] = {
        "record_schema_version": record_schema_version,
        "dataset_results": _serialize_results(dataset_results),
        "episode_results": {
            str(episode_index): _serialize_results(results)
            for episode_index, results in sorted(episode_results.items())
        },
        "fully_validated": fully_validated,
        "disposition": disposition.value,
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp_path, final_path)
    return final_path
