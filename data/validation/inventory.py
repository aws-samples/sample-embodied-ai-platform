"""VALID-06/07/10: the three dataset-scoped read-back checks —
`metadata.episode_length_agreement`, `anomalies.nan_inf`, and
`metadata.task_index_join`. Every artifact open and parse is wrapped in a
`try`/`except` scoped to the library's own exception types and converted
into `passed=False` with the exception text under an `error` evidence key,
so a corrupted fixture reports one failed named check instead of aborting
the run — mirroring `data/adapters/checks.py`'s catch-and-report idiom.
"""

import json
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.contracts.containers import CheckResult
from data.validation.context import ValidationContext
from data.validation.registry import UNEVALUATED_KEY

EPISODE_LENGTH_CHECK_NAME = "metadata.episode_length_agreement"
NAN_INF_CHECK_NAME = "anomalies.nan_inf"
TASK_INDEX_JOIN_CHECK_NAME = "metadata.task_index_join"

_NAN_INF_SCAN_SCOPE = {
    "scanned": ["observation.state", "action", "meta/stats.json"],
    "excluded_reason": {
        "decoded_video": (
            "a uint8 pixel buffer has no bit pattern representing a "
            "non-finite value, so scanning it would be vacuous rather "
            "than reassuring"
        )
    },
}
"""D-flagged VALID-07 assumption: state columns, action columns, and stats
values are scanned; decoded video is deliberately not. Recorded in every
`anomalies.nan_inf` CheckResult's evidence under `scan_scope`, never left
unstated."""


def _read_jsonl_lines(path: Any) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def check_episode_length_agreement(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """VALID-06: compares each `meta/episodes.jsonl` row's declared `length`
    against that episode's actual parquet row count, read as an int from
    the parquet file's own row-group metadata. `==` between two ints, no
    tolerance anywhere. Mismatches are reported sorted ascending by
    `episode_index`, each naming the declared length and the actual count.
    """
    episodes_path = ctx.out_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return (
            CheckResult(
                EPISODE_LENGTH_CHECK_NAME,
                "fatal",
                False,
                {UNEVALUATED_KEY: True, "episode_count": 0},
            ),
        )

    lines = _read_jsonl_lines(episodes_path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as e:  # noqa: BLE001 - reported not raised
            return (
                CheckResult(
                    EPISODE_LENGTH_CHECK_NAME,
                    "fatal",
                    False,
                    {"error": f"line {line_number}: {type(e).__name__}: {e}", "line": line_number},
                ),
            )
        rows.append(row)

    if not rows:
        return (
            CheckResult(
                EPISODE_LENGTH_CHECK_NAME,
                "fatal",
                False,
                {UNEVALUATED_KEY: True, "episode_count": 0},
            ),
        )

    mismatches: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: r["episode_index"]):
        episode_index = row["episode_index"]
        declared_length = row["length"]
        parquet_path = ctx.parquet_path(episode_index)
        try:
            actual_row_count: int | None = (
                int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)
                if parquet_path.exists()
                else None
            )
        except Exception as e:  # noqa: BLE001 - reported not raised
            mismatches.append(
                {
                    "episode_index": episode_index,
                    "declared_length": declared_length,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue
        if actual_row_count is None or declared_length != actual_row_count:
            mismatches.append(
                {
                    "episode_index": episode_index,
                    "declared_length": declared_length,
                    "actual_row_count": actual_row_count,
                }
            )

    return (
        CheckResult(
            EPISODE_LENGTH_CHECK_NAME,
            "fatal",
            not mismatches,
            {"mismatches": mismatches, "episode_count": len(rows)},
        ),
    )


def _scan_feature_for_non_finite(
    ctx: ValidationContext, feature: str
) -> tuple[list[dict[str, Any]], int]:
    findings: list[dict[str, Any]] = []
    total = 0
    for episode_index in ctx.episode_indices:
        parquet_path = ctx.parquet_path(episode_index)
        if not parquet_path.exists():
            continue  # zero-frame episode — nothing written to scan
        try:
            table = pq.read_table(parquet_path)
        except Exception as e:  # noqa: BLE001 - reported not raised
            findings.append(
                {
                    "feature": feature,
                    "episode_index": episode_index,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            total += 1
            continue
        if feature not in table.schema.names:
            continue
        field_type = table.schema.field(feature).type
        if not pa.types.is_fixed_size_list(field_type) or field_type.list_size == 0:
            # A zero-width feature is a legitimate world-model-style input
            # (empty state/action), not an edge case to raise on.
            continue
        rows = table.column(feature).to_pylist()
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        non_finite_count = int(arr.size - int(np.sum(finite_mask)))
        if non_finite_count:
            total += non_finite_count
            first_bad = np.argwhere(~finite_mask)[0]
            findings.append(
                {
                    "feature": feature,
                    "episode_index": episode_index,
                    "row_index": int(first_bad[0]),
                    "channel_index": int(first_bad[1]),
                }
            )
    return findings, total


def _scan_stats_for_non_finite(ctx: ValidationContext) -> tuple[list[dict[str, Any]], int]:
    stats_path = ctx.out_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return [], 0
    try:
        with open(stats_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # noqa: BLE001 - reported not raised
        return [{"feature": "meta/stats.json", "error": f"{type(e).__name__}: {e}"}], 1

    findings: list[dict[str, Any]] = []
    total = 0
    for feature, stat_map in payload.items():
        if not isinstance(stat_map, dict):
            continue
        for stat_key, values in stat_map.items():
            if stat_key == "count" or not isinstance(values, list) or not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            finite_mask = np.isfinite(arr)
            non_finite_count = int(arr.size - int(np.sum(finite_mask)))
            if non_finite_count:
                total += non_finite_count
                first_bad = int(np.argwhere(~finite_mask)[0][0])
                findings.append(
                    {"feature": feature, "stat": stat_key, "channel_index": first_bad}
                )
    return findings, total


def check_nan_inf(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """VALID-07: scans every episode's `observation.state`/`action` parquet
    columns and every numeric value in `meta/stats.json` for non-finite
    values via `numpy.isfinite`. Reports the feature name and the row/
    channel index (parquet) or statistic key/channel index (stats) of the
    first non-finite value found per feature, plus a total non-finite
    count. Decoded video is deliberately excluded — see `scan_scope` in
    this check's own evidence.
    """
    if not ctx.episode_indices:
        return (
            CheckResult(
                NAN_INF_CHECK_NAME,
                "fatal",
                False,
                {UNEVALUATED_KEY: True, "scan_scope": _NAN_INF_SCAN_SCOPE},
            ),
        )

    findings: list[dict[str, Any]] = []
    total_non_finite = 0
    for feature in ("observation.state", "action"):
        feature_findings, feature_total = _scan_feature_for_non_finite(ctx, feature)
        findings.extend(feature_findings)
        total_non_finite += feature_total

    stats_findings, stats_total = _scan_stats_for_non_finite(ctx)
    findings.extend(stats_findings)
    total_non_finite += stats_total

    return (
        CheckResult(
            NAN_INF_CHECK_NAME,
            "fatal",
            total_non_finite == 0,
            {
                "findings": findings,
                "total_non_finite": total_non_finite,
                "scan_scope": _NAN_INF_SCAN_SCOPE,
            },
        ),
    )


def _read_task_rows(tasks_path: Any) -> list[dict[str, Any]]:
    if not tasks_path.exists():
        return []
    return [
        json.loads(line)
        for line in _read_jsonl_lines(tasks_path)
        if line.strip()
    ]


def check_task_index_join(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """VALID-10: builds a mapping from `meta/tasks.jsonl` rows keyed by
    `task_index`, then for every distinct `task_index` present in any
    episode parquet resolves it through a guarded accessor
    (`dict.get`, which cannot raise). Orphan indices are reported sorted
    ascending, alongside the highest index present in `meta/tasks.jsonl`.
    A `task_index` value that is not a Python `int` fails the check naming
    the row, the value, and its type — never silently coerced to match an
    equal-looking int.
    """
    tasks_path = ctx.out_dir / "meta" / "tasks.jsonl"
    task_rows = _read_task_rows(tasks_path)

    type_errors: list[dict[str, Any]] = []
    tasks_map: dict[int, dict[str, Any]] = {}
    for line_number, row in enumerate(task_rows, start=1):
        value = row.get("task_index")
        if isinstance(value, bool) or not isinstance(value, int):
            type_errors.append(
                {"line": line_number, "value": value, "type": type(value).__name__}
            )
            continue
        tasks_map[value] = row

    present_task_indices: set[int] = set()
    for episode_index in ctx.episode_indices:
        parquet_path = ctx.parquet_path(episode_index)
        if not parquet_path.exists():
            continue
        try:
            table = pq.read_table(parquet_path)
        except Exception:  # noqa: BLE001 - reported not raised
            continue
        if "task_index" not in table.schema.names:
            continue
        present_task_indices.update(int(v) for v in table.column("task_index").to_pylist())

    if not task_rows and not present_task_indices:
        return (
            CheckResult(
                TASK_INDEX_JOIN_CHECK_NAME,
                "fatal",
                False,
                {UNEVALUATED_KEY: True, "task_row_count": 0},
            ),
        )

    highest_present_task_index = max(tasks_map) if tasks_map else None

    if type_errors:
        return (
            CheckResult(
                TASK_INDEX_JOIN_CHECK_NAME,
                "fatal",
                False,
                {
                    "type_errors": type_errors,
                    "highest_present_task_index": highest_present_task_index,
                },
            ),
        )

    orphan_task_indices = sorted(idx for idx in present_task_indices if tasks_map.get(idx) is None)
    return (
        CheckResult(
            TASK_INDEX_JOIN_CHECK_NAME,
            "fatal",
            not orphan_task_indices,
            {
                "orphan_task_indices": orphan_task_indices,
                "highest_present_task_index": highest_present_task_index,
            },
        ),
    )
