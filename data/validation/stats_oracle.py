"""VALID-09/D-13: independent numpy recomputation of `meta/stats.json`.

**The trust trap this check exists to close.** GR00T's training path calls
`generate_stats()` and writes the file itself when ours is absent or
shape-invalid, but *trusts* ours without recomputing when it merely passes
`check_stats_validity`'s shape check — so an absent `stats.json` is
harmless (GR00T fixes it) while a plausible-looking WRONG one is silent
poison for every normalized training batch. This check is the oracle
GR00T itself never runs.

**The oracle is numpy, deliberately not a second Welford pass (D-13).**
`recompute_feature_stats` uses plain `np.mean`/`np.std`/`np.min`/`np.max`
over the full concatenated column, and `q01`/`q99` via sorting plus the
writer's own `data/writers/stats.py::_rank_quantile` — imported, never
reimplemented, and never substituted with `numpy`'s own `quantile`/
`percentile` (whose default linear-interpolation convention disagrees with
the writer's nearest-rank method and would fail on every prefix for a
reason unrelated to any real defect). A second Welford implementation is
equally excluded: the value of this check is that a bug in the writer's
*incremental* accumulator becomes visible against an *independent,
single-pass* reduction — which it would not be against a second copy of
the same algorithm.

**Tolerance is the contract (D-13).** `EXACT_STATS_KEYS` (`min`, `max`,
`q01`, `q99`) are compared with strict equality; `APPROX_STATS_KEYS`
(`mean`, `std`) are compared with `np.allclose(..., rtol=MEAN_STD_RTOL,
atol=0.0)`. Widening either bound to make a real prefix pass would absorb
a genuine arithmetic defect — turning this check into a manufacturer of
false confidence, which is strictly worse than no validator. Narrowing
either bound below what float32 accumulation actually produces would
false-positive on the legitimate summation-order difference between
`merge_partials` and a single full-array reduction (float addition is not
associative). Neither direction is a knob this check exists to be tuned
by; a disagreement outside these bounds is a finding to report, not a
number to adjust.

**Recorded narrowing of D-13's video clause.** D-13 originally read, in
part, "for video, re-merge the persisted `Uint8Histogram` partials rather
than re-decoding MP4s." That clause presupposed a video entry in
`meta/stats.json` to compare a re-merged histogram *against*. Research
settled that no such entry exists or should ever exist: both this
project's own `LeRobotV21Writer.finalize()` and GR00T's `generate_stats()`
select stats targets by the identical `"float" in dtype` substring test,
which structurally excludes a `video`-dtype feature
(`03-RESEARCH.md` § "Resolved during this research pass"). So this check
narrows that clause: instead of re-merging `Uint8Histogram` partials, it
asserts video features are *absent* from `meta/stats.json` and treats a
video entry there as a failure (see `check_stats_recomputation`'s feature-
set-equality branch below) — unchanged, still active, by this module.

The persisted `Uint8Histogram` sidecars themselves now DO have a
verification oracle, added by `03-09-PLAN.md` (WINDOWS.md id 2):
`data/validation/image_histogram.py::check_image_histogram_integrity`
asserts exact structural, mass, channel-agreement and cross-episode
merge-additivity invariants against the writer's own encode manifest and
`meta/info.json`'s `video_info` — never by re-merging against a
`meta/stats.json` video entry, which still does not and must not exist.
That oracle's own module docstring records the residual it deliberately
leaves open: it proves a histogram's mass and structure are correct, not
that its per-bin distribution is right.

**Read-only.** This check opens `meta/stats.json`, `meta/info.json`,
`meta/stats_partials/*.json`, and the parquet files for reading only. It
writes nothing into any of them.
"""

import json
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.contracts.containers import CheckResult
from data.validation.constant_channels import load_merged_partials
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.writers.stats import _rank_quantile

CHECK_NAME = "statistics.stats_recomputation"

EXACT_STATS_KEYS = ("min", "max", "q01", "q99")
"""Compared with strict equality — no float32/float64 summation-order
ambiguity applies to a min/max reduction or a rank-based quantile."""

APPROX_STATS_KEYS = ("mean", "std")
"""Compared with `np.allclose(..., rtol=MEAN_STD_RTOL, atol=0.0)` — merged
Welford accumulation and a single full-array reduction visit the same
float32 values in different orders, and float addition is not
associative."""

MEAN_STD_RTOL = 1e-6
"""D-13's fixed relative tolerance for `mean`/`std` — never to be widened
to absorb a real disagreement, never to be narrowed below what float32
accumulation legitimately produces."""


def _read_json(path: Any) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recompute_feature_stats(column: np.ndarray) -> dict[str, np.ndarray]:
    """Independent numpy recomputation of the seven `finalize_stats`
    statistics (minus `count`, handled separately by the caller) over one
    already-concatenated `(rows, width)` float64 column. `mean`/`std`/
    `min`/`max` are plain single-pass numpy reductions; `q01`/`q99` sort
    the column and call the writer's own `_rank_quantile` — the one thing
    this oracle deliberately shares with the code under test (D-13).

    A zero-row column returns all-zero vectors of the declared width — the
    caller is responsible for routing a genuinely empty feature to an
    unevaluated result before calling this; this function itself never
    raises on an empty column (mirroring the writer's `_rank_quantile`'s
    own zero-`n` branch, which returns zeros rather than raising).
    """
    width = column.shape[1] if column.ndim == 2 else 0
    if column.shape[0] == 0:
        zeros = np.zeros(width, dtype=np.float64)
        return {
            "mean": zeros,
            "std": zeros,
            "min": zeros,
            "max": zeros,
            "q01": zeros.copy(),
            "q99": zeros.copy(),
        }

    sorted_column = np.sort(column, axis=0)
    return {
        "mean": np.mean(column, axis=0),
        "std": np.std(column, axis=0),
        "min": np.min(column, axis=0),
        "max": np.max(column, axis=0),
        "q01": _rank_quantile(sorted_column, 0.01, "lower"),
        "q99": _rank_quantile(sorted_column, 0.99, "higher"),
    }


def _included_episode_indices(ctx: ValidationContext) -> set[int]:
    """Reads each `meta/stats_partials/episode_*.json` file's own
    `included_in_merge` boolean directly — a single flag per file, not a
    reconstruction of its `WelfordPartial` — so this check knows which
    episodes' parquet rows belong in the recomputed column without
    duplicating `load_merged_partials`'s merge arithmetic (that function
    is still reused below, for the zero-count guard and as the reference
    total row count)."""
    partials_dir = ctx.out_dir / "meta" / "stats_partials"
    included: set[int] = set()
    if not partials_dir.is_dir():
        return included
    for path in sorted(partials_dir.glob("episode_*.json")):
        try:
            payload = _read_json(path)
        except Exception:  # noqa: BLE001 - a corrupt partial file is simply excluded
            continue
        if payload.get("included_in_merge", True):
            try:
                included.add(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return included


def _read_feature_column(
    ctx: ValidationContext, feature: str, included_episode_indices: set[int]
) -> np.ndarray:
    """Reads `feature`'s parquet column from every included episode, in
    ascending episode order, and concatenates into one `(rows, width)`
    float64 array — "the full column read back from every episode's
    parquet," per this module's docstring, never from the writer's own
    retained-value cache.

    `meta/info.json`'s `"float" in dtype` selection includes not only the
    fixed-size-list `observation.state`/`action` columns (width > 1) but
    also the plain scalar `timestamp` column (`dtype: "float32"`, width
    1) — `_build_table`'s schema (`data/writers/lerobot_v21.py`) writes
    `timestamp` as `pa.float32()`, not `pa.list_(pa.float32(), N)`. Both
    shapes are handled here rather than assuming every float feature is a
    fixed-size list.
    """
    columns: list[np.ndarray] = []
    width = 0
    for episode_index in sorted(i for i in ctx.episode_indices if i in included_episode_indices):
        path = ctx.parquet_path(episode_index)
        if not path.exists():
            continue
        table = pq.read_table(path, columns=[feature])
        field_type = table.schema.field(feature).type
        is_list = pa.types.is_fixed_size_list(field_type)
        width = int(field_type.list_size) if is_list else 1
        rows = table.column(feature).to_pylist()
        if rows:
            flat_rows = rows if is_list else [[v] for v in rows]
            columns.append(np.asarray(flat_rows, dtype=np.float64).reshape(len(rows), width))
    if not columns:
        return np.zeros((0, width), dtype=np.float64)
    return np.concatenate(columns, axis=0)


def check_stats_recomputation(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset.

    1. Derives the expected float-feature set from `meta/info.json`'s own
       `"float" in dtype` substring test (exactly `finalize()`'s own
       selection) and asserts it against `meta/stats.json`'s keys in BOTH
       directions — a missing entry and an extra entry (including a
       `video`-dtype feature that should never appear there) are separate
       named failures.
    2. For each expected feature: routes a zero-width or zero-count merged
       partial (from `load_merged_partials`, reused rather than
       re-derived) to an explicit unevaluated result. Otherwise
       recomputes `mean`/`std`/`min`/`max`/`q01`/`q99` from the parquet
       columns of every included episode and compares against
       `meta/stats.json` at the tolerances D-13 fixed, plus an exact
       `count` comparison against the summed row count over the included
       episodes.
    """
    try:
        info = _read_json(ctx.out_dir / "meta" / "info.json")
        stats_payload = _read_json(ctx.out_dir / "meta" / "stats.json")
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (CheckResult(CHECK_NAME, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),)

    expected_features = sorted(
        name for name, feat in info.get("features", {}).items() if "float" in feat.get("dtype", "")
    )
    actual_features = sorted(stats_payload.keys())

    missing_from_stats_json = sorted(set(expected_features) - set(actual_features))
    extra_in_stats_json = sorted(set(actual_features) - set(expected_features))

    if missing_from_stats_json or extra_in_stats_json:
        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {
                    "missing_from_stats_json": missing_from_stats_json,
                    "extra_in_stats_json": extra_in_stats_json,
                },
            ),
        )

    try:
        merged_partials = load_merged_partials(ctx)
        included_episode_indices = _included_episode_indices(ctx)
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (CheckResult(CHECK_NAME, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),)

    per_feature: list[dict[str, Any]] = []
    unevaluated_features: list[dict[str, Any]] = []
    any_mismatch = False

    for feature in expected_features:
        partial = merged_partials.get(feature)
        if partial is None or partial.count == 0:
            unevaluated_features.append(
                {"feature": feature, "reason": "merged partial has zero count"}
            )
            continue

        try:
            column = _read_feature_column(ctx, feature, included_episode_indices)
        except Exception as e:  # noqa: BLE001 - reported not raised
            per_feature.append({"feature": feature, "error": f"{type(e).__name__}: {e}"})
            any_mismatch = True
            continue

        if column.shape[1] == 0:
            unevaluated_features.append({"feature": feature, "reason": "zero-width feature"})
            continue

        recomputed = recompute_feature_stats(column)
        recomputed_count = int(column.shape[0])
        stats_entry = stats_payload[feature]

        mismatches: list[dict[str, Any]] = []
        comparisons: dict[str, dict[str, Any]] = {}

        for key in EXACT_STATS_KEYS:
            expected_vals = np.asarray(stats_entry[key], dtype=np.float64)
            actual_vals = recomputed[key]
            equal = bool(np.array_equal(expected_vals, actual_vals))
            comparisons[key] = {"tolerance": "exact", "equal": equal}
            if not equal:
                mismatches.append(
                    {
                        "statistic": key,
                        "expected": expected_vals.tolist(),
                        "actual": actual_vals.tolist(),
                    }
                )

        for key in APPROX_STATS_KEYS:
            expected_vals = np.asarray(stats_entry[key], dtype=np.float64)
            actual_vals = recomputed[key]
            close = bool(np.allclose(expected_vals, actual_vals, rtol=MEAN_STD_RTOL, atol=0.0))
            comparisons[key] = {"tolerance": f"rtol={MEAN_STD_RTOL}", "equal": close}
            if not close:
                mismatches.append(
                    {
                        "statistic": key,
                        "expected": expected_vals.tolist(),
                        "actual": actual_vals.tolist(),
                    }
                )

        expected_count = int(stats_entry["count"][0])
        count_equal = expected_count == recomputed_count
        comparisons["count"] = {"tolerance": "exact", "equal": count_equal}
        if not count_equal:
            mismatches.append(
                {"statistic": "count", "expected": expected_count, "actual": recomputed_count}
            )

        if mismatches:
            any_mismatch = True

        per_feature.append(
            {
                "feature": feature,
                "row_count": recomputed_count,
                "comparisons": comparisons,
                "mismatches": mismatches,
            }
        )

    if expected_features and len(unevaluated_features) == len(expected_features):
        reason = "; ".join(f"{u['feature']}: {u['reason']}" for u in unevaluated_features)
        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {
                    UNEVALUATED_KEY: True,
                    SKIP_REASON_KEY: reason,
                    "unevaluated_features": unevaluated_features,
                },
            ),
        )

    return (
        CheckResult(
            CHECK_NAME,
            "fatal",
            not any_mismatch,
            {
                "per_feature": per_feature,
                "unevaluated_features": unevaluated_features,
                "exact_stats_keys": list(EXACT_STATS_KEYS),
                "approx_stats_keys": list(APPROX_STATS_KEYS),
                "mean_std_rtol": MEAN_STD_RTOL,
            },
        ),
    )
