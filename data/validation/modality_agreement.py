"""VALID-05/VALID-08: cross-artifact modality agreement and the slice
round-trip.

`meta/modality.json` and the generated `meta/modality_config.py` are the two
artifacts the training consumer uses to slice a flat `observation.state`/
`action` vector into named fields. Both are generated from the SAME
`EmbodimentProfile` (`data/writers/modality.py`), so two artifacts
re-derived from that profile always agree with each other trivially — that
recomputation proves nothing about the actual bytes on disk. This module
therefore reads what is on disk (`meta/modality.json`, the generated
`meta/modality_config.py`'s source text, `meta/info.json`) and compares it
against exactly ONE independent expectation — `derive_slice_map_from_profile`
— never against a second hand-written accumulator and never by re-deriving
both sides of the comparison from the profile.

**`check_modality_agreement` (VALID-08)** cross-checks `meta/modality.json`,
the generated Python module, the profile's own declared orderings, and
`meta/info.json`'s declared feature widths for exact structural agreement:
key sets, boundaries, contiguity, width totals, and orderings.

**`check_modality_slice_range` (VALID-05)** takes one known frame (the
first row of the lowest-indexed episode with at least one written parquet
row) and, for each declared field, slices `observation.state`/`action` by
the ON-DISK `meta/modality.json` range, then confirms every sliced value
lies within the numeric range independently derived for that field: the
field's index range from `derive_slice_map_from_profile` (never the on-disk
range under test) indexes into `meta/stats.json`'s per-channel `min`/`max`
lists. A shifted on-disk slice map pulls values from a different joint
group whose bounds generally differ — that is exactly the mis-slice this
check exists to catch.

**Never imports the generated Python module.** `meta/modality_config.py`'s
own top-level imports target the `gr00t` training package, which is
deliberately not installed in this environment (three independent Pythons,
none shared with the training containers — see this repo's own CLAUDE.md).
`parse_modality_config_keys` reads that module's SOURCE TEXT with the
standard-library `ast` module and resolves list literals with
`ast.literal_eval` — no dynamic-module-loading machinery of any kind
anywhere in this package.

Every file read, JSON parse, and AST parse is wrapped in a `try`/`except`
scoped at the top of each check function, converting any failure into
`passed=False` with the exception text under an `error` evidence key —
matching this package's existing catch-and-report idiom
(`data/validation/frame_integrity.py`, `inventory.py`).
"""

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from data.contracts.containers import CheckResult
from data.contracts.profile import EmbodimentProfile
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.writers.modality import _action_slice_map, _state_slice_map

CHECK_NAME_AGREEMENT = "consistency.modality_agreement"
CHECK_NAME_SLICE_RANGE = "consistency.modality_slice_range"

_EXPECTED_MODALITY_CATEGORIES = frozenset({"video", "state", "action", "language"})
"""The four `ModalityConfig` entry keys the generated module's top-level
dict always carries — used to locate that dict literal in the parsed AST
without depending on the generated config variable's name (which is
derived from `profile.embodiment` and therefore not a fixed literal)."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def derive_slice_map_from_profile(
    profile: EmbodimentProfile, scope: str
) -> dict[str, dict[str, int]]:
    """The ONE independent expectation both checks in this module compare
    on-disk artifacts against: `data/writers/modality.py`'s own
    `_state_slice_map`/`_action_slice_map`, imported rather than copied, so
    a comparison against this function is a comparison against the writer's
    own accumulate-from-zero-over-declared-widths construction — never a
    second, independently hand-written accumulator that could itself
    disagree with the writer for reasons unrelated to any real defect."""
    if scope == "state":
        return _state_slice_map(profile)
    if scope == "action":
        return _action_slice_map(profile)
    raise ValueError(f"derive_slice_map_from_profile: unknown scope {scope!r}")


def _find_modality_config_dict(tree: ast.AST) -> ast.Dict | None:
    """Walks every `ast.Dict` node in `tree` and returns the first one whose
    literal keys are exactly `_EXPECTED_MODALITY_CATEGORIES` — this is the
    generated module's top-level `{config_var} = {...}` dict, found without
    depending on `config_var`'s name."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        try:
            literal_keys = {ast.literal_eval(k) for k in node.keys if k is not None}
        except (ValueError, TypeError, SyntaxError):
            continue
        if literal_keys == _EXPECTED_MODALITY_CATEGORIES:
            return node
    return None


def _find_call_kwarg(node: ast.AST, name: str) -> ast.expr | None:
    if not isinstance(node, ast.Call):
        return None
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def parse_modality_config_keys(source_path: str | Path) -> dict[str, list[str]]:
    """Parses the generated `MODALITY_CONFIG_PATH` module's SOURCE TEXT with
    the standard-library `ast` module and extracts each `ModalityConfig`
    entry's `modality_keys` list literal via `ast.literal_eval` — **never**
    by importing the module. The module's own top-level imports
    (`gr00t.configs.data.embodiment_configs`, `gr00t.data.embodiment_tags`,
    `gr00t.data.types`) target a package that is deliberately not installed
    in this environment; importing it here would raise rather than
    validate. Returns one entry per category (`video`/`state`/`action`/
    `language`) present in the parsed dict.
    """
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    config_dict = _find_modality_config_dict(tree)
    if config_dict is None:
        raise ValueError(
            f"{source_path}: no dict literal with keys "
            f"{sorted(_EXPECTED_MODALITY_CATEGORIES)} found"
        )
    result: dict[str, list[str]] = {}
    for key_node, value_node in zip(config_dict.keys, config_dict.values):
        if key_node is None:
            continue
        category = ast.literal_eval(key_node)
        modality_keys_node = _find_call_kwarg(value_node, "modality_keys")
        if modality_keys_node is None:
            continue
        result[category] = list(ast.literal_eval(modality_keys_node))
    return result


def _match_list_range_upper_bound(node: ast.expr | None) -> int | None:
    """Structurally matches `list(range(0, N))` — the shape
    `_generate_modality_config_source` writes for the `action` entry's
    `delta_indices` — and returns `N`. This is a `Call`, not a literal, so
    `ast.literal_eval` cannot resolve it directly."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
        and len(node.args[0].args) == 2
    ):
        try:
            return int(ast.literal_eval(node.args[0].args[1]))
        except (ValueError, TypeError, SyntaxError):
            return None
    return None


def parse_modality_config_action_horizon(source_path: str | Path) -> int | None:
    """Extracts the `action` `ModalityConfig` entry's `delta_indices` upper
    bound (the action-chunking horizon) via a small structural AST match —
    informational only, echoed into `check_modality_agreement`'s evidence;
    never used by any pass/fail decision in this module. Returns `None` if
    the expected shape is not found, never raises."""
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    config_dict = _find_modality_config_dict(tree)
    if config_dict is None:
        return None
    for key_node, value_node in zip(config_dict.keys, config_dict.values):
        if key_node is None or ast.literal_eval(key_node) != "action":
            continue
        delta_node = _find_call_kwarg(value_node, "delta_indices")
        bound = _match_list_range_upper_bound(delta_node)
        if bound is not None:
            return bound
    return None


def _adjacency_violations(scope: str, actual_ranges: dict[str, Any]) -> list[dict[str, Any]]:
    """Ascending-`start`-order contiguity check over the ON-DISK ranges for
    one scope: the first range must start at 0, and each range's `end` must
    exactly equal the next range's `start` — a gap of one and an overlap of
    one both fail, each naming the two fields and both boundaries. Shared
    by `check_modality_agreement` (VALID-08) and `check_modality_slice_range`
    (VALID-05) so the two checks apply the identical adjacency rule to the
    same on-disk data."""
    violations: list[dict[str, Any]] = []
    sorted_items = sorted(actual_ranges.items(), key=lambda kv: kv[1].get("start", 0))
    if not sorted_items:
        return violations

    first_key, first_range = sorted_items[0]
    first_start = first_range.get("start")
    if first_start != 0:
        violations.append(
            {
                "scope": scope,
                "kind": "does_not_start_at_zero",
                "field": first_key,
                "start": first_start,
            }
        )

    for (key_a, range_a), (key_b, range_b) in zip(sorted_items, sorted_items[1:]):
        end_a = range_a.get("end")
        start_b = range_b.get("start")
        if end_a != start_b:
            kind = (
                "gap"
                if (isinstance(end_a, int) and isinstance(start_b, int) and start_b > end_a)
                else "overlap"
            )
            violations.append(
                {
                    "scope": scope,
                    "kind": kind,
                    "field_a": key_a,
                    "end_a": end_a,
                    "field_b": key_b,
                    "start_b": start_b,
                }
            )
    return violations


def _declared_width(info: dict[str, Any], feature_name: str) -> int | None:
    feature_entry = info.get("features", {}).get(feature_name)
    if not feature_entry:
        return None
    shape = feature_entry.get("shape") or []
    return int(shape[0]) if shape else None


# ---------------------------------------------------------------------------
# check_modality_agreement (VALID-08)
# ---------------------------------------------------------------------------


def _check_scope_agreement(
    *,
    scope: str,
    profile: EmbodimentProfile,
    profile_order: list[str],
    modality_json: dict[str, Any],
    info: dict[str, Any],
    feature_name: str,
    python_keys: list[str],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    expected_ranges = derive_slice_map_from_profile(profile, scope)
    actual_ranges = modality_json.get(scope, {})
    declared_width = _declared_width(info, feature_name)

    json_key_set = set(actual_ranges.keys())
    python_key_set = set(python_keys)
    profile_key_set = set(profile_order)

    # Exact UTF-8 key-set equality across all three artifacts — never a
    # case-folded, whitespace-stripped, or otherwise normalized comparison,
    # because the consumer joins on the literal key.
    if not (json_key_set == python_key_set == profile_key_set):
        violations.append(
            {
                "scope": scope,
                "kind": "key_set_mismatch",
                "json_keys": sorted(json_key_set),
                "python_config_keys": sorted(python_key_set),
                "profile_order_keys": sorted(profile_key_set),
                "missing_from_json": sorted((python_key_set | profile_key_set) - json_key_set),
                "missing_from_python_config": sorted(
                    (json_key_set | profile_key_set) - python_key_set
                ),
                "missing_from_profile": sorted((json_key_set | python_key_set) - profile_key_set),
            }
        )

    # Per-field boundary comparison against the ONE independent
    # (profile-derived) expectation — only for keys present on both sides;
    # a missing/extra key is already named above.
    for key in sorted(json_key_set & set(expected_ranges)):
        expected_range = expected_ranges[key]
        actual_range = actual_ranges[key]
        if (
            int(actual_range.get("start", -1)) != expected_range["start"]
            or int(actual_range.get("end", -1)) != expected_range["end"]
        ):
            violations.append(
                {
                    "scope": scope,
                    "kind": "boundary_mismatch",
                    "field": key,
                    "expected": expected_range,
                    "actual": {"start": actual_range.get("start"), "end": actual_range.get("end")},
                }
            )

    violations.extend(_adjacency_violations(scope, actual_ranges))

    expected_total = max((r["end"] for r in expected_ranges.values()), default=0)
    actual_total = max((int(r.get("end", 0)) for r in actual_ranges.values()), default=0)

    if declared_width is not None:
        if expected_total != declared_width:
            # The profile-derived total disagreeing with meta/info.json's
            # own declared width means the profile handed to this check is
            # not the one that produced info.json — a plumbing-level
            # finding, not a hand-edit scenario, but still worth naming
            # rather than silently comparing against a stale expectation.
            violations.append(
                {
                    "scope": scope,
                    "kind": "declared_width_disagrees_with_profile",
                    "profile_derived_total": expected_total,
                    "declared_width": declared_width,
                }
            )
        if actual_total != declared_width:
            violations.append(
                {
                    "scope": scope,
                    "kind": "width_sum_mismatch",
                    "actual_total": actual_total,
                    "declared_width": declared_width,
                }
            )
    elif actual_ranges or expected_ranges:
        violations.append(
            {"scope": scope, "kind": "declared_width_unavailable", "feature": feature_name}
        )

    # Ordering: the Python config's modality_keys order must equal the
    # profile's declared order and equal the ascending-start order of the
    # on-disk JSON map.
    json_ascending_order = [
        key for key, _ in sorted(actual_ranges.items(), key=lambda kv: kv[1].get("start", 0))
    ]
    if not (json_ascending_order == profile_order == python_keys):
        violations.append(
            {
                "scope": scope,
                "kind": "ordering_mismatch",
                "json_ascending_start_order": json_ascending_order,
                "profile_order": profile_order,
                "python_config_order": python_keys,
            }
        )

    return violations


def _check_video_agreement(
    profile: EmbodimentProfile, modality_json: dict[str, Any], info: dict[str, Any]
) -> list[dict[str, Any]]:
    """The `video` map's keys must match the profile's `emit: true` camera
    dict keys, each `original_key` must match that camera's declared
    `video_key`, and each such `video_key` must be present in
    `meta/info.json`'s features with `dtype: "video"`."""
    violations: list[dict[str, Any]] = []
    expected_cameras = {
        name: cam.video_key
        for name, cam in profile.cameras.items()
        if cam.emit and cam.video_key is not None
    }
    actual_video_map = modality_json.get("video", {})

    expected_keys = set(expected_cameras.keys())
    actual_keys = set(actual_video_map.keys())
    if expected_keys != actual_keys:
        violations.append(
            {
                "scope": "video",
                "kind": "key_set_mismatch",
                "expected_camera_keys": sorted(expected_keys),
                "actual_video_map_keys": sorted(actual_keys),
            }
        )

    features = info.get("features", {})
    for name in sorted(expected_keys & actual_keys):
        expected_video_key = expected_cameras[name]
        actual_original_key = actual_video_map[name].get("original_key")
        if actual_original_key != expected_video_key:
            violations.append(
                {
                    "scope": "video",
                    "kind": "original_key_mismatch",
                    "camera": name,
                    "expected_video_key": expected_video_key,
                    "actual_original_key": actual_original_key,
                }
            )
            continue
        feature_entry = features.get(expected_video_key)
        if feature_entry is None or feature_entry.get("dtype") != "video":
            violations.append(
                {
                    "scope": "video",
                    "kind": "video_feature_missing_or_wrong_dtype",
                    "camera": name,
                    "video_key": expected_video_key,
                    "feature_entry": feature_entry,
                }
            )
    return violations


def _check_annotation_agreement(
    modality_json: dict[str, Any], python_modality_keys: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """The JSON `annotation` map's bare subkey and the Python config's
    prefixed `language` `modality_keys` entry name the SAME field under two
    different spellings by design (`annotation.` prefix) — this asserts
    that legitimate asymmetry holds exactly, rather than treating it as a
    mismatch."""
    violations: list[dict[str, Any]] = []
    annotation_map = modality_json.get("annotation", {})
    if len(annotation_map) != 1:
        violations.append(
            {
                "scope": "annotation",
                "kind": "unexpected_annotation_key_count",
                "annotation_keys": sorted(annotation_map.keys()),
            }
        )
        return violations

    (json_subkey,) = annotation_map.keys()
    language_keys = python_modality_keys.get("language", [])
    if len(language_keys) != 1:
        violations.append(
            {
                "scope": "annotation",
                "kind": "unexpected_language_key_count",
                "language_keys": language_keys,
            }
        )
        return violations

    expected_prefixed = f"annotation.{json_subkey}"
    if language_keys[0] != expected_prefixed:
        violations.append(
            {
                "scope": "annotation",
                "kind": "annotation_prefix_mismatch",
                "json_annotation_subkey": json_subkey,
                "python_language_key": language_keys[0],
                "expected_python_language_key": expected_prefixed,
            }
        )
    return violations


def check_modality_agreement(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset. Reads `meta/modality.json`,
    `meta/modality_config.py` (parsed, never imported), and `meta/info.json`
    from disk, and compares them against the ONE independent expectation
    `derive_slice_map_from_profile` produces from `ctx.profile` — never
    against a second re-derivation of the on-disk artifacts themselves.

    Works identically whether or not any episode has been written: every
    file this check reads is written by `LeRobotV21Writer.finalize()` from
    the profile alone, so a zero-episode dataset still has all three files
    present.
    """
    try:
        modality_json_path = ctx.out_dir / "meta" / "modality.json"
        modality_config_path = ctx.out_dir / "meta" / "modality_config.py"
        info_path = ctx.out_dir / "meta" / "info.json"

        with open(modality_json_path, encoding="utf-8") as f:
            modality_json = json.load(f)
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        python_modality_keys = parse_modality_config_keys(modality_config_path)
        try:
            action_delta_bound = parse_modality_config_action_horizon(modality_config_path)
        except Exception:  # noqa: BLE001 - informational only, never fails the check
            action_delta_bound = None

        violations: list[dict[str, Any]] = []
        for scope, profile_order, feature_name in (
            ("state", list(ctx.profile.state_vector_order), "observation.state"),
            ("action", list(ctx.profile.action_primary), "action"),
        ):
            violations.extend(
                _check_scope_agreement(
                    scope=scope,
                    profile=ctx.profile,
                    profile_order=profile_order,
                    modality_json=modality_json,
                    info=info,
                    feature_name=feature_name,
                    python_keys=python_modality_keys.get(scope, []),
                )
            )

        violations.extend(_check_video_agreement(ctx.profile, modality_json, info))
        violations.extend(_check_annotation_agreement(modality_json, python_modality_keys))

        violations.sort(key=lambda v: (v.get("scope", ""), v.get("kind", ""), v.get("field", "")))

        return (
            CheckResult(
                CHECK_NAME_AGREEMENT,
                "fatal",
                not violations,
                {
                    "violations": violations,
                    "action_delta_indices_upper_bound": action_delta_bound,
                },
            ),
        )
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (CheckResult(CHECK_NAME_AGREEMENT, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),)


# ---------------------------------------------------------------------------
# check_modality_slice_range (VALID-05)
# ---------------------------------------------------------------------------


def _unevaluated_slice_range(*, reason: str) -> CheckResult:
    return CheckResult(
        CHECK_NAME_SLICE_RANGE, "fatal", False, {UNEVALUATED_KEY: True, SKIP_REASON_KEY: reason}
    )


def _find_known_frame(ctx: ValidationContext) -> tuple[int, Any] | tuple[None, None]:
    """Returns `(episode_index, parquet_table)` for the lowest-indexed
    episode with at least one written parquet row — the "known frame" this
    check slices — or `(None, None)` if no such episode exists."""
    for episode_index in ctx.episode_indices:
        path = ctx.parquet_path(episode_index)
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["observation.state", "action"])
        if table.num_rows > 0:
            return episode_index, table
    return None, None


def check_modality_slice_range(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset. Takes the first row of the
    lowest-indexed episode with at least one written parquet row and, for
    each declared field, slices `observation.state`/`action` by the ON-DISK
    `meta/modality.json` range, then confirms every sliced value lies
    within the UNION of the per-channel `min`/`max` bounds `meta/stats.json`
    records at the field's INDEPENDENTLY-derived (`derive_slice_map_from_profile`)
    index range — never the on-disk range under test. A shifted on-disk
    slice map pulls values from a different joint group whose bounds
    generally differ, which is exactly what this comparison is built to
    catch.

    Deliberately a coarse (union, not per-channel) bound: when two adjacent
    groups happen to share identical per-channel bounds, a shift between
    them cannot be separated by this range test — named in every evaluated
    field's evidence under `blind_spot`, never left unstated. The profile
    declares no per-joint POSITION limits (only optional velocity/
    acceleration fields, which bound rates, not positions), so the range
    source here is the measured `meta/stats.json` per-channel bounds, never
    a declared physical limit.
    """
    try:
        episode_index, table = _find_known_frame(ctx)
        if table is None:
            reason = (
                "zero-episode dataset — no known frame exists to slice"
                if not ctx.episode_indices
                else "no episode has at least one written parquet row"
            )
            return (_unevaluated_slice_range(reason=reason),)

        with open(ctx.out_dir / "meta" / "modality.json", encoding="utf-8") as f:
            modality_json = json.load(f)
        with open(ctx.out_dir / "meta" / "stats.json", encoding="utf-8") as f:
            stats_payload = json.load(f)

        state_row = np.asarray(table.column("observation.state")[0].as_py(), dtype=np.float32)
        action_row = np.asarray(table.column("action")[0].as_py(), dtype=np.float32)

        per_field_evidence: list[dict[str, Any]] = []
        adjacency_violations: list[dict[str, Any]] = []
        any_failure = False

        for scope, row, feature_name, order_keys in (
            ("state", state_row, "observation.state", list(ctx.profile.state_vector_order)),
            ("action", action_row, "action", list(ctx.profile.action_primary)),
        ):
            actual_ranges = modality_json.get(scope, {})
            expected_ranges = derive_slice_map_from_profile(ctx.profile, scope)
            stats_entry = stats_payload.get(feature_name, {})
            min_list = [float(x) for x in stats_entry.get("min", [])]
            max_list = [float(x) for x in stats_entry.get("max", [])]
            row_width = int(row.shape[0])

            scope_adjacency = _adjacency_violations(scope, actual_ranges)
            adjacency_violations.extend(scope_adjacency)
            if scope_adjacency:
                any_failure = True

            sorted_keys = sorted(
                (k for k in order_keys if k in actual_ranges),
                key=lambda k: actual_ranges[k].get("start", 0),
            )
            for key in sorted_keys:
                actual_range = actual_ranges[key]
                actual_start = int(actual_range.get("start", -1))
                actual_end = int(actual_range.get("end", -1))
                actual_width = actual_end - actual_start

                field_evidence: dict[str, Any] = {
                    "scope": scope,
                    "field": key,
                    "actual_range": {"start": actual_start, "end": actual_end},
                }

                if actual_width <= 0:
                    field_evidence.update({"passed": False, "reason": "zero_width_field"})
                    per_field_evidence.append(field_evidence)
                    any_failure = True
                    continue

                if actual_end > row_width:
                    field_evidence.update(
                        {
                            "passed": False,
                            "reason": "slice_end_exceeds_vector_width",
                            "actual_parquet_column_width": row_width,
                        }
                    )
                    per_field_evidence.append(field_evidence)
                    any_failure = True
                    continue

                expected_range = expected_ranges.get(key)
                if expected_range is None:
                    field_evidence.update(
                        {"passed": False, "reason": "field_not_derivable_from_profile"}
                    )
                    per_field_evidence.append(field_evidence)
                    any_failure = True
                    continue

                expected_start = expected_range["start"]
                expected_end = expected_range["end"]
                if expected_end > len(min_list) or expected_end > len(max_list):
                    field_evidence.update(
                        {
                            "passed": False,
                            "reason": "stats_json_has_too_few_channels_for_profile_derived_range",
                            "expected_range": expected_range,
                        }
                    )
                    per_field_evidence.append(field_evidence)
                    any_failure = True
                    continue

                field_min = min(min_list[expected_start:expected_end])
                field_max = max(max_list[expected_start:expected_end])
                sliced_values = row[actual_start:actual_end]
                within_bounds = bool(
                    np.all((sliced_values >= field_min) & (sliced_values <= field_max))
                )

                field_evidence.update(
                    {
                        "passed": within_bounds,
                        "expected_range": expected_range,
                        "field_min": field_min,
                        "field_max": field_max,
                        "sliced_values": [float(v) for v in sliced_values],
                        "blind_spot": (
                            "when the profile-derived range for an adjacent field shares "
                            "identical per-channel bounds with this field, a shift between "
                            "them cannot be separated by this range test; the profile "
                            "declares no per-joint POSITION limits (only optional velocity/"
                            "acceleration fields, which bound rates, not positions), so this "
                            "bound is the measured meta/stats.json per-channel range, never a "
                            "declared physical limit"
                        ),
                    }
                )
                if not within_bounds:
                    field_evidence["reason"] = "value_out_of_range"
                    any_failure = True
                per_field_evidence.append(field_evidence)

        return (
            CheckResult(
                CHECK_NAME_SLICE_RANGE,
                "fatal",
                not any_failure,
                {
                    "episode_index": episode_index,
                    "per_field": per_field_evidence,
                    "adjacency_violations": adjacency_violations,
                },
            ),
        )
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (
            CheckResult(CHECK_NAME_SLICE_RANGE, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),
        )
