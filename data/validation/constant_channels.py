"""VALID-04/D-07/D-08: dataset-wide constant-channel detection.

**Scope is dataset-wide (D-07).** GR00T's default min-max normalization
divides by `max - min` using the DATASET-WIDE merged statistics, so a
channel that is constant within one episode but varies across the dataset
is globally fine — only a channel whose merged `min == max` across every
included episode is the divide-by-zero hazard. `load_merged_partials`
reconstructs that dataset-wide merge from the writer's own persisted
per-episode partials (`meta/stats_partials/episode_*.json`) via
`data/writers/stats.py::merge_partials` — never a fresh reduction, which
would be a second implementation of the arithmetic under test.

**Constancy is exact.** A channel fails when its merged `min == max` as a
float64 equality — no variance floor, no approximate-equality helper, no
relative tolerance. The hazard is literally `max - min == 0`, so exact
equality is the defensible test.

**Severity is advisory, superseding D-08's original "no escape hatch"
stance (G-03).** Phase 3's D-08 originally stated: "There is no
acknowledge-and-continue profile field, config flag, or severity downgrade
for an acknowledged constant channel anywhere in this module or elsewhere
under `data/`." G-03 supersedes exactly that paragraph, on measured
evidence: three real Galaxea recordings (WINDOWS.md id 8) in which the
teleoperator did not move the left arm, leaving `arm_l_j1`, `arm_l_j5`,
`arm_l_j7` and every left-arm action channel exactly constant across all
three episodes — a true property of those samples, not a code defect.

What changed: the severity, from `fatal` to `advisory`, so a legitimately
idle limb in a real recording no longer fails the whole batch fatally on
its own. What did NOT change: constancy is still tested by exact `min ==
max` equality with no variance floor, the arithmetic hazard is unchanged
(GR00T's default min-max normalization still divides by `max - min`, which
is still zero on a flagged channel), and acknowledging a channel still does
not fix that divide-by-zero — it only authorizes proceeding with a named,
recorded risk. The acknowledgement itself does not live in this module: it
lives in `data/cli.py`'s constant-channel acceptance gate
(`resolve_constant_channel_acceptance`) and its always-present run-report
`constant_channel_acceptance` block, so a reader of this file should not
mistake the severity downgrade alone for the whole story. The author still
removes a channel for good — the only fix for the underlying hazard — by
editing the profile key this check names in its evidence
(`state_vector_order` / `action_primary`).

**The measured emitted width is a recorded finding, never a literal
(D-09).** `emitted_width_measured` is computed by summing
`channel_names_for_feature` over both features — no numeric total is ever
written down as a literal in this module.
"""

import json
from typing import Any

import numpy as np

from data.contracts.containers import CheckResult
from data.contracts.profile import EmbodimentProfile
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.writers import stats

CHECK_NAME = "statistics.constant_channel"

FEATURES = ("observation.state", "action")
"""The only two float features a constant channel can meaningfully apply
to — a camera feature carries `dtype: "video"` and is never a candidate."""

PROFILE_KEY_FOR_FEATURE = {
    "observation.state": "state_vector_order",
    "action": "action_primary",
}
"""Maps each feature name to the profile field an author edits to remove a
constant channel — derived once here so the removal instruction in a
flagged channel's evidence is never typed twice."""


def channel_names_for_feature(profile: EmbodimentProfile, feature: str) -> list[str]:
    """The flat-index-to-joint-name mapping for `feature`, built from the
    profile's own declared ordering — the same construction
    `data/writers/lerobot_v21.py`'s `_state_names`/`_action_names` use
    (via `_joint_names_for_state`/`_joint_names_for_action`), reproduced
    here rather than imported so this module stays independent of the
    writer's private helpers."""
    if feature == "observation.state":
        state_by_key = {f.state_key: f for f in profile.state_topics.values()}
        names: list[str] = []
        for key in profile.state_vector_order:
            names.extend(state_by_key[key].joint_names)
        return names
    elif feature == "action":
        action_by_key = {f.action_key: f for f in profile.action_topics.values()}
        action_names: list[str] = []
        for key in profile.action_primary:
            action_names.extend(action_by_key[key].joint_names)
        return action_names
    else:
        raise ValueError(f"channel_names_for_feature: unknown feature {feature!r}")


def load_merged_partials(ctx: ValidationContext) -> dict[str, stats.WelfordPartial]:
    """Reads every `meta/stats_partials/episode_*.json`, reconstructs each
    feature's per-episode `WelfordPartial` via `WelfordPartial.from_json_dict`
    (supplying the persisted `values` array), honours each file's own
    recorded `included_in_merge` flag (an episode this writer excluded from
    the dataset-wide merge — e.g. a `REJECT`-dispositioned episode —
    contributes nothing here either, matching the writer's own rule
    exactly), and folds every included partial with `merge_partials`. This
    is the same dataset-wide merge `LeRobotV21Writer.finalize()` computes
    when writing `meta/stats.json` — reusing `merge_partials` rather than a
    fresh reduction is what makes that agreement mechanical, not
    conventional."""
    merged: dict[str, stats.WelfordPartial] = {}
    partials_dir = ctx.out_dir / "meta" / "stats_partials"
    if not partials_dir.is_dir():
        return merged

    for path in sorted(partials_dir.glob("episode_*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not payload.get("included_in_merge", True):
            continue
        for feature, entry in payload.get("features", {}).items():
            values = entry.get("values", [])
            partial = stats.WelfordPartial.from_json_dict(entry, values=_to_values_array(values))
            merged[feature] = (
                partial if feature not in merged else stats.merge_partials(merged[feature], partial)
            )
    return merged


def _to_values_array(values: list[list[float]]) -> np.ndarray | None:
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def _classify_likely_cause(min_value: float) -> str:
    """Distinguishes the two situations a flagged channel's constant value
    can plausibly represent — informational only; both dispose as failures
    regardless of which applies (see module/`check_constant_channel`
    docstrings). A channel whose single recorded value is exactly zero
    reads as a likely dead sensor (no real sensor is ever driven to
    literal zero for an entire dataset by design); any other constant
    value reads as possibly an intentionally still joint, removable from
    the declared ordering."""
    return "likely_dead_sensor" if min_value == 0.0 else "possibly_intentionally_still"


def check_constant_channel(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset. Merges every included
    episode's persisted partial per feature (`load_merged_partials`), then
    flags every channel whose merged `min` exactly equals its `max`.

    A feature with zero declared width, or whose merged partial has zero
    count (no episode contributed one — e.g. every episode carried a
    `REJECT` disposition), is recorded as unevaluated for that feature. If
    **every** feature ends up unevaluated, the whole check reports
    `UNEVALUATED_KEY` true rather than a bare pass — there is nothing here
    to divide by.
    """
    try:
        merged = load_merged_partials(ctx)
    except Exception as e:  # noqa: BLE001 - reported not raised
        # A partial-loading failure has no basis for a verdict at all -- it
        # is visibly unevaluated (the UNEVALUATED_KEY marker plus a skip
        # reason naming the exception), not an invisible advisory failure
        # that would merely carry `passed=False` with an error buried in
        # evidence. This is a small honesty fix the G-03 severity downgrade
        # makes necessary: an advisory check that silently `passed=False`
        # on a load error would be invisible in both the exit code and
        # `fully_validated`.
        skip_reason = f"{type(e).__name__}: {e}"
        return (
            CheckResult(
                CHECK_NAME,
                "advisory",
                False,
                {
                    UNEVALUATED_KEY: True,
                    SKIP_REASON_KEY: skip_reason,
                    "error": skip_reason,
                },
            ),
        )

    # `emitted_width_measured` names the SAME quantity STATE.md's own
    # unreconciled figures (26 nominal / 20 measured / ~19-post-removal)
    # track — the emitted `observation.state` vector's width, measured by
    # summing `state_vector_order`, never a literal (D-09). `action`'s
    # width is recorded alongside it under its own key for completeness,
    # not folded into the same number: the two features are validated by
    # the identical mechanism (D-08) but are two separate, independently
    # measured emitted widths, not one combined total.
    emitted_width_measured = len(channel_names_for_feature(ctx.profile, "observation.state"))
    action_emitted_width_measured = len(channel_names_for_feature(ctx.profile, "action"))

    flagged: list[dict[str, Any]] = []
    unevaluated_features: list[dict[str, Any]] = []

    for feature in FEATURES:
        names = channel_names_for_feature(ctx.profile, feature)
        width = len(names)
        if width == 0:
            unevaluated_features.append({"feature": feature, "reason": "zero-width feature"})
            continue

        partial = merged.get(feature)
        if partial is None or partial.count == 0:
            unevaluated_features.append(
                {"feature": feature, "reason": "merged partial has zero count"}
            )
            continue

        for idx in range(width):
            min_value = float(partial.min[idx])
            max_value = float(partial.max[idx])
            if min_value == max_value:
                flagged.append(
                    {
                        "feature": feature,
                        "channel_index": idx,
                        "joint_name": names[idx],
                        "profile_key": PROFILE_KEY_FOR_FEATURE[feature],
                        "merged_min": min_value,
                        "merged_max": max_value,
                        "likely_cause": _classify_likely_cause(min_value),
                    }
                )

    if len(unevaluated_features) == len(FEATURES):
        reason = "; ".join(f"{u['feature']}: {u['reason']}" for u in unevaluated_features)
        return (
            CheckResult(
                CHECK_NAME,
                "advisory",
                False,
                {
                    UNEVALUATED_KEY: True,
                    SKIP_REASON_KEY: reason,
                    "unevaluated_features": unevaluated_features,
                    "emitted_width_measured": emitted_width_measured,
                    "action_emitted_width_measured": action_emitted_width_measured,
                },
            ),
        )

    return (
        CheckResult(
            CHECK_NAME,
            "advisory",
            not flagged,
            {
                "flagged_channels": flagged,
                "emitted_width_measured": emitted_width_measured,
                "action_emitted_width_measured": action_emitted_width_measured,
                "unevaluated_features": unevaluated_features,
                "verdict_independent_of_likely_cause": (
                    "both likely_dead_sensor and possibly_intentionally_still "
                    "channels fail identically — intent does not change the "
                    "divide-by-zero arithmetic"
                ),
            },
        ),
    )
