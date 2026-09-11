"""VALID-03: timestep-lag correlation between the state delta and the paired
action, over profile-derived, width-matched channels.

The prior design here ran a **single flattened correlation across every
channel** and argued that was the correct reading of the requirement. It was
not: `03-VERIFICATION.md`'s independent re-run at multiple frame counts found
it false-positived on genuinely correct data and reported the *identical*
`peak_lag` for clean and shifted inputs at several counts — the check could
not discriminate the exact defect it exists to catch. Root cause was twofold:
(1) the shipped Galaxea profile's real `observation.state` (20) and `action`
(16) widths differ, so a flattened lag runs over samples, not timesteps — a
one-timestep action rotation lands the peak at ±16 (the action width), not
±1; (2) the fixtures both this check's own tests and the writer-adapter test
suite build from tend to have a **constant per-channel delta** (a linear ramp
in the raw source), which is a genuinely degenerate signal with no defined
correlation peak — `np.argmax` breaks the tie arbitrarily, which is exactly
why the measured lag wandered across runs.

The redesign here does three things, composed, not as alternatives:

1. **Width-match by profile.** `action_aligned_state_columns` derives, from
   `profile.state_vector_order`/`profile.action_primary`, exactly the state
   columns that pair channel-for-channel with the action column, in
   `action_primary`'s own order (the action column's own on-disk layout is
   built from `action_primary`, so pairing in any other order would silently
   mismatch channels).
2. **Search the lag in timesteps**, not flattened samples, over the
   width-matched channels: `peak_lag_timesteps` shifts by whole rows and
   scores each shift with `lag_score`, a per-channel Pearson correlation
   mean-centred and unit-normalized over that shift's own overlapping rows
   (the overlap shrinks as the candidate lag grows, so an unnormalized score
   would shrink with lag and bias the argmax toward zero — normalizing per
   lag is what keeps the comparison meaningful across lags).
3. **An explicit degenerate-signal branch.** When no aligned channel carries
   variance on both sides at any searched lag — the constant-per-channel-
   delta case above — there is no defined correlation peak, and the check
   reports `UNEVALUATED_KEY` true rather than a `passed=False` that would
   otherwise measure nothing but which tied index `np.argmax` happened to
   pick.

`REQUIRED_LAG` stays 0 and stays exact: no tolerance window is introduced. A
tolerance wide enough to absorb the old false positives would also be wide
enough to admit the off-by-one this check exists to catch — the false
positives were a width/degeneracy problem, not a tolerance problem.

**Superseded by measurement (plan 03-11, D-GAP-01 Defect B, VALID-03).**
The redesign above fixed the false positives measured against SYNTHETIC
fixtures. Measured against the three REAL Galaxea recordings under
`data/samples/vr-teleop-20260904/` (`data/tools/measure_action_state_pairing.py`
— the permanent tool that reproduces every number below), the pairing itself
turned out to be wrong on real robot dynamics, independent of the
width/degeneracy fix:

1. **The action source is an absolute POSITION target, not a delta.**
   `median|action| = 0.636` tracks `median|state| = 0.635` to within ~0.4%
   on the right arm's real motion, while the per-sample state delta
   (`median|diff(state)| = 0.00085`) is ~750x smaller. `action_aligned_state_columns`
   correctly width-matches the two, but `diff(state)` vs `action[:-1]`
   (`REQUIRED_LAG`'s own pairing above) compares a delta against an
   absolute position — a category error, not a lag error.
2. **That shipped pairing has no interior peak on real data.** `peak_lag_timesteps`
   on `diff(state)` vs `action[:-1]` returns the search BOUNDARY
   (`peak_lag = 5` at `MAX_LAG_SEARCH`) in every window of every real
   recording, with the score rising monotonically toward it — there is no
   defined maximum inside the searched window at all.
3. **The real robot's target stream LEADS its own achieved feedback by
   about four camera ticks.** `argmin_L median|action[t+L] - state[t]|` at
   the persisted ~30Hz grid rate measures `argmin_lag = -4` in all three
   real recordings, moving to `-5` under a deliberate one-tick rotation of
   the action series (exactly the off-by-one detector needs), with the
   tracking error at the optimum ~6x below its value at lag 0. This is
   consistent with the vendor's own `hdas_msg/msg/MotorControl` definition
   (`p_des`, `v_des`, `kp`, `kd`, `t_ff`) — a PD-plus-feedforward command, so
   the joint accelerates in proportion to the tracking error
   `p_des - p_actual`, and a non-zero optimum lag is the expected
   consequence of that control law, not a tolerance bolted on to force a
   pass. The `log_time`/`header.stamp` clock skew that could have
   contaminated this number with recorder buffering latency was measured
   and found immaterial (~0.01 grid periods median; `03-11-SUMMARY.md`
   Finding 7) — the -4 is the robot's own actuation lag, not a recording
   artifact.

The developer decided, at the `gate="blocking-human"` checkpoint carrying
this evidence (`03-11-PLAN.md` Task 2), on **Option B**: declare the
expected lag on the embodiment profile
(`EmbodimentProfile.expected_action_state_lag`) and compare it against the
tracking-error-minimizing lag (`tracking_error_lag`, below) rather than the
zero-lag correlation peak. `REQUIRED_LAG`/`peak_lag_timesteps`/`lag_score`
below are the SUPERSEDED design, kept only because
`data/tools/measure_action_state_pairing.py` still imports
`peak_lag_timesteps`/`MAX_LAG_SEARCH` for its own diagnostic
`CANDIDATE_PAIRINGS` comparison (never reimplemented, per this project's own
D-12 discipline) — `check_action_state_lag` itself no longer calls them.

Every artifact open (parquet metadata/read) is wrapped in a `try`/`except`
scoped to the parquet library's own exception types and converted to
`passed=False` with the exception text under an `error` evidence key —
mirroring `data/adapters/checks.py::_check_mcap_crc`'s catch-and-report shape,
matching `data/validation/frame_integrity.py`/`inventory.py`'s own idiom — so
a deliberately corrupted fixture surfaces as one failed named check, never an
aborted run.
"""

from typing import Any

import numpy as np
import pyarrow.parquet as pq

from data.contracts.containers import CheckResult
from data.contracts.profile import EmbodimentProfile
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY

CHECK_NAME = "consistency.action_state_lag"

REQUIRED_LAG = 0
"""VALID-03's pass criterion: the measured peak lag, in timesteps, must equal
this value exactly — no tolerance, no rounding."""

MAX_LAG_SEARCH = 5
"""The maximum timestep-lag magnitude the search considers in either
direction. Bounds the search cost to be linear in row count regardless of
episode length (T-03-06-04) — the check body further tightens this per
episode so the largest searched lag always leaves at least 2 overlapping
rows (see `check_action_state_lag`)."""


def action_aligned_state_columns(profile: EmbodimentProfile) -> list[int] | None:
    """The `observation.state` column indices that pair channel-for-channel
    with `action`'s own channels, **in `profile.action_primary`'s order** —
    not `state_vector_order`'s order. The order matters and is the whole
    point of this function: the `action` column's own on-disk layout is
    built from `action_primary` (`data/writers/lerobot_v21.py`'s action
    layout helper), so pairing channel i of the returned selection with
    channel i of the action column is only correct in that order.

    Every width and offset is derived by summing `joint_names` over
    `profile.state_vector_order` — never a width literal or a hardcoded
    column-index list. Returns `None` (no selection) when an
    `action_primary` key is absent from `state_vector_order`, or when its
    state-side and action-side `joint_names` lengths disagree: a mis-paired
    channel comparison would be worse than an absent one, so the caller
    turns a `None` here into an unevaluated result rather than proceeding.
    """
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for state_key in profile.state_vector_order:
        width = len(by_state_key[state_key].joint_names)
        offsets[state_key] = (offset, width)
        offset += width

    by_action_key = {f.action_key: f for f in profile.action_topics.values()}

    columns: list[int] = []
    for action_key in profile.action_primary:
        pair = offsets.get(action_key)
        action_field = by_action_key.get(action_key)
        if pair is None or action_field is None:
            return None
        state_start, state_width = pair
        if state_width != len(action_field.joint_names):
            return None
        columns.extend(range(state_start, state_start + state_width))
    return columns


def lag_score(state_delta: np.ndarray, action_trim: np.ndarray, lag: int) -> float | None:
    """The normalized per-channel score at one integer timestep `lag`:
    shifts `action_trim` by `lag` whole rows relative to `state_delta`
    (`lag > 0` compares `state_delta[t]` against `action_trim[t + lag]`;
    `lag < 0` the mirror), takes the overlapping rows of each, and returns
    the mean over channels of the per-channel Pearson correlation —
    mean-centre each channel over the overlap, divide by its own L2 norm
    over that same overlap, multiply the two, sum down rows, average across
    channels.

    Scores only the channels where BOTH sides carry variance,
    OVER-THE-OVERLAP, that is distinguishable from float32 representation
    noise — see `_VARIANCE_NOISE_FLOOR` below. Returns `None` when the
    overlap is under 2 rows or when no channel qualifies — there is no
    defined correlation for a channel that is constant (or effectively so)
    over the window, and including it would silently pull the mean toward
    zero rather than reporting "not scoreable" honestly.

    The per-lag renormalization (dividing by each side's own norm over that
    lag's own overlap) is required, not cosmetic: each candidate lag has a
    different overlap length, so an unnormalized score would shrink with
    lag and bias the argmax toward zero, manufacturing a false pass.
    """
    n = min(state_delta.shape[0], action_trim.shape[0])
    if lag >= 0:
        n_eff = max(n - lag, 0)
        s = state_delta[:n_eff]
        a = action_trim[lag : lag + n_eff]
    else:
        shift = -lag
        n_eff = max(n - shift, 0)
        s = state_delta[shift : shift + n_eff]
        a = action_trim[:n_eff]

    if n_eff < 2:
        return None

    s_centered = s - s.mean(axis=0, keepdims=True)
    a_centered = a - a.mean(axis=0, keepdims=True)
    s_norm = np.sqrt((s_centered**2).sum(axis=0))
    a_norm = np.sqrt((a_centered**2).sum(axis=0))
    valid = _carries_variance(s, s_norm) & _carries_variance(a, a_norm)
    if not np.any(valid):
        return None

    numerator = (s_centered[:, valid] * a_centered[:, valid]).sum(axis=0)
    denominator = s_norm[valid] * a_norm[valid]
    per_channel = numerator / denominator
    return float(per_channel.mean())


_VARIANCE_NOISE_FLOOR = 1e-4
"""A channel's centered-norm, relative to its own raw magnitude, must exceed
this to count as carrying variance. Measured empirically against
`data/tests/mcap_fixture.py`'s default per-channel ramp
(`1.0 + i*0.1 + j*0.01`) run through the real synchronizer
(`build_varying_baseline_dataset`): the true signal is a CONSTANT per-channel
delta, but linear interpolation onto the camera clock does not reproduce that
constant exactly in float32 — the persisted values differ by ~1e-7 relative
noise (float32's own representable-precision limit at this magnitude), not by
any real motion. An exact `norm > 0` test (the naive reading) treats that
noise as a real signal and hands `np.argmax` an effectively flat surface to
break the tie on — reproducing the exact false-positive/false-tie failure
mode this redesign exists to close, just one level down. `1e-4` sits three
orders of magnitude above that measured noise floor and three-to-four orders
below this project's real per-timestep motion scale (this plan's own
random-walk test fixtures move by ~0.1-1 per step), so it separates the two
without risk of masking genuine (if slow) motion."""


def _carries_variance(raw: np.ndarray, centered_norm: np.ndarray) -> np.ndarray:
    """Per-channel boolean: does `raw` (the RAW, uncentered slice used to
    compute `centered_norm`) vary by more than `_VARIANCE_NOISE_FLOOR`,
    relative to its own magnitude? `scale` floors at 1.0 so a near-zero-
    magnitude feature (e.g. a gripper channel hovering near 0) still gets an
    absolute noise floor rather than dividing by a near-zero scale."""
    scale = np.maximum(np.abs(raw).max(axis=0), 1.0)
    return centered_norm > _VARIANCE_NOISE_FLOOR * scale * np.sqrt(raw.shape[0])


def peak_lag_timesteps(
    state_delta: np.ndarray, action_trim: np.ndarray, max_lag: int
) -> tuple[int | None, dict[int, float]]:
    """Scores every integer lag in `-max_lag .. +max_lag` via `lag_score`,
    drops the unscoreable ones, and returns `(best_lag, scores)` — the lag
    with the maximum score, plus the full per-lag score mapping (only the
    scoreable lags) for evidence. Returns `(None, {})` when no lag in the
    window is scoreable at all (the degenerate-signal case)."""
    scores: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        score = lag_score(state_delta, action_trim, lag)
        if score is not None:
            scores[lag] = score
    if not scores:
        return None, {}
    best_lag = max(scores, key=lambda candidate: scores[candidate])
    return best_lag, scores


def tracking_error_lag(
    state: np.ndarray, action: np.ndarray, max_lag: int
) -> tuple[int | None, dict[int, float]]:
    """VALID-03 Option B's statistic (plan 03-11's developer decision,
    module docstring above): per integer lag `L` in `-max_lag..+max_lag`,
    the MEAN-ACROSS-CHANNELS of each channel's own MEDIAN-OVER-ROWS
    absolute difference `|action[t + L] - state[t]|`, skipping any lag
    whose overlap is under 2 rows. Returns `(argmin_lag, scores)` — the lag
    with the MINIMUM tracking error (not the correlation MAXIMUM
    `peak_lag_timesteps` searches for), plus the full per-lag score map for
    evidence. Returns `(None, {})` when no lag in the window has a 2-row
    overlap.

    Neither a flat mean nor a flat median over every action-aligned
    channel AND row at once, and deliberately so — both measured broken on
    real Galaxea data (plan 03-11 Task 3, empirically compared against
    `data/tools/measure_action_state_pairing.py::tracking_error_profile`'s
    own per-action-key median, the statistic Finding 4 measured `-4` with):
    a FLAT MEDIAN is swamped by the shipped profile's 9 exactly-constant
    action-aligned channels (Finding 5 — the left arm and both grippers
    are all-zero in every real recording, 9 of 16 channels) — with more
    than half the flattened values identically zero at every lag, the
    order statistic falls in the zero-majority cluster and returns zero at
    every lag, erasing the signal the other 7 (right-arm) channels carry.
    A FLAT MEAN survives that (a channel identically zero on both sides
    contributes an exact, lag-invariant zero, diluting magnitude but never
    moving the argmin) but is NOT robust to a single-recording outlier
    sample within the genuinely-moving channels — measured to disagree
    with the tool's own per-channel-median Finding on real right-arm data
    (`-5` vs `-4`, a margin of ~2%, well within one noisy sample's reach).
    MEDIAN-then-MEAN composes both robustness properties: each channel's
    own median is insensitive to that channel's own outlier rows, and the
    cross-channel mean cannot be swamped by a degenerate channel's exact
    zero the way a cross-channel-AND-row median can.

    Unlike `lag_score`, this is a plain magnitude functional with NO
    variance requirement: it is well-defined even over a constant
    (zero-variance) channel, because it measures `action`'s absolute
    proximity to `state`, not their correlation — exactly why it can detect
    the real recordings' target-lead lag (module docstring point 3) where
    the correlation-based `peak_lag_timesteps` search found no interior
    peak at all (module docstring point 2).

    Mirrors `lag_score`'s own shift convention (`lag >= 0` compares
    `state[t]` against `action[t + lag]`; `lag < 0` the mirror) and
    `data/tools/measure_action_state_pairing.py::tracking_error_profile`'s
    own arithmetic, so a value here is directly comparable to what that
    tool measures against a real recording (per single action key — that
    tool never flattens multiple action keys together the way this
    function's caller does).
    """
    n = min(state.shape[0], action.shape[0])
    scores: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            n_eff = n - lag
            if n_eff < 2:
                continue
            s = state[:n_eff]
            a = action[lag : lag + n_eff]
        else:
            shift = -lag
            n_eff = n - shift
            if n_eff < 2:
                continue
            s = state[shift : shift + n_eff]
            a = action[:n_eff]
        per_channel_median = np.median(np.abs(a - s), axis=0)
        scores[lag] = float(np.mean(per_channel_median))
    if not scores:
        return None, {}
    best_lag = min(scores, key=lambda candidate: scores[candidate])
    return best_lag, scores


def _read_fixed_size_list_column(table: Any, name: str) -> np.ndarray:
    """Reads a `pa.list_(pa.float32(), width)` column back as a `(rows,
    width)` float64 array. `width` may be 0 (a legitimate world-model-style
    empty feature) — `np.asarray` on a list of zero-length lists still
    yields a well-shaped `(rows, 0)` array, never raising."""
    field_type = table.schema.field(name).type
    width = int(field_type.list_size)
    rows = table.column(name).to_pylist()
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), width)


def _unevaluated(*, reason: str, row_count: int) -> CheckResult:
    return CheckResult(
        CHECK_NAME,
        "fatal",
        False,
        {UNEVALUATED_KEY: True, SKIP_REASON_KEY: reason, "row_count": row_count},
    )


def check_action_state_lag(
    ctx: ValidationContext, episode_index: int
) -> tuple[CheckResult, ...]:
    """One `CheckResult` per episode. Reads the episode's parquet
    `observation.state` and `action` fixed-size-list columns back as
    `(rows, width)` float arrays, selects the profile-derived action-aligned
    state columns, and compares the RAW (non-differenced) selected state
    against the RAW action via `tracking_error_lag` — VALID-03 Option B
    (module docstring above): the measured tracking-error-minimizing lag
    must equal `ctx.profile.expected_action_state_lag` exactly.

    Branches explicitly, before any reduction, in this order: no parquet
    written (a zero-frame episode); a parquet read/metadata error; fewer
    than two rows; a zero-width state or action column; a profile-derived
    channel mis-pairing (an `action_primary` key absent from
    `state_vector_order`, a joint-count mismatch between the two, or the
    aligned selection's width disagreeing with the actual on-disk action
    width); no `expected_action_state_lag` declared on the profile
    (`data/adapters/checks.py`'s `no_limit_declared` precedent, Phase 2
    D-08 — a check never reports passing without a declared expectation);
    too few rows for any non-zero lag to be scoreable; and — distinct from
    that last guard — a row count too small for the search window to reach
    the declared lag's own magnitude (`abs(expected_lag) > max_lag`; CR-02,
    03-REVIEW.md). That last guard exists because `max_lag` is sized purely
    from `row_count` and `MAX_LAG_SEARCH`, never from `expected_lag`: on the
    shipped `galaxea_r1_pro.yaml` profile (`expected_action_state_lag=-4`),
    a 4-6 row episode has `max_lag` in `{1,2,3}`, none of which can ever
    equal `-4` — without this guard, a genuinely correct short episode would
    be falsely REJECTed rather than honestly abstained. Every one of these
    branches yields `UNEVALUATED_KEY` true with a named reason rather than a
    `passed=False` that would otherwise measure nothing.
    """
    parquet_path = ctx.parquet_path(episode_index)
    if not parquet_path.exists():
        return (_unevaluated(reason="no parquet written for this episode", row_count=0),)

    try:
        table = pq.read_table(parquet_path)
        state = _read_fixed_size_list_column(table, "observation.state")
        action = _read_fixed_size_list_column(table, "action")
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {"error": f"{type(e).__name__}: {e}"},
            ),
        )

    row_count = state.shape[0]
    state_width = state.shape[1]
    action_width = action.shape[1]

    if row_count < 2:
        return (_unevaluated(reason=f"row_count {row_count} < 2", row_count=row_count),)
    if state_width == 0 or action_width == 0:
        return (
            _unevaluated(
                reason=(
                    f"zero-width feature (state_width={state_width}, "
                    f"action_width={action_width})"
                ),
                row_count=row_count,
            ),
        )

    aligned_columns = action_aligned_state_columns(ctx.profile)
    aligned_width = len(aligned_columns) if aligned_columns is not None else 0
    if aligned_columns is None or aligned_width != action_width:
        return (
            _unevaluated(
                reason=(
                    "profile-derived action-aligned state columns "
                    f"(width={aligned_width}) do not match the action column "
                    f"(width={action_width}) — an action_primary key is absent "
                    "from state_vector_order, or its state-side and action-side "
                    "joint_names lengths disagree"
                ),
                row_count=row_count,
            ),
        )

    expected_lag = ctx.profile.expected_action_state_lag
    if expected_lag is None:
        return (
            _unevaluated(
                reason=(
                    "no expected_action_state_lag declared on the embodiment "
                    "profile — consistency.action_state_lag has nothing to "
                    "compare a measured lag against (mirrors "
                    "data/adapters/checks.py's no_limit_declared precedent)"
                ),
                row_count=row_count,
            ),
        )

    max_lag = min(MAX_LAG_SEARCH, row_count - 3)
    if max_lag < 1:
        return (
            _unevaluated(
                reason=(
                    f"row_count {row_count} too small for any non-zero lag to be "
                    "scoreable"
                ),
                row_count=row_count,
            ),
        )

    required_lag_magnitude = abs(expected_lag)
    if required_lag_magnitude > max_lag:
        return (
            _unevaluated(
                reason=(
                    f"declared expected_action_state_lag={expected_lag} has magnitude "
                    f"{required_lag_magnitude}, which exceeds max_lag={max_lag} "
                    f"(min(MAX_LAG_SEARCH={MAX_LAG_SEARCH}, row_count-3) for "
                    f"row_count={row_count}) — this episode is too short to test the "
                    "declared lag, so the declared expectation cannot be evaluated "
                    "(distinct from the 'too few rows for any non-zero lag' guard above)"
                ),
                row_count=row_count,
            ),
        )

    state_sel = state[:, aligned_columns]

    lag, lag_scores = tracking_error_lag(state_sel, action, max_lag)
    if lag is None:
        return (
            _unevaluated(
                reason=(
                    f"row_count {row_count} left no lag in "
                    f"[-{max_lag}, {max_lag}] with a 2-row overlap"
                ),
                row_count=row_count,
            ),
        )

    passed = lag == expected_lag

    return (
        CheckResult(
            CHECK_NAME,
            "fatal",
            passed,
            {
                "measured_lag": lag,
                "expected_lag": expected_lag,
                "row_count": row_count,
                "state_width": state_width,
                "action_width": action_width,
                "aligned_width": aligned_width,
                "max_lag_searched": max_lag,
                "tracking_error_scores": {str(k): float(v) for k, v in lag_scores.items()},
                "error_at_expected_lag": lag_scores.get(expected_lag),
                "error_at_measured_lag": lag_scores.get(lag),
                "lag_unit": "timesteps",
                "statistic": "tracking_error_argmin",
            },
        ),
    )
