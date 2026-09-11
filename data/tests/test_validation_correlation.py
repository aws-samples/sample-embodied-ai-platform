"""VALID-03 tests: the declared expected lag between `observation.state`
and its paired `action`, over profile-derived, width-matched channels.

**Superseded pairing, superseded criterion (plan 03-11, D-GAP-01 Defect B).**
This file originally tested a timestep-lag CORRELATION between the state
DELTA and the paired action, required to peak at exactly lag 0
(`REQUIRED_LAG`). Measured against the three real Galaxea recordings, that
pairing was wrong on two independent axes: the action source is an
absolute POSITION target, not a delta (comparing it against `diff(state)`
is a category error), and the real robot's target stream LEADS its own
achieved feedback by a nonzero, physically-expected lag (not zero) — see
`data/validation/correlation.py`'s module docstring for the full measured
evidence and `03-11-SUMMARY.md` for the developer's decision record. The
developer selected Option B: compare `tracking_error_lag` (RAW action
against RAW state, no delta) against a profile-declared
`expected_action_state_lag` (`-4` on the shipped `galaxea_r1_pro` profile).
Evidence keys changed accordingly: `peak_lag` -> `measured_lag`,
`required_lag` -> `expected_lag`.

Every "clean passes" fixture below now uses an ABSOLUTE-POSITION-TARGET
pairing (`action[t] == state[t - lag]`, clipped at the series boundary) via
`_leading_target_state_action`/`_leading_target_on_shipped_profile` (direct
`CanonicalEpisode` construction) or `mcap_fixture.leading_target_joint_topics`
(the real adapter/synchronizer/writer path) — never the achieved-DELTA
pairing plan 03-10 built (`_random_walk_state_action`,
`_coupled_state_action_on_shipped_profile`, `varying_motion_joint_topics`),
which this file no longer imports. Those plan-03-10 functions are
UNCHANGED and still used by `test_action_state_pairing.py` (Task 1, already
committed) and by their own docstrings' worked examples — they are simply
no longer the right coupling for THIS file's "clean pass" fixtures under
Option B's criterion.

Most mechanism-proof tests below declare `expected_action_state_lag=0` on
a `model_copy`'d profile rather than relying on the shipped `-4`: at small
frame counts (4-10), `MAX_LAG_SEARCH`'s `row_count - 3` cap cannot reach a
lag of `-4`/`-5` at all, so a lag-0 model is what keeps every frame count
in `03-10`'s own eight-count matrix reachable. One dedicated test
(`test_shipped_profile_declared_lag_passes_on_leading_target_fixture`)
proves the shipped `-4` value is genuinely consulted, at a frame count
large enough for the search to reach it.

**Pipeline-path coverage, named explicitly per test group (03-REVIEW.md
IN-01's original concern, carried forward):**

1. **Direct `CanonicalEpisode` construction** (`build_synthetic_episode`):
   exercises `check_action_state_lag`'s numerics directly against a
   hand-built episode, including the real 20/16 channel-width mismatch, but
   neither `GalaxeaInputAdapter.parse` nor `MasterClockSynchronizer.synchronize`.
2. **The real adapter/synchronizer/writer path**
   (`build_varying_baseline_dataset(..., joint_motion="leading_target")`):
   a genuinely varying (seeded random-walk) source MCAP, parsed by
   `GalaxeaInputAdapter.parse`, resampled by `MasterClockSynchronizer.synchronize`,
   and written by `LeRobotV21Writer.write_episode`. Proves the check's
   MECHANISM end-to-end through the real pipeline under Option B's
   criterion, not the check's behaviour on real recording DYNAMICS (that
   is `test_action_state_pairing.py`'s real-recording-skip-if-absent
   coverage).
3. **The no-declared-expectation abstention branch**
   (`test_no_declared_expected_lag_is_unevaluated`): a profile with
   `expected_action_state_lag=None` (the field's own default) abstains
   rather than silently passing — the Option B analogue of
   `data/adapters/checks.py`'s `no_limit_declared` precedent (Phase 2
   D-08), and this criterion's only abstention path (unlike the superseded
   correlation-based design, `tracking_error_lag` has no variance
   requirement, so there is no "degenerate signal" abstention branch left
   to test).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.contracts.containers import Disposition
from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.frame_fixtures import build_synthetic_episode
from data.tests.mutators import build_varying_baseline_dataset, copy_dataset, shift_action_column
from data.validation.context import ValidationContext
from data.validation.correlation import check_action_state_lag
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.validation.validate import _disposition_for, validate_dataset
from data.writers.lerobot_v21 import _CHUNK_INDEX, _DATA_PATH_TEMPLATE, LeRobotV21Writer

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"

MATCHED_KEYS = ["left_arm", "right_arm", "left_gripper", "right_gripper"]
"""The shipped profile's `action_primary`, verbatim — also valid
`state_key` values, so narrowing `state_vector_order` to this exact list
makes the emitted `observation.state` width equal the emitted `action`
width (16 == 16), with the four keys in identical order on both sides."""


@pytest.fixture()
def profile() -> EmbodimentProfile:
    """The shipped profile, verbatim — declares `expected_action_state_lag
    == -4`."""
    return load_profile(PROFILE_PATH)


@pytest.fixture()
def matched_profile(profile: EmbodimentProfile) -> EmbodimentProfile:
    """Equal-width (16==16) profile for the mechanism-proof tests, with
    `expected_action_state_lag` overridden to `0` — a lag-0 leading-target
    model keeps every one of 03-10's small frame counts (4-10) reachable
    within `MAX_LAG_SEARCH`'s `row_count - 3` cap, which the shipped
    profile's real `-4` cannot at those counts."""
    return profile.model_copy(
        update={"state_vector_order": list(MATCHED_KEYS), "expected_action_state_lag": 0}
    )


@pytest.fixture()
def profile_lag0(profile: EmbodimentProfile) -> EmbodimentProfile:
    """The shipped profile's real 20/16 width mismatch, with
    `expected_action_state_lag` overridden to `0` — same rationale as
    `matched_profile`, for the tests that specifically need the real
    width mismatch rather than the narrowed one."""
    return profile.model_copy(update={"expected_action_state_lag": 0})


@pytest.fixture()
def profile_no_declared_lag(profile: EmbodimentProfile) -> EmbodimentProfile:
    """The shipped profile with `expected_action_state_lag` reset to
    `None` — the abstention-path fixture."""
    return profile.model_copy(update={"expected_action_state_lag": None})


def _leading_target_state_action(
    profile: EmbodimentProfile, frame_count: int, *, lag: int = 0, seed: int = 42
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Builds a per-key random-walk `state_values`/`action_values` pair
    (`data/tests/frame_fixtures.py::build_synthetic_episode`'s override
    shape) where `action[t] == state[t - lag]` (clipped at the series
    boundary) — VALID-03 Option B's literal absolute-position-TARGET
    pairing, matching `tracking_error_lag`'s own shift convention exactly
    (`lag >= 0` compares `state[t]` against `action[t + lag]`; `lag < 0`
    the mirror): at the true `lag`, `action[t + lag] == state[t]` for
    every non-boundary `t`, giving `tracking_error_lag` an EXACT zero at
    that lag and a strictly positive value everywhere else (state is a
    random walk with white increments, so `|state[t+L]-state[t]|` grows in
    expectation with `|L - lag|`) — a mathematical certainty, not a
    coincidence of the chosen values. `profile.state_vector_order`'s keys
    must double as valid `action_key` values (true for `MATCHED_KEYS`) —
    this helper is only meant for a matched-width profile.
    """
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    keys_widths = [(key, len(by_state_key[key].joint_names)) for key in profile.state_vector_order]
    total_width = sum(w for _, w in keys_widths)

    rng = np.random.default_rng(seed)
    increments = rng.uniform(-1.0, 1.0, size=(frame_count, total_width))
    state_matrix = np.cumsum(increments, axis=0).astype(np.float32)

    action_matrix = np.zeros_like(state_matrix)
    for t in range(frame_count):
        src = min(max(t - lag, 0), frame_count - 1)
        action_matrix[t] = state_matrix[src]

    state_values: dict[str, list[np.ndarray]] = {}
    action_values: dict[str, list[np.ndarray]] = {}
    offset = 0
    for key, width in keys_widths:
        state_values[key] = [state_matrix[i, offset : offset + width] for i in range(frame_count)]
        action_values[key] = [action_matrix[i, offset : offset + width] for i in range(frame_count)]
        offset += width
    return state_values, action_values


def _leading_target_on_shipped_profile(
    profile: EmbodimentProfile, frame_count: int, *, lag: int = 0, seed: int = 42
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """The real-width (20/16) leading-target helper: builds a random-walk
    `state_values` for EVERY `state_vector_order` key (including `torso`,
    which is not in `action_primary`), and `action_values` for only the
    `action_primary` keys, each computed as `action[t] == state[t - lag]`
    (clipped) of that key's own state columns — VALID-03 Option B's
    literal pairing, at the real profile's own channel widths. `torso`'s
    own state channels vary too (an independent slice of the same random
    walk) so nothing about the fixture's non-degeneracy depends on which
    keys happen to be action-aligned.
    """
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    keys_widths = [(key, len(by_state_key[key].joint_names)) for key in profile.state_vector_order]
    total_width = sum(w for _, w in keys_widths)

    rng = np.random.default_rng(seed)
    increments = rng.uniform(-1.0, 1.0, size=(frame_count, total_width))
    state_matrix = np.cumsum(increments, axis=0).astype(np.float32)

    target_matrix = np.zeros_like(state_matrix)
    for t in range(frame_count):
        src = min(max(t - lag, 0), frame_count - 1)
        target_matrix[t] = state_matrix[src]

    state_values: dict[str, list[np.ndarray]] = {}
    offsets_by_key: dict[str, tuple[int, int]] = {}
    offset = 0
    for key, width in keys_widths:
        state_values[key] = [state_matrix[i, offset : offset + width] for i in range(frame_count)]
        offsets_by_key[key] = (offset, width)
        offset += width

    by_action_key = {f.action_key: f for f in profile.action_topics.values()}
    action_values: dict[str, list[np.ndarray]] = {}
    for action_key in profile.action_primary:
        state_start, state_width = offsets_by_key[action_key]
        assert state_width == len(by_action_key[action_key].joint_names)
        action_values[action_key] = [
            target_matrix[i, state_start : state_start + state_width] for i in range(frame_count)
        ]
    return state_values, action_values


def _write_single_episode_dataset(
    dest: Path, profile: EmbodimentProfile, episode: Any
) -> Path:
    """Writes one episode through the real `LeRobotV21Writer` — no
    provenance record, no encode manifest assertions: `check_action_state_lag`
    reads only the parquet file, so nothing else is needed."""
    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    writer.write_episode(episode)
    writer.finalize()
    return dest


def _write_minimal_parquet(
    dataset_dir: Path,
    episode_index: int,
    *,
    row_count: int,
    state_width: int,
    action_width: int,
) -> Path:
    """Writes a parquet file directly with pyarrow, bypassing
    `LeRobotV21Writer` entirely — the only way to produce a zero-width
    `observation.state`/`action` column (see module docstring)."""
    path = dataset_dir / _DATA_PATH_TEMPLATE.format(
        episode_chunk=_CHUNK_INDEX, episode_index=episode_index
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), state_width)),
            pa.field("action", pa.list_(pa.float32(), action_width)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "observation.state": pa.array(
                [[0.0] * state_width for _ in range(row_count)],
                type=schema.field("observation.state").type,
            ),
            "action": pa.array(
                [[0.0] * action_width for _ in range(row_count)],
                type=schema.field("action").type,
            ),
            "timestamp": pa.array([float(i) for i in range(row_count)], type=pa.float32()),
            "frame_index": pa.array(list(range(row_count)), type=pa.int64()),
            "episode_index": pa.array([episode_index] * row_count, type=pa.int64()),
            "index": pa.array(list(range(row_count)), type=pa.int64()),
            "task_index": pa.array([0] * row_count, type=pa.int64()),
        },
        schema=schema,
    )
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# Test 1: action[t] == state[t] exactly (lag 0) passes with measured_lag 0.
# ---------------------------------------------------------------------------


def test_clean_prefix_with_exact_leading_target_lag_zero_passes(
    tmp_path: Path, matched_profile: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_state_action(
        matched_profile, frame_count=10, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=10,
        fps=30.0,
        profile=matched_profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-test",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "clean", matched_profile, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=matched_profile)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.passed is True
    assert result.evidence["measured_lag"] == 0
    assert result.evidence["expected_lag"] == 0
    assert result.evidence["row_count"] == 10
    assert result.evidence["state_width"] == 16
    assert result.evidence["action_width"] == 16
    assert result.evidence["aligned_width"] == 16
    assert result.evidence["lag_unit"] == "timesteps"
    assert result.evidence["statistic"] == "tracking_error_argmin"
    assert result.evidence["error_at_measured_lag"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 2: shift_action_column by one row fails, recording the measured
# (nonzero) lag and the declared expected lag of 0.
# ---------------------------------------------------------------------------


def test_shift_action_column_fails_recording_nonzero_measured_lag(
    tmp_path: Path, matched_profile: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_state_action(
        matched_profile, frame_count=10, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=10,
        fps=30.0,
        profile=matched_profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-test",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "clean", matched_profile, episode)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    shift_action_column(mutated, episode_index=0, shift=1)

    ctx = ValidationContext(out_dir=mutated, profile=matched_profile)
    result = check_action_state_lag(ctx, 0)[0]

    assert result.passed is False
    assert result.evidence["measured_lag"] != 0
    assert result.evidence["expected_lag"] == 0


# ---------------------------------------------------------------------------
# Test 3: a one-row episode is explicitly unevaluated, not a numpy error.
# ---------------------------------------------------------------------------


def test_one_row_episode_is_unevaluated(
    tmp_path: Path, matched_profile: EmbodimentProfile
) -> None:
    episode = build_synthetic_episode(
        frame_count=1, fps=30.0, profile=matched_profile, task_prompt="one-row"
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "one_row", matched_profile, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=matched_profile)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.evidence["row_count"] == 1
    assert "row_count" in result.evidence[SKIP_REASON_KEY]
    assert result.passed is False


# ---------------------------------------------------------------------------
# Test 4: zero state AND action width is explicitly unevaluated, naming the
# zero width — written directly with pyarrow (see module docstring).
# ---------------------------------------------------------------------------


def test_zero_width_state_and_action_is_unevaluated(
    tmp_path: Path, matched_profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pyarrow` cannot round-trip a genuine zero-width `fixed_size_list`
    column through an actual parquet file in this pinned version — writing
    then reading one back raises `ArrowInvalid: Expected all lists to be of
    size=0 but index 1 had size=1` even for a single row, verified
    empirically and reproducible with no state/action involved at all. A
    real on-disk zero-width parquet file is therefore not constructible
    here. This test instead writes a real (nonzero-width) parquet file so
    the path exists, then monkeypatches `pq.read_table` — for this test
    only — to return an in-memory zero-width `pa.Table` (which pyarrow
    builds without error; only the parquet round-trip is broken), so the
    check's real code path still runs against genuinely zero-width arrays.
    """
    import data.validation.correlation as correlation_module

    dataset_dir = tmp_path / "zero_width"
    _write_minimal_parquet(
        dataset_dir, episode_index=0, row_count=3, state_width=1, action_width=1
    )

    zero_width_table = pa.table(
        {
            "observation.state": pa.array(
                [[] for _ in range(3)], type=pa.list_(pa.float32(), 0)
            ),
            "action": pa.array([[] for _ in range(3)], type=pa.list_(pa.float32(), 0)),
        }
    )
    monkeypatch.setattr(correlation_module.pq, "read_table", lambda path: zero_width_table)

    ctx = ValidationContext(out_dir=dataset_dir, profile=matched_profile)
    result = check_action_state_lag(ctx, 0)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert "0" in result.evidence[SKIP_REASON_KEY]
    assert result.passed is False


# ---------------------------------------------------------------------------
# Test 5: unequal (but both nonzero) state/action widths still produce a
# PASSING verdict via profile-derived width-matching — the real Galaxea
# profile's own shape (state 20, action 16), with genuinely coupled,
# non-degenerate data.
# ---------------------------------------------------------------------------


def test_unequal_nonzero_widths_pass_via_profile_derived_alignment(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_on_shipped_profile(
        profile_lag0, frame_count=6, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=6,
        fps=30.0,
        profile=profile_lag0,
        state_values=state_values,
        action_values=action_values,
        task_prompt="mismatched-widths",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "mismatched", profile_lag0, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile_lag0)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.evidence.get(UNEVALUATED_KEY) is not True
    assert result.evidence["state_width"] == 20
    assert result.evidence["action_width"] == 16
    assert result.evidence["aligned_width"] == 16
    assert result.evidence["state_width"] != result.evidence["action_width"]
    assert result.passed is True
    assert result.evidence["measured_lag"] == 0
    assert result.evidence["lag_unit"] == "timesteps"


# ---------------------------------------------------------------------------
# Test 6: two runs over the same prefix produce identical evidence.
# ---------------------------------------------------------------------------


def test_two_runs_over_same_prefix_produce_identical_evidence(
    tmp_path: Path, matched_profile: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_state_action(
        matched_profile, frame_count=8, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=8,
        fps=30.0,
        profile=matched_profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="repeat-run",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "repeat", matched_profile, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=matched_profile)

    first = check_action_state_lag(ctx, 0)[0]
    second = check_action_state_lag(ctx, 0)[0]

    assert first.evidence == second.evidence
    assert first.passed == second.passed


# ---------------------------------------------------------------------------
# Task 1 (03-06): real-width (20/16) clean vs. one-row-shifted at frame
# count 10 — proven end to end before Task 2 expands to the full
# frame-count matrix.
# ---------------------------------------------------------------------------


def test_clean_prefix_real_widths_passes_measured_lag_zero_at_frame_count_10(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_on_shipped_profile(
        profile_lag0, frame_count=10, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=10,
        fps=30.0,
        profile=profile_lag0,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-real-clean",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "clean_real", profile_lag0, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile_lag0)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.passed is True
    assert result.evidence["measured_lag"] == 0
    assert result.evidence["state_width"] == 20
    assert result.evidence["action_width"] == 16
    assert result.evidence["aligned_width"] == 16
    assert result.evidence["lag_unit"] == "timesteps"
    assert UNEVALUATED_KEY not in result.evidence


def test_shift_action_column_fails_abs_measured_lag_one_at_frame_count_10(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    state_values, action_values = _leading_target_on_shipped_profile(
        profile_lag0, frame_count=10, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=10,
        fps=30.0,
        profile=profile_lag0,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-real-shifted",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "clean_real2", profile_lag0, episode)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated_real")
    shift_action_column(mutated, episode_index=0, shift=1)

    ctx = ValidationContext(out_dir=mutated, profile=profile_lag0)
    result = check_action_state_lag(ctx, 0)[0]

    assert result.passed is False
    # The magnitude is 1 — NOT 16 (the action width) — proving the unit
    # actually stays timesteps, not flattened samples.
    assert abs(result.evidence["measured_lag"]) == 1


# ---------------------------------------------------------------------------
# Task 2 (03-06): the frame-count matrix that would have caught the shipped
# defect, plus the insufficient-rows branch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_count", [4, 6, 8, 10])
def test_clean_vs_shifted_discriminate_at_every_frame_count(
    tmp_path: Path, profile_lag0: EmbodimentProfile, frame_count: int
) -> None:
    state_values, action_values = _leading_target_on_shipped_profile(
        profile_lag0, frame_count=frame_count, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=frame_count,
        fps=30.0,
        profile=profile_lag0,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-matrix",
    )
    clean_dir = _write_single_episode_dataset(
        tmp_path / "clean", profile_lag0, episode
    )
    mutated_dir = copy_dataset(clean_dir, tmp_path / "mutated")
    shift_action_column(mutated_dir, episode_index=0, shift=1)

    clean_ctx = ValidationContext(out_dir=clean_dir, profile=profile_lag0)
    mutated_ctx = ValidationContext(out_dir=mutated_dir, profile=profile_lag0)

    clean_result = check_action_state_lag(clean_ctx, 0)[0]
    shifted_result = check_action_state_lag(mutated_ctx, 0)[0]

    assert clean_result.passed is True, clean_result.evidence
    assert clean_result.evidence["measured_lag"] == 0
    assert shifted_result.passed is False, shifted_result.evidence
    assert clean_result.evidence["measured_lag"] != shifted_result.evidence["measured_lag"]


def test_no_declared_expected_lag_is_unevaluated(
    tmp_path: Path, profile_no_declared_lag: EmbodimentProfile
) -> None:
    """VALID-03 Option B's only abstention path (module docstring point
    3): with no `expected_action_state_lag` declared,
    `consistency.action_state_lag` has nothing to compare a measured lag
    against, and abstains rather than silently passing — mirroring
    `data/adapters/checks.py`'s `no_limit_declared` precedent (Phase 2
    D-08). Runs the real pipeline (`build_varying_baseline_dataset`'s ramp
    default is a fine, non-degenerate-in-the-old-sense baseline here: this
    test exercises the DECLARATION gate, which fires before any lag
    computation, not the motion model)."""
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "no_declared", profile_no_declared_lag, camera_frames=6
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile_no_declared_lag)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert "expected_action_state_lag" in result.evidence[SKIP_REASON_KEY]
    assert result.passed is not True


def test_insufficient_rows_for_lag_search_is_unevaluated(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    """At frame_count 3, `min(MAX_LAG_SEARCH, row_count - 3)` is 0 — no
    non-zero lag is scoreable at all, so an `argmin == 0` pass would be
    vacuous. Proves the check refuses that rather than claiming a pass that
    proves nothing. Unaffected by the Option B pairing/criterion change:
    this branch is evaluated before any lag search runs."""
    state_values, action_values = _leading_target_on_shipped_profile(
        profile_lag0, frame_count=3, lag=0
    )
    episode = build_synthetic_episode(
        frame_count=3,
        fps=30.0,
        profile=profile_lag0,
        state_values=state_values,
        action_values=action_values,
        task_prompt="too-few-rows",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "too_few", profile_lag0, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile_lag0)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert "row_count" in result.evidence[SKIP_REASON_KEY]
    assert "3" in result.evidence[SKIP_REASON_KEY]
    assert result.passed is not True


# ---------------------------------------------------------------------------
# CR-02 (03-REVIEW.md): the search window is not sized to the declared lag's
# own magnitude — a short-but-genuinely-correct episode on the shipped
# profile must abstain, not falsely REJECT. Contrasted against a longer
# episode on the identical profile/pairing that still evaluates normally, so
# this test pins the boundary rather than just the abstention.
# ---------------------------------------------------------------------------


def test_shipped_profile_short_episode_abstains_instead_of_false_reject(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """The shipped profile declares `expected_action_state_lag=-4`. At
    `row_count=5`, `max_lag = min(MAX_LAG_SEARCH=5, row_count-3) = 2`, which
    cannot reach magnitude 4 — `lag == -4` is arithmetically impossible
    regardless of whether the data genuinely exhibits the declared lag.
    Before the CR-02 fix, this episode (built with the genuinely-correct
    leading-target pairing at lag=-4) fell through to `passed = lag ==
    expected_lag`, which is unconditionally `False`, driving a false
    REJECT on correct data. After the fix, the check abstains instead."""
    assert profile.expected_action_state_lag == -4

    short_frame_count = 5
    state_values, action_values = _leading_target_on_shipped_profile(
        profile, frame_count=short_frame_count, lag=-4
    )
    short_episode = build_synthetic_episode(
        frame_count=short_frame_count,
        fps=30.0,
        profile=profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-short-correct",
    )
    short_dir = _write_single_episode_dataset(
        tmp_path / "short_correct", profile, short_episode
    )
    short_ctx = ValidationContext(out_dir=short_dir, profile=profile)

    short_result = check_action_state_lag(short_ctx, 0)[0]

    assert short_result.evidence[UNEVALUATED_KEY] is True
    assert short_result.passed is False
    reason = short_result.evidence[SKIP_REASON_KEY]
    assert "-4" in reason  # the declared lag
    assert "max_lag=2" in reason  # the searched window
    assert str(short_frame_count) in reason  # the row count
    assert _disposition_for((short_result,)) is not Disposition.REJECT

    # Contrast: a longer episode on the identical profile/pairing still
    # evaluates normally — the fix must not turn this into a blanket
    # abstention for the shipped profile's declared lag.
    long_frame_count = 20
    state_values, action_values = _leading_target_on_shipped_profile(
        profile, frame_count=long_frame_count, lag=-4
    )
    long_episode = build_synthetic_episode(
        frame_count=long_frame_count,
        fps=30.0,
        profile=profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-long-correct",
    )
    long_dir = _write_single_episode_dataset(tmp_path / "long_correct", profile, long_episode)
    long_ctx = ValidationContext(out_dir=long_dir, profile=profile)

    long_result = check_action_state_lag(long_ctx, 0)[0]

    assert UNEVALUATED_KEY not in long_result.evidence
    assert long_result.passed is True
    assert long_result.evidence["measured_lag"] == -4
    assert _disposition_for((long_result,)) is Disposition.PASS


# ---------------------------------------------------------------------------
# Plan 03-11, Task 3: proves the SHIPPED profile's real declared
# `expected_action_state_lag == -4` is genuinely consulted by the check —
# at a frame count large enough for MAX_LAG_SEARCH to reach it.
# ---------------------------------------------------------------------------


def test_shipped_profile_declared_lag_passes_on_leading_target_fixture(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    assert profile.expected_action_state_lag == -4
    state_values, action_values = _leading_target_on_shipped_profile(
        profile, frame_count=20, lag=-4
    )
    episode = build_synthetic_episode(
        frame_count=20,
        fps=30.0,
        profile=profile,
        state_values=state_values,
        action_values=action_values,
        task_prompt="corr-shipped-lag",
    )
    dataset_dir = _write_single_episode_dataset(tmp_path / "shipped_lag", profile, episode)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_action_state_lag(ctx, 0)[0]

    assert result.passed is True, result.evidence
    assert result.evidence["measured_lag"] == -4
    assert result.evidence["expected_lag"] == -4
    assert UNEVALUATED_KEY not in result.evidence


# ---------------------------------------------------------------------------
# Plan 03-10, Task 1 (re-coupled for Option B, plan 03-11): one real-pipeline
# path — a genuinely varying-motion, absolute-position-target source MCAP
# through GalaxeaInputAdapter.parse -> MasterClockSynchronizer.synchronize
# -> LeRobotV21Writer.write_episode — proves the check discriminates clean
# from a one-row action shift with NO abstention marker on either side, and
# that the failure drives Disposition.REJECT and exit_code=1 through
# validate_dataset, attributable to VALID-03's check alone.
# ---------------------------------------------------------------------------


def test_real_pipeline_leading_target_clean_vs_shifted_discriminate(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    clean_dir = build_varying_baseline_dataset(
        tmp_path / "clean_leading", profile_lag0, camera_frames=12, joint_motion="leading_target"
    )
    mutated_dir = copy_dataset(clean_dir, tmp_path / "mutated_leading")
    shift_action_column(mutated_dir, episode_index=0, shift=1)

    clean_ctx = ValidationContext(out_dir=clean_dir, profile=profile_lag0)
    mutated_ctx = ValidationContext(out_dir=mutated_dir, profile=profile_lag0)

    clean_result = check_action_state_lag(clean_ctx, 0)[0]
    shifted_result = check_action_state_lag(mutated_ctx, 0)[0]

    # Assert the absence of the abstention marker on BOTH sides explicitly
    # — an UNEVALUATED verdict masquerading as either a pass or a fail is
    # the exact defect this task closes (D-GAP-01 Defect A / WINDOWS.md
    # id 5), invariant under Option B too.
    assert clean_result.passed is True, clean_result.evidence
    assert clean_result.evidence["measured_lag"] == 0
    assert clean_result.evidence["state_width"] == 20
    assert clean_result.evidence["action_width"] == 16
    assert clean_result.evidence["aligned_width"] == 16
    assert UNEVALUATED_KEY not in clean_result.evidence

    assert shifted_result.passed is False, shifted_result.evidence
    assert shifted_result.evidence["measured_lag"] == -1
    assert UNEVALUATED_KEY not in shifted_result.evidence


def test_real_pipeline_shift_drives_reject_and_nonzero_exit_code(
    tmp_path: Path, profile_lag0: EmbodimentProfile
) -> None:
    clean_dir = build_varying_baseline_dataset(
        tmp_path / "clean_disposition",
        profile_lag0,
        camera_frames=12,
        joint_motion="leading_target",
    )
    mutated_dir = copy_dataset(clean_dir, tmp_path / "mutated_disposition")
    shift_action_column(mutated_dir, episode_index=0, shift=1)

    clean_outcome = validate_dataset(clean_dir, profile_lag0)
    shifted_outcome = validate_dataset(mutated_dir, profile_lag0)

    clean_lag_result = next(
        r
        for r in clean_outcome.episode_results[0]
        if r.check_name == "consistency.action_state_lag"
    )
    shifted_lag_result = next(
        r
        for r in shifted_outcome.episode_results[0]
        if r.check_name == "consistency.action_state_lag"
    )

    # _disposition_for is imported deliberately, not just used incidentally
    # through validate_dataset's own overall exit code: it is the one
    # function that excludes an unevaluated result from a disposition
    # downgrade, so calling it on a one-tuple holding ONLY the lag result
    # proves the non-zero exit is caused by VALID-03's check alone, not
    # incidentally by some other failing check in the same run.
    assert clean_outcome.exit_code == 0
    assert _disposition_for((clean_lag_result,)) is Disposition.PASS

    assert shifted_outcome.exit_code == 1
    assert _disposition_for((shifted_lag_result,)) is Disposition.REJECT


# ---------------------------------------------------------------------------
# Plan 03-10, Task 2 (re-coupled for Option B, plan 03-11): expand to the
# eight camera-frame counts that opened the gap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera_frames", [4, 6, 8, 10, 12, 30, 60, 120])
def test_real_pipeline_clean_vs_shifted_discriminate_at_every_frame_count(
    tmp_path: Path, profile_lag0: EmbodimentProfile, camera_frames: int
) -> None:
    clean_dir = build_varying_baseline_dataset(
        tmp_path / "clean",
        profile_lag0,
        camera_frames=camera_frames,
        joint_motion="leading_target",
    )
    mutated_dir = copy_dataset(clean_dir, tmp_path / "mutated")
    shift_action_column(mutated_dir, episode_index=0, shift=1)

    clean_ctx = ValidationContext(out_dir=clean_dir, profile=profile_lag0)
    mutated_ctx = ValidationContext(out_dir=mutated_dir, profile=profile_lag0)

    clean_result = check_action_state_lag(clean_ctx, 0)[0]
    shifted_result = check_action_state_lag(mutated_ctx, 0)[0]

    assert clean_result.passed is True, clean_result.evidence
    assert clean_result.evidence["measured_lag"] == 0
    assert UNEVALUATED_KEY not in clean_result.evidence

    assert shifted_result.passed is False, shifted_result.evidence
    assert shifted_result.evidence["measured_lag"] == -1
    assert UNEVALUATED_KEY not in shifted_result.evidence

    assert clean_result.evidence["measured_lag"] != shifted_result.evidence["measured_lag"], (
        clean_result.evidence,
        shifted_result.evidence,
    )


def test_build_varying_baseline_dataset_ramp_default_does_not_falsely_pass_under_declared_lag(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """Pins a load-bearing negative property of the shipped profile's real
    `expected_action_state_lag == -4` against `build_varying_baseline_dataset`'s
    retained ramp default (no `joint_motion` argument): the ramp's
    action/state coupling (`action[t] - state[t]` a channel-wise CONSTANT
    offset, from `mcap_fixture.py`'s two independent default ramps) does
    not happen to land on the declared `-4` by accident, so the check does
    not falsely pass a dataset with no genuine absolute-position-target
    relationship to the declared physical lag. Supersedes plan 03-10's
    `test_build_varying_baseline_dataset_default_joint_motion_still_hits_degenerate_branch`,
    which pinned a `no variance` abstention branch `tracking_error_lag`
    (module docstring above) no longer has — this is the branch's Option B
    replacement, guarding against the exact false-positive class VALID-03
    exists to catch, not against a stale assertion."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "ramp_default", profile, camera_frames=12)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_action_state_lag(ctx, 0)[0]

    assert UNEVALUATED_KEY not in result.evidence
    assert result.evidence["measured_lag"] != -4
    assert result.passed is False
