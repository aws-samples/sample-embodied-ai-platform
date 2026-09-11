"""VALID-01 end-to-end tracer tests: the writer's own measured
`packets_muxed` counter, `data/validation/frame_integrity.py`'s four-way
comparison, the `ProvenanceRecorder.amend_checks` amend path, and the
`python -m data.validate` shim's exit code — proven by mutating a copy of a
source-MCAP-backed synthetic dataset (Task 1, plan 03-01).

Plan 03-03 (Phase 3) adds VALID-02's relative spot-decode content check
(`check_spot_decode_content`, over a mapping recomputed by
`data.sync.master_clock.nearest_camera_matches`) and D-10's post-write
decoded-pixel-uniqueness check (`check_decoded_pixel_uniqueness`), both
proven against the source-MCAP-backed, per-frame-distinct
`build_varying_baseline_dataset` baseline.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import (
    build_baseline_dataset,
    build_varying_baseline_dataset,
    copy_dataset,
    freeze_camera_frames,
    shift_camera_frames,
    truncate_camera_mp4,
)
from data.validation.context import ValidationContext
from data.validation.frame_integrity import (
    check_decoded_pixel_uniqueness,
    check_frame_count_agreement,
    check_spot_decode_content,
    resolve_source_recording,
)
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.validation.validate import validate_dataset

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


def _profile_path_without_declared_lag(tmp_path: Path, profile: EmbodimentProfile) -> Path:
    """Plan 03-11 (VALID-03 Option B) declared `expected_action_state_lag
    == -4` on the SHIPPED `galaxea_r1_pro.yaml`. This test's baseline
    fixture (`build_varying_baseline_dataset`'s ramp default) never claimed
    to model any real absolute-position-tracking relationship — it is an
    unrelated-by-construction backdrop for THIS test's own VALID-01/D-10
    checks — so it now genuinely (and correctly) fails
    `consistency.action_state_lag` under the declared lag, which is not
    what this "clean prefix, all checks pass" smoke test is about. Writes a
    temp copy of the shipped profile with `expected_action_state_lag` unset
    (reverting to the check's `no_limit_declared`-style abstention) so this
    test's own subprocess CLI invocation stays a clean-vs-mutated
    comparison of the ONE check it actually targets, not incidentally of
    VALID-03."""
    dumped = profile.model_dump(mode="json")
    dumped.pop("expected_action_state_lag", None)
    path = tmp_path / "profile_no_declared_lag.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dumped, f)
    return path
MASTER_VIDEO_KEY = "observation.images.head"
WRIST_LEFT_VIDEO_KEY = "observation.images.wrist_left"
WRIST_RIGHT_VIDEO_KEY = "observation.images.wrist_right"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


# ---------------------------------------------------------------------------
# Test 1: clean prefix passes for every emitted camera, all four counters
# equal and recorded.
# ---------------------------------------------------------------------------


def test_clean_prefix_passes_frame_count_agreement_for_every_camera(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=6)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_frame_count_agreement(ctx, 0)

    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.passed is True
        assert result.evidence["frames_submitted"] == 6
        assert result.evidence["packets_muxed"] == 6
        assert result.evidence["decoded_frame_count"] == 6
        assert result.evidence["parquet_row_count"] == 6


# ---------------------------------------------------------------------------
# Test 2: 3 dropped frames from one camera fails, naming decoded_frame_count
# vs parquet_row_count, dropped_frames == 3.
# ---------------------------------------------------------------------------


def test_truncated_camera_fails_naming_disagreeing_counters(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    # Uses the MASTER camera (role: master_clock), not a secondary camera —
    # the master camera IS the timeline, so a truncated master-camera MP4
    # takes the generic "disagreeing counters" branch rather than the
    # secondary-camera unmatched-grid-ticks branch.
    build_baseline_dataset(tmp_path / "clean", profile, frame_count=6)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    truncate_camera_mp4(mutated, video_key=MASTER_VIDEO_KEY, drop_last_n=3)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_frame_count_agreement(ctx, 0)}

    failing = results[MASTER_VIDEO_KEY]
    assert failing.passed is False
    assert failing.evidence["decoded_frame_count"] == 3
    assert failing.evidence["parquet_row_count"] == 6
    assert failing.evidence["dropped_frames"] == 3
    assert "decoded_frame_count!=parquet_row_count" in failing.evidence["disagreeing_counters"]

    # Untouched cameras still pass.
    assert results[WRIST_LEFT_VIDEO_KEY].passed is True


# ---------------------------------------------------------------------------
# Test 3: after a full validate_dataset run, the mutated episode's provenance
# record amends to REJECT with ingest-time evidence intact.
# ---------------------------------------------------------------------------


def test_amend_preserves_ingest_time_checks_and_sets_reject(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_baseline_dataset(tmp_path / "clean", profile, frame_count=6)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    truncate_camera_mp4(mutated, video_key=WRIST_LEFT_VIDEO_KEY, drop_last_n=3)

    record_path = mutated / "meta" / "data_engine" / "episode_000000.json"
    before = json.loads(record_path.read_text())
    ingest_time_checks = before["checks"]
    assert len(ingest_time_checks) == 2

    outcome = validate_dataset(mutated, profile)
    assert outcome.exit_code != 0

    after = json.loads(record_path.read_text())
    assert after["disposition"] == "REJECT"
    # Original ingest-time entries survive at their original positions.
    assert after["checks"][:2] == ingest_time_checks
    # New post-write results are appended after them.
    assert len(after["checks"]) > 2
    assert any(c["check_name"] == "videos.frame_count_agreement" for c in after["checks"][2:])


# ---------------------------------------------------------------------------
# Test 4: the CLI shim's exit code, exercised via subprocess.
# ---------------------------------------------------------------------------


def test_cli_shim_exit_code_clean_vs_mutated(tmp_path: Path, profile: EmbodimentProfile) -> None:
    # Source-backed (plan 03-03, Rule 1): the registry now also runs
    # VALID-02's spot-decode content check and D-10's decoded-pixel-
    # uniqueness check, both of which compare real pixel content. The
    # structural-only `build_baseline_dataset` (synthetic, unrelated video
    # content) would fail those two new checks even on an otherwise "clean"
    # prefix, so this exit-code smoke test needs the source-backed,
    # per-frame-distinct baseline to stay green end to end.
    #
    # camera_frames=6/drop_last_n=3 (the natural values, restored 03-06):
    # plan 03-04 temporarily recalibrated this fixture to camera_frames=4/
    # drop_last_n=1 to dodge `consistency.action_state_lag`'s old flattened
    # cross-correlation, which false-positived on this fixture's own
    # unrelated-by-construction action/state ramps (see 03-04-SUMMARY.md).
    # 03-06 redesigned that check to width-match via the profile and search
    # the lag in timesteps, with an explicit degenerate-signal branch for a
    # constant-per-channel delta — exactly this fixture's shape
    # (`mcap_fixture.py`'s `joint_state_messages`/`motor_control_messages`
    # defaults are a per-channel linear ramp, so the true state delta is
    # constant). `consistency.action_state_lag` now reports this fixture as
    # UNEVALUATED rather than passed/failed at any frame count, and an
    # unevaluated fatal result drives neither a REJECT disposition nor a
    # non-zero exit code (`data/validation/validate.py::_disposition_for`),
    # so the natural fixture values work again.
    # plan 03-11 (VALID-03 Option B): build against a profile with no
    # declared expected_action_state_lag — see
    # _profile_path_without_declared_lag's docstring.
    no_decl_profile_path = _profile_path_without_declared_lag(tmp_path, profile)
    no_decl_profile = load_profile(no_decl_profile_path)
    build_varying_baseline_dataset(tmp_path / "clean", no_decl_profile, camera_frames=6)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    truncate_camera_mp4(mutated, video_key=WRIST_LEFT_VIDEO_KEY, drop_last_n=3)

    clean_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "data.validate",
            str(tmp_path / "clean"),
            "--profile",
            str(no_decl_profile_path),
        ],
        capture_output=True,
        text=True,
    )
    assert clean_proc.returncode == 0, clean_proc.stderr

    mutated_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "data.validate",
            str(mutated),
            "--profile",
            str(no_decl_profile_path),
        ],
        capture_output=True,
        text=True,
    )
    assert mutated_proc.returncode != 0, mutated_proc.stdout


# ---------------------------------------------------------------------------
# Test 5: a deleted camera MP4 fails naming that camera's video_key and the
# absent path — not a silent skip.
# ---------------------------------------------------------------------------


def test_deleted_camera_mp4_fails_naming_video_key_and_absent_path(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    build_baseline_dataset(tmp_path / "clean", profile, frame_count=4)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    ctx = ValidationContext(out_dir=mutated, profile=profile)
    video_path = ctx.video_path(0, WRIST_LEFT_VIDEO_KEY)
    video_path.unlink()

    results = {r.evidence["video_key"]: r for r in check_frame_count_agreement(ctx, 0)}
    failing = results[WRIST_LEFT_VIDEO_KEY]
    assert failing.passed is False
    assert str(video_path) in failing.evidence["error"]


# ---------------------------------------------------------------------------
# Test 6: a zero-frame episode reports unevaluated, parquet_row_count 0,
# never a silent pass.
# ---------------------------------------------------------------------------


def test_zero_frame_episode_is_unevaluated_not_a_bare_pass(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "zero", profile, frame_count=0)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_frame_count_agreement(ctx, 0)

    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.evidence[UNEVALUATED_KEY] is True
        assert result.evidence["parquet_row_count"] == 0
        assert result.passed is False


# ---------------------------------------------------------------------------
# Test 7: the baseline's provenance record is source-backed — source_uri
# exists, re-hashes to source_sha256, and ingest-time checks are non-empty.
# ---------------------------------------------------------------------------


def test_baseline_provenance_record_is_source_backed(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=3)
    record_path = dataset_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text())

    source_path = Path(record["source_uri"])
    assert source_path.exists()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record["source_sha256"]
    assert len(record["checks"]) > 0


# ---------------------------------------------------------------------------
# Plan 03-03, Task 1: VALID-02's relative spot-decode content check.
# ---------------------------------------------------------------------------


def test_spot_decode_passes_on_source_backed_baseline(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """The load-bearing positive test (VALID-02's non-vacuity criterion):
    on the unmutated, source-MCAP-backed baseline, every emitted camera's
    `videos.spot_decode_content` verdict is `passed=True` with
    `unevaluated` absent."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "varying", profile, camera_frames=12)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_spot_decode_content(ctx, 0)

    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.evidence.get(UNEVALUATED_KEY) is None, result.evidence
        assert result.passed is True, result.evidence
        assert len(result.evidence["per_index"]) > 0


def test_spot_decode_fails_on_rotated_copy(tmp_path: Path, profile: EmbodimentProfile) -> None:
    """A master-camera MP4 rotated by one frame — same frame count, shifted
    content — fails, naming at least one spot-check index whose mapped
    source frame was out-ranked by a neighbour."""
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=12)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    shift_camera_frames(mutated, video_key=MASTER_VIDEO_KEY, shift=1)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_spot_decode_content(ctx, 0)}
    failing = results[MASTER_VIDEO_KEY]

    assert failing.passed is False
    assert failing.evidence.get(UNEVALUATED_KEY) is None
    assert any(entry["winning_offset"] != 0 for entry in failing.evidence["per_index"])


def test_spot_decode_fails_on_rotated_secondary_camera(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """The mapping for a secondary camera goes through the same
    `nearest_camera_matches` call — secondary cameras are where that logic
    is least exercised, so this is proven separately from the master-camera
    case."""
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=12)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    shift_camera_frames(mutated, video_key=WRIST_LEFT_VIDEO_KEY, shift=1)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_spot_decode_content(ctx, 0)}
    failing = results[WRIST_LEFT_VIDEO_KEY]

    assert failing.passed is False
    assert failing.evidence.get(UNEVALUATED_KEY) is None
    # The untouched master camera and the other untouched secondary camera
    # still pass — the mutation is isolated to the one camera mutated.
    assert results[MASTER_VIDEO_KEY].passed is True
    assert results[WRIST_RIGHT_VIDEO_KEY].passed is True


def test_spot_decode_fails_naming_both_numbers_on_secondary_length_mismatch(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """The recomputed matched-tick count for a secondary camera must equal
    its decoded frame count; a truncated secondary-camera MP4 disagrees
    with the source-recomputed mapping (which is untouched, since it is
    recomputed from the unmodified source recording) and fails naming both
    numbers."""
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=12)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    truncate_camera_mp4(mutated, video_key=WRIST_LEFT_VIDEO_KEY, drop_last_n=1)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_spot_decode_content(ctx, 0)}
    failing = results[WRIST_LEFT_VIDEO_KEY]

    assert failing.passed is False
    assert "matched_tick_count" in failing.evidence
    assert "decoded_frame_count" in failing.evidence
    assert failing.evidence["matched_tick_count"] != failing.evidence["decoded_frame_count"]


def test_spot_decode_unevaluated_when_source_recording_deleted(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """Deleting the source recording named by the episode's provenance
    record produces an explicitly-unevaluated result per camera, naming the
    absent path — and the run's report marks the dataset not fully
    validated, never a pass."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6)
    record_path = dataset_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text())
    source_path = Path(record["source_uri"])
    source_path.unlink()

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    resolved_path, reason = resolve_source_recording(ctx, 0)
    assert resolved_path is None
    assert "absent" in (reason or "")

    results = check_spot_decode_content(ctx, 0)
    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.evidence[UNEVALUATED_KEY] is True
        assert "absent" in result.evidence[SKIP_REASON_KEY]

    outcome = validate_dataset(dataset_dir, profile)
    assert outcome.fully_validated is False


def test_spot_decode_unevaluated_when_source_recording_hash_mismatches(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """Leaving the source recording in place but changing its bytes yields
    an unevaluated result whose skip reason names a hash mismatch —
    distinct from the absent-path reason."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=6)
    record_path = dataset_dir / "meta" / "data_engine" / "episode_000000.json"
    record = json.loads(record_path.read_text())
    source_path = Path(record["source_uri"])
    with open(source_path, "ab") as f:
        f.write(b"\x00")

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    resolved_path, reason = resolve_source_recording(ctx, 0)
    assert resolved_path is None
    assert "sha256 mismatch" in (reason or "")

    results = check_spot_decode_content(ctx, 0)
    for result in results:
        assert result.evidence[UNEVALUATED_KEY] is True
        assert "sha256 mismatch" in result.evidence[SKIP_REASON_KEY]
        assert "absent" not in result.evidence[SKIP_REASON_KEY]


def test_spot_decode_tie_with_neighbour_fails_recording_both_errors(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact tie between the target's error and a neighbour's error
    fails the check — a tie carries no evidence the target matched better
    — and both values are recorded in evidence."""
    dataset_dir = build_varying_baseline_dataset(tmp_path / "varying", profile, camera_frames=12)
    monkeypatch.setattr("data.validation.frame_integrity.mean_absolute_error", lambda a, b: 0.5)

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_spot_decode_content(ctx, 0)}
    failing = results[MASTER_VIDEO_KEY]

    assert failing.passed is False
    first_entry = failing.evidence["per_index"][0]
    assert first_entry["target_error"] == 0.5
    assert first_entry["neighbour_errors"]
    assert all(v == 0.5 for v in first_entry["neighbour_errors"].values())


def test_spot_decode_one_frame_episode_checks_index_zero_once(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """A one-frame episode's deduplicated fractional index set collapses to
    exactly index 0, checked exactly once. CR-01 (03-REVIEW.md): a
    single-decoded-frame camera has zero valid neighbour offsets for that
    index (`NEIGHBOUR_WINDOW`'s offsets all fall outside `[0, 1)`), so
    nothing is ever compared against — the check must report
    `UNEVALUATED_KEY`, never a bare `passed=True` resting on `all([])`."""
    pinned_profile = profile.model_copy(
        update={"master_clock": profile.master_clock.model_copy(update={"target_fps": 25.0})}
    )
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "one_frame", pinned_profile, camera_frames=1
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=pinned_profile)

    results = check_spot_decode_content(ctx, 0)
    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert len(result.evidence["per_index"]) == 1
        assert result.evidence["per_index"][0]["index"] == 0
        assert result.evidence[UNEVALUATED_KEY] is True, result.evidence
        assert result.passed is False


def test_spot_decode_one_frame_camera_with_wrong_content_never_silently_passes(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR-01 regression: a one-frame camera has no neighbour to compare
    against, so `check_spot_decode_content` must never report
    `passed=True` for it — regardless of whether the frame's actual
    content is right or wrong. Proven here by forcing the "expected"
    (source-decoded) pixels to be an arbitrary, deliberately-wrong image
    via a monkeypatched decoder backend, and confirming the check still
    reports `UNEVALUATED_KEY` rather than a silent pass driven by
    `all([])`."""
    pinned_profile = profile.model_copy(
        update={"master_clock": profile.master_clock.model_copy(update={"target_fps": 25.0})}
    )
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "one_frame_wrong", pinned_profile, camera_frames=1
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=pinned_profile)

    class _WrongBackend:
        def decode(self, data: bytes) -> np.ndarray:
            # Deliberately wrong content: a fixed all-white image,
            # guaranteed to disagree with any real decoded frame.
            return np.full((48, 64, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(
        "data.validation.frame_integrity.get_decoder_backend", lambda: _WrongBackend()
    )

    results = check_spot_decode_content(ctx, 0)
    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.passed is not True, result.evidence
        assert result.evidence.get(UNEVALUATED_KEY) is True, result.evidence


def test_spot_decode_zero_frame_episode_is_unevaluated(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """A zero-frame episode (no parquet ever written) emits an
    explicitly-unevaluated result rather than a pass."""
    dataset_dir = build_baseline_dataset(tmp_path / "zero", profile, frame_count=0)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_spot_decode_content(ctx, 0)
    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.evidence[UNEVALUATED_KEY] is True
        assert result.passed is False


# ---------------------------------------------------------------------------
# Plan 03-03, Task 2: D-10's post-write decoded-pixel-uniqueness check.
# ---------------------------------------------------------------------------


def test_decoded_pixel_uniqueness_passes_on_clean_distinct_frames(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "varying", profile, camera_frames=8)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_decoded_pixel_uniqueness(ctx, 0)

    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.passed is True, result.evidence
        assert result.evidence["distinct_digest_count"] == result.evidence["decoded_frame_count"]


def test_decoded_pixel_uniqueness_fails_on_frozen_camera_reencoded_but_pixel_identical(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """`freeze_camera_frames` re-encodes the target camera with every frame
    equal to frame 0 — byte-different (re-encoded) but pixel-identical, the
    exact case the ingest-time compressed-payload digest check cannot see
    and this post-write check exists to catch.

    Two camera frames (not more): a real H.264 encoder's rate-control
    lookahead assigns slightly different quantization to frames near a
    lossy re-encode's start/end boundary even when their input pixels are
    byte-identical, so a longer frozen run can decode to a handful of
    near-identical-but-not-digest-identical groups rather than exactly one
    — a genuine encoder property this project accepts rather than papering
    over with a perceptual-hash tolerance (which was rejected for the exact
    reason it is too insensitive to a real difference). Two frames is
    enough to prove the check's own comparison logic without depending on
    that boundary effect resolving in the test's favor."""
    build_varying_baseline_dataset(tmp_path / "clean", profile, camera_frames=2)
    mutated = copy_dataset(tmp_path / "clean", tmp_path / "mutated")
    freeze_camera_frames(mutated, video_key=WRIST_LEFT_VIDEO_KEY)

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    results = {r.evidence["video_key"]: r for r in check_decoded_pixel_uniqueness(ctx, 0)}
    failing = results[WRIST_LEFT_VIDEO_KEY]

    assert failing.passed is False
    assert failing.evidence["distinct_digest_count"] == 1
    assert failing.evidence["decoded_frame_count"] > 1
    # Untouched cameras still pass.
    assert results[MASTER_VIDEO_KEY].passed is True


def test_decoded_pixel_uniqueness_single_frame_camera_is_unevaluated(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    """A single decoded frame cannot be non-unique — reported as
    unevaluated, never a silent pass."""
    dataset_dir = build_baseline_dataset(tmp_path / "single", profile, frame_count=1)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    results = check_decoded_pixel_uniqueness(ctx, 0)
    assert len(results) == len(ctx.emitted_video_keys)
    for result in results:
        assert result.evidence[UNEVALUATED_KEY] is True
        assert result.passed is False
