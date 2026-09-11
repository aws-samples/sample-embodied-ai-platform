"""Tests for `data/validation/fingerprint.py`: REGR-01's fingerprint/compare
machinery and the gitignored `data/goldens/baseline.json` swap seam.

**D-18, recorded here as an explicit, accepted tradeoff:** REGR-01's
end-to-end comparison against a real, configured local golden
(`test_end_to_end_against_configured_local_golden`, Test 8) has NO
CI-executing coverage on a fresh clone — it is a single skip-if-absent
probe. The fingerprint/compare machinery itself (Tests 1-3, 7, 9) and the
baseline-resolution failure paths (Tests 4-6) are fully covered against
synthetic prefixes, all of which run unconditionally in CI. A
self-contained synthetic round-trip standing in for the real end-to-end
comparison was presented and deliberately not chosen (03-RESEARCH.md D-18)
— do not widen this module's coverage to close that gap, and do not weaken
the machinery's own tests to compensate; only the golden artifact itself is
deferred to a later phase.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import data.validation.fingerprint as fingerprint
from data.contracts.profile import EmbodimentProfile, load_profile
from data.contracts.profile import profile_hash as compute_profile_hash
from data.provenance.recorder import PIPELINE_VERSION
from data.tests.mutators import (
    build_varying_baseline_dataset,
    copy_dataset,
    shift_camera_frames,
)
from data.validation.context import BaselineConfig, ValidationContext
from data.validation.fingerprint import (
    check_golden_fingerprint,
    fingerprint_prefix,
    golden_applies_to,
    load_baseline_config,
)
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.validation.report import VALIDATION_REPORT_FILENAME
from data.validation.validate import validate_dataset

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _ctx(
    dataset_dir: Path,
    profile: EmbodimentProfile,
    *,
    baseline=None,
    use_local_golden: bool = False,
) -> ValidationContext:
    return ValidationContext(
        out_dir=dataset_dir,
        profile=profile,
        baseline=baseline,
        use_local_golden=use_local_golden,
    )


def _write_baseline_json(
    path: Path,
    *,
    golden_prefix: Path,
    source_sha256: tuple[str, ...],
    pipeline_version: str = "1.0.0",
    profile_hash: str = "deadbeef",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "golden_prefix": str(golden_prefix),
        "source_sha256": list(source_sha256),
        "pipeline_version": pipeline_version,
        "profile_hash": profile_hash,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


def _golden_recorded_source_sha256(golden_prefix: Path) -> tuple[str, ...]:
    """Re-hashes the golden's own provenance-recorded source recordings —
    the exact tuple a correctly-authored `baseline.json` would carry for
    this golden."""
    from data.provenance.recorder import _hash_file_sha256

    uris = fingerprint._golden_source_uris(golden_prefix)
    return tuple(_hash_file_sha256(uri) for uri in uris)


# ---------------------------------------------------------------------------
# Test 1: two fingerprint_prefix runs over an unmodified prefix produce
# byte-identical documents.
# ---------------------------------------------------------------------------


def test_two_runs_over_unmodified_prefix_are_byte_identical(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=6)

    doc_a = fingerprint_prefix(dataset_dir, profile)
    doc_b = fingerprint_prefix(dataset_dir, profile)

    assert json.dumps(doc_a, sort_keys=True) == json.dumps(doc_b, sort_keys=True)
    assert doc_a["episodes"], "fixture must produce at least one episode"


# ---------------------------------------------------------------------------
# Test 2: mutating one camera's frames changes exactly that camera's
# sampled-frame digests, naming the frame index and both hex digests.
# ---------------------------------------------------------------------------


def test_mutating_one_camera_changes_exactly_that_camera_digest(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=6)
    before = fingerprint_prefix(dataset_dir, profile)

    target_video_key = sorted(before["episodes"][0]["cameras"])[0]
    shift_camera_frames(dataset_dir, video_key=target_video_key, episode_index=0, shift=1)

    after = fingerprint_prefix(dataset_dir, profile)

    ep_before = before["episodes"][0]
    ep_after = after["episodes"][0]

    for video_key in sorted(ep_before["cameras"]):
        before_hashes = ep_before["cameras"][video_key]["sampled_frame_hashes"]
        after_hashes = ep_after["cameras"][video_key]["sampled_frame_hashes"]
        if video_key == target_video_key:
            assert before_hashes != after_hashes
            changed = [k for k in before_hashes if before_hashes[k] != after_hashes.get(k)]
            assert changed, "expected at least one sampled frame index to change"
        else:
            assert before_hashes == after_hashes

    # numeric columns (state/action/etc.) are untouched by a camera-only mutation
    assert ep_before["numeric_columns"] == ep_after["numeric_columns"]


# ---------------------------------------------------------------------------
# Test 3: perturbing one numeric column value changes exactly that column's
# digest.
# ---------------------------------------------------------------------------


def test_perturbing_one_numeric_column_changes_exactly_that_column(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    import pyarrow.parquet as pq

    from data.tests.mutators import _rewrite_fixed_size_list_column
    from data.writers.lerobot_v21 import _CHUNK_INDEX, _DATA_PATH_TEMPLATE

    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=6)
    before = fingerprint_prefix(dataset_dir, profile)

    parquet_path = dataset_dir / _DATA_PATH_TEMPLATE.format(episode_chunk=_CHUNK_INDEX, episode_index=0)
    table = pq.read_table(parquet_path)
    width = table.schema.field("action").type.list_size
    rows = table.column("action").to_pylist()
    rows[0][0] = float(rows[0][0]) + 1.0
    new_table = _rewrite_fixed_size_list_column(table, "action", rows, width)
    tmp_path_2 = parquet_path.parent / f".{parquet_path.name}.tmp"
    pq.write_table(new_table, tmp_path_2)
    import os

    os.replace(tmp_path_2, parquet_path)

    after = fingerprint_prefix(dataset_dir, profile)

    ep_before = before["episodes"][0]["numeric_columns"]
    ep_after = after["episodes"][0]["numeric_columns"]
    for column_name in ep_before:
        if column_name == "action":
            assert ep_before[column_name] != ep_after[column_name]
        else:
            assert ep_before[column_name] == ep_after[column_name]

    # camera fingerprints are untouched by a parquet-only mutation
    assert before["episodes"][0]["cameras"] == after["episodes"][0]["cameras"]


# ---------------------------------------------------------------------------
# Test 4: no data/goldens/baseline.json present, WITH the opt-in ->
# unevaluated, skip reason names the absent baseline file; the report's
# fully_validated is false.
# ---------------------------------------------------------------------------


def test_absent_baseline_file_is_unevaluated_naming_absent_file(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_path = tmp_path / "no_such_baseline.json"
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", missing_path)

    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    assert str(missing_path) in result.evidence[SKIP_REASON_KEY]
    assert result.evidence["golden_opt_in"] is True

    outcome = validate_dataset(dataset_dir, profile, use_local_golden=True)
    assert outcome.fully_validated is False


# ---------------------------------------------------------------------------
# Test 5: a baseline file present but naming a nonexistent prefix, WITH the
# opt-in -> the skip reason names the absent prefix, a distinct reason from
# Test 4.
# ---------------------------------------------------------------------------


def test_baseline_naming_nonexistent_prefix_is_unevaluated_distinct_reason(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.json"
    nonexistent_prefix = tmp_path / "no_such_golden_prefix"
    _write_baseline_json(baseline_path, golden_prefix=nonexistent_prefix, source_sha256=())
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    reason = result.evidence[SKIP_REASON_KEY]
    assert str(nonexistent_prefix) in reason
    assert "does not exist" in reason


# ---------------------------------------------------------------------------
# Test 6: a baseline file present whose recorded source hashes no longer
# match the files on disk, WITH the opt-in -> the skip reason names the hash
# mismatch, a third distinct reason.
# ---------------------------------------------------------------------------


def test_baseline_with_stale_source_hashes_is_unevaluated_distinct_reason(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden_dir = build_varying_baseline_dataset(tmp_path / "golden", profile, camera_frames=2)
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_json(
        baseline_path,
        golden_prefix=golden_dir,
        source_sha256=("0" * 64,),  # deliberately wrong — never matches a real hash
    )
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    reason = result.evidence[SKIP_REASON_KEY]
    assert "hash mismatch" in reason


# ---------------------------------------------------------------------------
# Test 6b (03-12, WR-02): a golden prefix that exists on disk but carries
# zero provenance-recorded source recordings (no meta/data_engine/ at all —
# "built without a ProvenanceRecorder", WR-02's own framing) is rejected
# with a FOURTH distinct reason, never treated as a trivially verified
# match via an empty-tuple-equals-empty-tuple comparison.
# ---------------------------------------------------------------------------


def test_baseline_with_no_provenance_recorded_sources_is_unevaluated_fourth_distinct_reason(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden_dir = build_varying_baseline_dataset(tmp_path / "golden", profile, camera_frames=2)
    # Strip the golden's own provenance records — "a golden built without a
    # ProvenanceRecorder", per WR-02's own description of the case.
    import shutil

    shutil.rmtree(golden_dir / "meta" / "data_engine")

    baseline_path = tmp_path / "baseline.json"
    _write_baseline_json(baseline_path, golden_prefix=golden_dir, source_sha256=())
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    # Check-level assertion: the golden never resolves as a verified match.
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    reason = result.evidence[SKIP_REASON_KEY]
    assert str(golden_dir) in reason
    # Distinct from the hash-mismatch reason (the pre-fix behaviour this
    # case used to fall into) and from the other two named reasons — a
    # hollow fix would merely trade one indistinguishable-from-a-pass
    # outcome for indistinguishable-from-a-different-failure.
    assert "hash mismatch" not in reason
    assert "does not exist" not in reason
    assert "no baseline config found" not in reason

    # Direct assertion against load_baseline_config itself: the pre-fix
    # behaviour (a populated BaselineConfig) is genuinely gone, not merely
    # rejected by some unrelated later branch that happens to also fire.
    baseline, direct_reason = load_baseline_config(baseline_path)
    assert baseline is None
    assert direct_reason is not None
    assert str(golden_dir) in direct_reason
    assert "hash mismatch" not in direct_reason


# ---------------------------------------------------------------------------
# Test 4b: opt-in OFF (the default) is inert — it never touches the
# filesystem, not even an exists() probe on DEFAULT_BASELINE_PATH, and
# reports a distinct "not requested" reason from every load_baseline_config
# failure reason.
# ---------------------------------------------------------------------------


class _ExplodingPath:
    """Stands in for `DEFAULT_BASELINE_PATH`: raises on any attribute
    access, so this test fails loudly the moment `check_golden_fingerprint`
    so much as calls `.exists()` on it while the opt-in is off."""

    def __getattr__(self, name: str) -> None:
        raise AssertionError(
            f"check_golden_fingerprint touched DEFAULT_BASELINE_PATH.{name} "
            "while use_local_golden was False — the opt-in-off path must not "
            "touch the filesystem at all"
        )

    def __str__(self) -> str:
        raise AssertionError(
            "check_golden_fingerprint stringified DEFAULT_BASELINE_PATH "
            "while use_local_golden was False"
        )


def test_opt_in_off_is_inert_and_never_touches_the_filesystem(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", _ExplodingPath())

    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    ctx = _ctx(dataset_dir, profile)  # use_local_golden defaults False
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.evidence["golden_opt_in"] is False
    reason = result.evidence[SKIP_REASON_KEY]
    assert "not requested" in reason
    assert "hash mismatch" not in reason
    assert "does not exist" not in reason
    assert "no baseline config found" not in reason


# ---------------------------------------------------------------------------
# Test 4c (the anti-poisoning proof — this whole gap's acceptance test): a
# real, hash-verifying golden sits on disk at DEFAULT_BASELINE_PATH while an
# UNRELATED synthetic fixture is validated through the FULL registry with no
# opt-in. Before this fix, this exact scenario made
# regression.golden_fingerprint report dozens of frame/column mismatches
# against unrelated content. Now the whole run passes clean because the
# check never resolves the ambient pointer.
# ---------------------------------------------------------------------------


def test_real_golden_on_disk_does_not_poison_an_unrelated_run(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    # plan 03-11 (VALID-03 Option B): the shipped profile now declares
    # expected_action_state_lag=-4. This test's ramp-default baseline
    # never claimed to model any real absolute-position-tracking
    # relationship — it is an unrelated-by-construction backdrop for THIS
    # test's own fingerprint/golden machinery — so build it against a
    # profile with no declared expectation, reverting VALID-03 to its
    # no_limit_declared-style abstention and keeping this test's own
    # "every check either passes or abstains" assertion about the checks
    # it actually targets.
    no_decl_profile = profile.model_copy(update={"expected_action_state_lag": None})
    golden_dir = build_varying_baseline_dataset(
        tmp_path / "golden", no_decl_profile, camera_frames=6
    )
    baseline_path = tmp_path / "goldens" / "baseline.json"
    _write_baseline_json(
        baseline_path,
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
    )
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)
    assert fingerprint.DEFAULT_BASELINE_PATH.exists(), "the golden pointer must be real on disk"

    unrelated_dir = build_varying_baseline_dataset(
        tmp_path / "unrelated", no_decl_profile, camera_frames=4
    )

    outcome = validate_dataset(unrelated_dir, no_decl_profile)  # no opt-in — the default call

    assert outcome.exit_code == 0, outcome.dataset_results

    for result in outcome.dataset_results:
        assert not (result.passed is False and not result.evidence.get(UNEVALUATED_KEY, False)), (
            result.check_name,
            result.evidence,
        )
    for results in outcome.episode_results.values():
        for result in results:
            assert not (
                result.passed is False and not result.evidence.get(UNEVALUATED_KEY, False)
            ), (result.check_name, result.evidence)

    fingerprint_results = [
        r for r in outcome.dataset_results if r.check_name == fingerprint.CHECK_NAME
    ]
    assert len(fingerprint_results) == 1
    fp_result = fingerprint_results[0]
    assert fp_result.evidence[UNEVALUATED_KEY] is True
    assert "not requested" in fp_result.evidence[SKIP_REASON_KEY]
    assert fp_result.evidence["golden_opt_in"] is False


# ---------------------------------------------------------------------------
# Test 4d (the opt-in's mirror proof): the SAME scenario WITH the opt-in
# true and the golden pointing at the unrelated fixture's own prefix passes
# — proving the opt-in actually re-enables a real comparison rather than
# disabling the check permanently.
# ---------------------------------------------------------------------------


def test_opt_in_true_with_self_built_golden_passes(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=4)
    golden_dir = copy_dataset(dataset_dir, tmp_path / "golden")

    baseline_path = tmp_path / "goldens" / "baseline.json"
    _write_baseline_json(
        baseline_path,
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        # Subject-scoping (Task 2) requires the golden's recorded
        # profile_hash/pipeline_version to match what the real dataset's own
        # provenance records carry — the fixed "1.0.0"/"deadbeef" defaults
        # would make golden_applies_to correctly report non-applicability.
        pipeline_version=PIPELINE_VERSION,
        profile_hash=compute_profile_hash(profile),
    )
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    outcome = validate_dataset(dataset_dir, profile, use_local_golden=True)

    fingerprint_results = [
        r for r in outcome.dataset_results if r.check_name == fingerprint.CHECK_NAME
    ]
    assert len(fingerprint_results) == 1
    fp_result = fingerprint_results[0]
    assert fp_result.passed is True, fp_result.evidence
    assert fp_result.evidence["mismatches"] == []
    assert fp_result.evidence["golden_opt_in"] is True


# ---------------------------------------------------------------------------
# Test 7: comparing a prefix against a golden built from itself passes with
# every hash matching.
# ---------------------------------------------------------------------------


def test_comparing_prefix_against_golden_built_from_itself_passes(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=6)
    golden_dir = copy_dataset(dataset_dir, tmp_path / "golden")

    recorded_hashes = _golden_recorded_source_sha256(golden_dir)
    baseline = BaselineConfig(
        golden_prefix=golden_dir,
        source_sha256=recorded_hashes,
        pipeline_version="1.0.0",
        profile_hash="unused-here",
    )

    ctx = _ctx(dataset_dir, profile, baseline=baseline)
    (result,) = check_golden_fingerprint(ctx)

    assert result.passed is True, result.evidence
    assert result.evidence["mismatches"] == []


# ---------------------------------------------------------------------------
# Test 8 (D-18): the REGR-01 end-to-end test against the configured local
# golden is skipped, with a visible reason, when that golden is absent.
# ---------------------------------------------------------------------------


def test_end_to_end_against_configured_local_golden(profile: EmbodimentProfile) -> None:
    baseline_path = fingerprint.DEFAULT_BASELINE_PATH
    if not baseline_path.exists():
        pytest.skip(
            f"no local golden configured at {baseline_path} — REGR-01 has no "
            "CI-executing end-to-end coverage on a fresh clone until the later "
            "golden/public-dataset phase lands (D-18, accepted tradeoff)"
        )

    baseline, reason = load_baseline_config(baseline_path)
    assert baseline is not None, reason

    ctx = ValidationContext(out_dir=baseline.golden_prefix, profile=profile, baseline=baseline)
    (result,) = check_golden_fingerprint(ctx)
    assert result.passed is True, result.evidence


# ---------------------------------------------------------------------------
# Test 9: data/goldens/baseline.example.json parses as JSON and carries
# exactly the four keys BaselineConfig declares.
# ---------------------------------------------------------------------------


def test_baseline_example_json_carries_exactly_baseline_config_fields() -> None:
    example_path = Path("data/goldens/baseline.example.json")
    assert example_path.exists()
    with open(example_path, encoding="utf-8") as f:
        payload = json.load(f)

    expected_fields = {f.name for f in BaselineConfig.__dataclass_fields__.values()}
    assert set(payload.keys()) == expected_fields


# ---------------------------------------------------------------------------
# Task 2 (subject-scoping, the `adjacency` edge probe): an opted-in golden
# only applies to a dataset it is genuinely about.
# ---------------------------------------------------------------------------


def _write_applicable_baseline_json(
    path: Path, *, golden_prefix: Path, profile: EmbodimentProfile
) -> None:
    """Writes a `baseline.json` whose recorded `profile_hash`/
    `pipeline_version` match what `build_varying_baseline_dataset` (real
    `ProvenanceRecorder.record` calls) actually records for `profile` — the
    subject-scoping tests' shared "genuinely applicable" fixture."""
    _write_baseline_json(
        path,
        golden_prefix=golden_prefix,
        source_sha256=_golden_recorded_source_sha256(golden_prefix),
        pipeline_version=PIPELINE_VERSION,
        profile_hash=compute_profile_hash(profile),
    )


def test_matching_subject_applies_and_compares(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=4)
    golden_dir = copy_dataset(dataset_dir, tmp_path / "golden")

    baseline = BaselineConfig(
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        pipeline_version=PIPELINE_VERSION,
        profile_hash=compute_profile_hash(profile),
    )

    applies, reason, subject_match = golden_applies_to(_ctx(dataset_dir, profile), baseline)
    assert applies is True, reason
    assert reason is None
    assert subject_match["dataset_profile_hash"] == subject_match["golden_profile_hash"]
    assert subject_match["dataset_pipeline_version"] == subject_match["golden_pipeline_version"]

    (result,) = check_golden_fingerprint(
        _ctx(dataset_dir, profile, baseline=baseline)
    )
    assert result.passed is True, result.evidence
    assert "mismatches" in result.evidence
    assert result.evidence["mismatches"] == []


def test_mismatched_profile_hash_separates_naming_profile_hash(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    golden_dir = copy_dataset(dataset_dir, tmp_path / "golden")

    baseline_path = tmp_path / "goldens" / "baseline.json"
    _write_baseline_json(
        baseline_path,
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        pipeline_version=PIPELINE_VERSION,
        profile_hash="deliberately-wrong-profile-hash",  # differs from the real one
    )
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.evidence["golden_opt_in"] is True
    reason = result.evidence[SKIP_REASON_KEY]
    assert "profile_hash" in reason
    assert "pipeline_version" not in reason
    assert "mismatches" not in result.evidence  # no fingerprinting ran
    subject_match = result.evidence["subject_match"]
    assert subject_match["golden_profile_hash"] == "deliberately-wrong-profile-hash"
    assert subject_match["dataset_profile_hash"] != subject_match["golden_profile_hash"]


def test_mismatched_pipeline_version_separates_naming_pipeline_version(
    tmp_path: Path, profile: EmbodimentProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)
    golden_dir = copy_dataset(dataset_dir, tmp_path / "golden")

    baseline_path = tmp_path / "goldens" / "baseline.json"
    _write_baseline_json(
        baseline_path,
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        pipeline_version="9.9.9-deliberately-wrong",  # differs from the real "1.0.0"
        profile_hash=compute_profile_hash(profile),
    )
    monkeypatch.setattr(fingerprint, "DEFAULT_BASELINE_PATH", baseline_path)

    ctx = _ctx(dataset_dir, profile, use_local_golden=True)
    (result,) = check_golden_fingerprint(ctx)

    assert result.evidence[UNEVALUATED_KEY] is True
    reason = result.evidence[SKIP_REASON_KEY]
    assert "pipeline_version" in reason
    assert "mismatches" not in result.evidence  # no fingerprinting ran
    subject_match = result.evidence["subject_match"]
    assert subject_match["golden_pipeline_version"] == "9.9.9-deliberately-wrong"
    assert subject_match["dataset_pipeline_version"] != subject_match["golden_pipeline_version"]


def test_absent_provenance_separates_naming_the_absence(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    # A dataset directory with no meta/data_engine/ at all — a synthetic
    # fixture written without a ProvenanceRecorder. golden_applies_to must
    # treat this as NOT matching, never as a wildcard match.
    dataset_dir = tmp_path / "no_provenance_dataset"
    dataset_dir.mkdir()

    golden_dir = build_varying_baseline_dataset(tmp_path / "golden", profile, camera_frames=2)
    baseline = BaselineConfig(
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        pipeline_version=PIPELINE_VERSION,
        profile_hash=compute_profile_hash(profile),
    )

    applies, reason, subject_match = golden_applies_to(_ctx(dataset_dir, profile), baseline)
    assert applies is False
    assert reason is not None
    assert "no provenance" in reason or "cannot be established" in reason
    assert subject_match["dataset_profile_hash"] is None
    assert subject_match["dataset_pipeline_version"] is None


def test_zero_episode_dataset_against_applicable_golden_never_passes(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    golden_dir = build_varying_baseline_dataset(tmp_path / "golden", profile, camera_frames=2)

    # A zero-episode dataset that nonetheless carries a real provenance
    # record matching the golden's subject (so subject-scoping applies),
    # via build_varying_baseline_dataset(episodes=0) — zero episodes means
    # zero provenance records too, which is itself the absent-provenance
    # case; assert on the outcome directly rather than assuming which named
    # path produced it.
    empty_dataset_dir = build_varying_baseline_dataset(
        tmp_path / "empty", profile, camera_frames=2, episodes=0
    )
    baseline = BaselineConfig(
        golden_prefix=golden_dir,
        source_sha256=_golden_recorded_source_sha256(golden_dir),
        pipeline_version=PIPELINE_VERSION,
        profile_hash=compute_profile_hash(profile),
    )

    ctx = _ctx(empty_dataset_dir, profile, baseline=baseline)
    (result,) = check_golden_fingerprint(ctx)

    assert result.passed is not True, result.evidence
    is_unevaluated = result.evidence.get(UNEVALUATED_KEY, False)
    has_episode_set_mismatch = any(
        m.get("kind") == "episode_set_mismatch" for m in result.evidence.get("mismatches", [])
    )
    assert is_unevaluated or has_episode_set_mismatch, result.evidence


# ---------------------------------------------------------------------------
# Task 3: the --use-local-golden CLI flag reaches the check. No baseline.json
# is written into the tracked tree here on purpose -- that would leave a
# golden artifact behind at commit time, which this plan's own prohibition
# forbids. Instead this proves the flag is wired by observing
# golden_opt_in/skip_reason in the WRITTEN validation_report.json for a
# fixture that has no golden pointer at any path the flag would resolve
# (the real repo-relative data/goldens/baseline.json is absent on a fresh
# clone per D-18, and this test does not create one). Do not "strengthen"
# this into writing data/goldens/baseline.json -- that is exactly the
# tracked-artifact prohibition this plan enforces.
# ---------------------------------------------------------------------------


def _dataset_result(report: dict, check_name: str) -> dict:
    matches = [r for r in report["dataset_results"] if r["check_name"] == check_name]
    assert len(matches) == 1, (check_name, report["dataset_results"])
    return matches[0]


def test_cli_flag_off_records_golden_opt_in_false(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)

    proc = subprocess.run(
        [sys.executable, "-m", "data.validate", str(dataset_dir), "--profile", str(PROFILE_PATH)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    report_path = dataset_dir / "meta" / "data_engine" / VALIDATION_REPORT_FILENAME
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    fp_result = _dataset_result(report, fingerprint.CHECK_NAME)
    assert fp_result["evidence"]["golden_opt_in"] is False
    assert fp_result["evidence"][UNEVALUATED_KEY] is True
    assert "not requested" in fp_result["evidence"][SKIP_REASON_KEY]


def test_cli_flag_on_records_golden_opt_in_true_and_still_exits_zero(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(tmp_path / "dataset", profile, camera_frames=2)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "data.validate",
            str(dataset_dir),
            "--profile",
            str(PROFILE_PATH),
            "--use-local-golden",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    report_path = dataset_dir / "meta" / "data_engine" / VALIDATION_REPORT_FILENAME
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    fp_result = _dataset_result(report, fingerprint.CHECK_NAME)
    assert fp_result["evidence"]["golden_opt_in"] is True
    assert fp_result["evidence"][UNEVALUATED_KEY] is True
    # The real repo-relative data/goldens/baseline.json pointer is absent on
    # this checkout (D-18); the reason must come from load_baseline_config's
    # absent-file branch, a distinct string from the not-requested reason.
    assert "no baseline config found" in fp_result["evidence"][SKIP_REASON_KEY]


def test_cli_help_lists_the_opt_in_flag() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "data.validate", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--use-local-golden" in proc.stdout
    assert "ignored" in proc.stdout.lower()


def test_no_env_var_read_anywhere_on_the_opt_ins_path() -> None:
    import re

    hits = [
        (path, i + 1, line.strip())
        for path in (
            "data/validate.py",
            "data/validation/fingerprint.py",
            "data/validation/context.py",
        )
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines())
        if not line.strip().startswith("#") and re.search(r"os\.environ|getenv", line)
    ]
    assert hits == []
