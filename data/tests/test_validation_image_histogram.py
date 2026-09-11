"""WINDOWS.md id 2 tests: the exact-invariant verification oracle for the
persisted per-camera `Uint8Histogram` sidecars.

Task 1's tracer proves the mass invariant end to end on one camera via
`build_baseline_dataset` (the structural baseline, per its own docstring —
pixel content is irrelevant to a mass-only comparison) plus the new
`perturb_histogram_counts` mutator. Task 2 (below the Task-1 marker) expands
to the full invariant set, cross-episode merge additivity (via
`build_varying_baseline_dataset`, which writes multiple real episodes
through one writer), the two interrupted-writer states, the
nothing-to-check unevaluated branch, the read-only proof, and the
`meta/stats.json` boundary this plan must not cross.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.contracts.profile import EmbodimentProfile, load_profile
from data.tests.mutators import (
    build_baseline_dataset,
    build_varying_baseline_dataset,
    copy_dataset,
    perturb_histogram_counts,
)
from data.validation.context import ValidationContext
from data.validation.image_histogram import (
    CHECK_NAME,
    HISTOGRAM_BIN_COUNT,
    check_image_histogram_integrity,
    expected_channel_mass,
    histogram_invariant_findings,
    load_persisted_histograms,
)
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY, ValidationRegistry
from data.validation.stats_oracle import check_stats_recomputation
from data.validation.validate import validate_dataset

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"
MASTER_VIDEO_KEY = "observation.images.head"


@pytest.fixture()
def profile() -> EmbodimentProfile:
    return load_profile(PROFILE_PATH)


def _stats_json_keys(dataset_dir: Path) -> list[str]:
    return sorted(json.loads((dataset_dir / "meta" / "stats.json").read_text()).keys())


def _hash_meta_tree(dataset_dir: Path) -> dict[str, str]:
    meta_dir = dataset_dir / "meta"
    return {
        str(p.relative_to(meta_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(meta_dir.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Task 1: registry placement.
# ---------------------------------------------------------------------------


def test_registered_as_13th_check_fatal_dataset_scoped() -> None:
    registry = ValidationRegistry.default()
    names = [c.name for c in registry.checks]
    assert len(names) == 13
    assert names[-1] == CHECK_NAME
    assert names[-2] == "regression.golden_fingerprint"
    entry = next(c for c in registry.checks if c.name == CHECK_NAME)
    assert entry.severity == "fatal"
    assert entry.scope == "dataset"


# ---------------------------------------------------------------------------
# Task 1 tracer: a clean baseline passes; a one-count perturbation fails,
# naming the mutated camera and episode.
# ---------------------------------------------------------------------------


def test_clean_baseline_passes(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=5)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is True
    assert result.evidence["findings"] == []


def test_one_count_perturbation_fails_naming_camera_and_episode(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=5)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    perturb_histogram_counts(
        mutated, video_key=MASTER_VIDEO_KEY, episode_index=0, bin_index=10, delta=1
    )

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is False
    finding = next(
        f
        for f in result.evidence["findings"]
        if f["video_key"] == MASTER_VIDEO_KEY and f["episode_index"] == 0
    )
    assert finding["invariant"] in ("mass_identity", "channel_agreement")


def test_no_mp4_decode_no_size_literal_no_adapters_checks_import() -> None:
    import inspect

    import data.validation.image_histogram as m

    source = inspect.getsource(m)
    assert "data.adapters.checks" not in source
    assert m.HISTOGRAM_BIN_COUNT == 256


def test_expected_channel_mass_reads_dimensions_from_info_json(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=5)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    height = info["features"][MASTER_VIDEO_KEY]["video_info"]["video.height"]
    width = info["features"][MASTER_VIDEO_KEY]["video_info"]["video.width"]

    mass, reason = expected_channel_mass(ctx, 0, MASTER_VIDEO_KEY)

    assert reason is None
    assert mass == 5 * height * width


def test_persisted_counts_loaded_as_integer_dtype(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=5)
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    persisted = load_persisted_histograms(ctx)
    histograms, included = persisted[0]

    assert included is True
    for counts in histograms.values():
        assert np.issubdtype(counts.dtype, np.integer)


def test_stats_json_key_set_unchanged_by_task1(tmp_path: Path, profile: EmbodimentProfile) -> None:
    before = build_baseline_dataset(tmp_path / "before", profile, frame_count=3)
    keys_before = _stats_json_keys(before)
    after = build_baseline_dataset(tmp_path / "after", profile, frame_count=3)
    ctx = ValidationContext(out_dir=after, profile=profile)
    check_image_histogram_integrity(ctx)
    keys_after = _stats_json_keys(after)
    assert keys_before == keys_after


# ---------------------------------------------------------------------------
# Task 2: `histogram_invariant_findings` — the full per-histogram invariant
# set, and a histogram that violates two invariants reporting both.
# ---------------------------------------------------------------------------


def test_wrong_bin_count_fails_naming_bin_count() -> None:
    wrong = np.zeros((3, 128), dtype=np.int64)
    findings = histogram_invariant_findings(wrong, expected_mass=100)
    assert len(findings) == 1
    assert findings[0]["invariant"] == "bin_count"


def test_negative_count_fails_naming_non_negative() -> None:
    counts = np.zeros((3, HISTOGRAM_BIN_COUNT), dtype=np.int64)
    counts[0, 0] = -1
    counts[0, 1] = 101
    counts[1, 0] = 100
    counts[2, 0] = 100
    findings = histogram_invariant_findings(counts, expected_mass=100)
    invariants = {f["invariant"] for f in findings}
    assert "non_negative" in invariants


def test_channel_disagreement_fails_naming_channel_agreement() -> None:
    counts = np.zeros((3, HISTOGRAM_BIN_COUNT), dtype=np.int64)
    counts[0, 0] = 100
    counts[1, 0] = 99
    counts[2, 0] = 100
    findings = histogram_invariant_findings(counts, expected_mass=100)
    invariants = {f["invariant"] for f in findings}
    assert "channel_agreement" in invariants


def test_two_violated_invariants_both_reported() -> None:
    counts = np.zeros((3, HISTOGRAM_BIN_COUNT), dtype=np.int64)
    counts[0, 0] = -1
    counts[0, 1] = 100
    counts[1, 0] = 50  # disagrees with channel 0's mass (99) and channel 2's (100)
    counts[2, 0] = 100
    findings = histogram_invariant_findings(counts, expected_mass=100)
    invariants = {f["invariant"] for f in findings}
    assert "non_negative" in invariants
    assert "channel_agreement" in invariants
    assert len(findings) >= 2


def test_correct_mass_and_structure_yields_no_findings() -> None:
    counts = np.zeros((3, HISTOGRAM_BIN_COUNT), dtype=np.int64)
    counts[:, 0] = 100
    findings = histogram_invariant_findings(counts, expected_mass=100)
    assert findings == []


# ---------------------------------------------------------------------------
# Task 2: cross-episode merge additivity on a clean multi-episode dataset.
# ---------------------------------------------------------------------------


def test_merge_additivity_passes_on_clean_multi_episode_dataset(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "multi", profile, camera_frames=6, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is True
    merge_evidence = result.evidence["merge_evidence"]
    for video_key in ctx.emitted_video_keys:
        entry = merge_evidence[video_key]
        assert entry["admitted_episode_indices"] == [0, 1]
        assert entry["expected_merged_mass"] == sum(entry["per_episode_masses"])
        assert all(m == entry["expected_merged_mass"] for m in entry["observed_merged_masses"])


def test_merge_additivity_fails_when_a_persisted_histogram_is_perturbed(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "multi", profile, camera_frames=6, episodes=2
    )
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    perturb_histogram_counts(
        mutated, video_key=MASTER_VIDEO_KEY, episode_index=1, bin_index=5, delta=3
    )

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is False
    invariants = {f["invariant"] for f in result.evidence["findings"]}
    assert "mass_identity" in invariants or "merge_additivity" in invariants


# ---------------------------------------------------------------------------
# Task 2: the two interrupted-writer states — a named failure, never a
# crash and never a silent skip.
# ---------------------------------------------------------------------------


def test_missing_encode_manifest_is_named_failure_not_a_crash(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=4)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    manifest_path = mutated / "meta" / "encode_manifests" / "episode_000000.json"
    manifest_path.unlink()

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is False
    failure = next(
        f for f in result.evidence["findings"] if f["invariant"] == "manifest_missing"
    )
    assert failure["episode_index"] == 0
    assert failure["video_key"] in ctx.emitted_video_keys


def test_missing_persisted_histogram_is_named_failure_not_a_crash(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=4)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    partial_path = mutated / "meta" / "stats_partials" / "episode_000000.json"
    payload = json.loads(partial_path.read_text())
    del payload["histograms"][MASTER_VIDEO_KEY]
    partial_path.write_text(json.dumps(payload))

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is False
    failure = next(
        f for f in result.evidence["findings"] if f["invariant"] == "histogram_missing"
    )
    assert failure["episode_index"] == 0
    assert failure["video_key"] == MASTER_VIDEO_KEY


# ---------------------------------------------------------------------------
# 03-12 (WR-01): a malformed manifest for one episode costs one named
# finding, and every finding already accumulated for other episodes
# survives — the failure this fix closes.
# ---------------------------------------------------------------------------


def test_malformed_manifest_mid_run_preserves_prior_findings_not_a_lost_run(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "multi", profile, camera_frames=6, episodes=3
    )
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")

    # Episode 0: a real mass-identity violation, computed BEFORE the
    # malformed episode 1 is ever reached (sorted(persisted) visits 0, 1, 2).
    perturb_histogram_counts(
        mutated, video_key=MASTER_VIDEO_KEY, episode_index=0, bin_index=3, delta=1
    )
    # Episode 1: overwrite its encode manifest with bytes that are not
    # valid JSON — a one-line write, deliberately not a shared mutator
    # (single call site).
    manifest_path = mutated / "meta" / "encode_manifests" / "episode_000001.json"
    manifest_path.write_text("{not valid json")

    ctx = ValidationContext(out_dir=mutated, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.passed is False
    assert "error" not in result.evidence
    findings = result.evidence["findings"]

    mass_finding = next(
        f
        for f in findings
        if f["episode_index"] == 0
        and f["video_key"] == MASTER_VIDEO_KEY
        and f["invariant"] in ("mass_identity", "channel_agreement")
    )
    assert mass_finding is not None

    unreadable_findings = [
        f for f in findings if f["invariant"] == "manifest_unreadable" and f["episode_index"] == 1
    ]
    assert unreadable_findings, findings
    assert all(f["video_key"] in ctx.emitted_video_keys for f in unreadable_findings)


def test_expected_channel_mass_malformed_manifest_returns_named_reason_never_raises(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_baseline_dataset(tmp_path / "clean", profile, frame_count=4)
    mutated = copy_dataset(dataset_dir, tmp_path / "mutated")
    manifest_path = mutated / "meta" / "encode_manifests" / "episode_000000.json"
    manifest_path.write_text("{not valid json")

    ctx = ValidationContext(out_dir=mutated, profile=profile)

    mass, reason = expected_channel_mass(ctx, 0, MASTER_VIDEO_KEY)

    assert mass is None
    assert reason is not None
    assert str(manifest_path) in reason


# ---------------------------------------------------------------------------
# Task 2: the nothing-to-check state yields UNEVALUATED, never a vacuous
# pass.
# ---------------------------------------------------------------------------


def test_zero_episode_dataset_is_unevaluated_not_a_vacuous_pass(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    from data.writers.lerobot_v21 import LeRobotV21Writer

    dest = tmp_path / "empty"
    dest.mkdir()
    writer = LeRobotV21Writer(dest, profile, stdin_isatty=lambda: False)
    writer.finalize()

    ctx = ValidationContext(out_dir=dest, profile=profile)
    result = check_image_histogram_integrity(ctx)[0]

    assert result.evidence[UNEVALUATED_KEY] is True
    assert result.passed is not True
    assert SKIP_REASON_KEY in result.evidence


# ---------------------------------------------------------------------------
# Task 2: read-only proof.
# ---------------------------------------------------------------------------


def test_check_is_read_only(tmp_path: Path, profile: EmbodimentProfile) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=5, episodes=2
    )
    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)

    digest_before = _hash_meta_tree(dataset_dir)
    result1 = check_image_histogram_integrity(ctx)[0]
    digest_after_first = _hash_meta_tree(dataset_dir)
    result2 = check_image_histogram_integrity(ctx)[0]
    digest_after_second = _hash_meta_tree(dataset_dir)

    assert digest_before == digest_after_first == digest_after_second
    assert result1.evidence == result2.evidence
    assert result1.passed == result2.passed


# ---------------------------------------------------------------------------
# Task 2: the meta/stats.json boundary this plan must not cross.
# ---------------------------------------------------------------------------


def test_full_registry_run_leaves_stats_json_key_set_unchanged_and_video_free(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=5, episodes=1
    )
    keys_before = _stats_json_keys(dataset_dir)

    validate_dataset(dataset_dir, profile)

    keys_after = _stats_json_keys(dataset_dir)
    assert keys_before == keys_after
    for video_key in ("observation.images.head", "observation.images.wrist_left"):
        assert video_key not in keys_after


def test_check_stats_recomputation_still_rejects_injected_video_entry(
    tmp_path: Path, profile: EmbodimentProfile
) -> None:
    dataset_dir = build_varying_baseline_dataset(
        tmp_path / "clean", profile, camera_frames=5, episodes=1
    )
    stats_path = dataset_dir / "meta" / "stats.json"
    payload = json.loads(stats_path.read_text())
    payload[MASTER_VIDEO_KEY] = {
        "mean": [0.5],
        "std": [0.1],
        "min": [0.0],
        "max": [1.0],
        "q01": [0.0],
        "q99": [1.0],
        "count": [1],
    }
    stats_path.write_text(json.dumps(payload, sort_keys=True, indent=2))

    ctx = ValidationContext(out_dir=dataset_dir, profile=profile)
    result = check_stats_recomputation(ctx)[0]

    assert result.passed is False
    assert MASTER_VIDEO_KEY in result.evidence["extra_in_stats_json"]
