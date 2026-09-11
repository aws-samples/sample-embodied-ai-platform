"""WINDOWS.md id 2: an exact-invariant verification oracle for the persisted
per-camera `Uint8Histogram` sidecars `data/writers/lerobot_v21.py` computes
and `data/writers/stats.py::write_partial_file` persists into
`meta/stats_partials/episode_*.json`, but which nothing in Phase 3 checked
before this plan.

**Chosen oracle: an exact invariant check anchored to the encode manifest,
not a lossy MP4-decode comparison (D-13's own rationale, restated here).**
The writer histograms the **source RGB frames it submits to the encoder**;
the persisted MP4 is a lossy H.264 encode. Decoded pixels therefore differ
from the histogrammed pixels by an amount no fixed tolerance can bound
usefully, and a tolerance wide enough to absorb H.264 loss is wide enough to
hide a wrong histogram — the same reasoning D-13 already applied to numeric
stats. So every invariant below is integer-exact with no tolerance knob, and
every input is already-persisted metadata: no MP4 video file is ever opened
by this module and no frame-size literal appears in it.

**The five invariants, four per-histogram and one cross-episode.**
Per-histogram (`histogram_invariant_findings`): the bin count is exactly
`HISTOGRAM_BIN_COUNT` (256, because `uint8` has exactly 256 possible values);
every count is a non-negative integer; every channel's total mass agrees
with every other channel's (every pixel contributes exactly once to every
channel); and the mass equals `frames_submitted * video.height * video.width`
— `frames_submitted` from that camera's encode manifest entry (D-14) and the
frame dimensions from `meta/info.json`'s `video_info` (never a decoded MP4,
never a hardcoded size). Cross-episode: the dataset-wide merged histogram's
mass equals the sum of the per-episode masses over exactly the episodes each
partial's own `included_in_merge` flag admits, folded with the writer's own
`Uint8Histogram.merge` (integer addition) — imported, never reimplemented.

**Scope of the invariant, stated plainly (the residual gap this plan
records, not silently absorbs).** These invariants prove a histogram is
*structurally and dimensionally* correct and consistent with the encode
manifest. They do not prove the *distribution* is right — a histogram with
the correct total mass but values shuffled between bins of the same channel
satisfies every invariant here. Closing that would require either the lossy
MP4-decode comparison this module deliberately does not perform, or a second
write-path artifact; see `.planning/WINDOWS.md` for the recorded residual
entry.

**Read-only, and every artifact-read isolated per episode.** This module
opens `meta/stats_partials/*.json`, `meta/encode_manifests/*.json` and
`meta/info.json` for reading only. `check_image_histogram_integrity` scopes
its exception handling to the unit that can actually fail: a malformed
encode manifest for one episode becomes one `manifest_unreadable` finding
naming that episode (distinct from the absent-file `manifest_missing`
finding), and any other exception raised while processing one episode
becomes one `episode_check_error` finding for that episode alone — the
per-episode loop always proceeds to the next episode rather than aborting,
so every finding already computed for other episodes survives, matching
`stats_oracle.py::check_stats_recomputation`'s per-feature isolation
pattern in this same phase. A single last-resort outer `try`/`except`
remains as defense in depth, but its result now carries whatever findings
had already accumulated alongside the `error` key rather than discarding
them — a corrupt sidecar is one failed named check, never an aborted run,
and never a lost one.

**No `meta/stats.json` change.** No key is added to `meta/stats.json` by
this module — `stats_oracle.py`'s own video-entry-is-a-failure guard is the
boundary this check must not cross, and it does not touch that file at all.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.contracts.containers import CheckResult
from data.validation.context import ValidationContext
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY
from data.writers.stats import Uint8Histogram

CHECK_NAME = "statistics.image_histogram_integrity"

HISTOGRAM_BIN_COUNT = 256
"""Exact, not approximate: `uint8` has exactly 256 possible values — the
writer's own stated reason `Uint8Histogram` is exact rather than
estimated (`data/writers/stats.py`)."""


def load_persisted_histograms(
    ctx: ValidationContext,
) -> dict[int, tuple[dict[str, np.ndarray], bool]]:
    """Reads every `meta/stats_partials/episode_*.json`, returning
    `{episode_index: (histograms_by_video_key, included_in_merge)}`. Reuses
    `constant_channels.py::load_merged_partials`'s directory-and-glob idiom
    rather than inventing a second traversal style. Counts are converted to
    `np.int64` explicitly and never touch a float dtype anywhere on this
    path — the persisted values are nested JSON lists, and a float cast
    anywhere here would silently defeat the integrality invariant this
    module exists to assert."""
    result: dict[int, tuple[dict[str, np.ndarray], bool]] = {}
    partials_dir = ctx.out_dir / "meta" / "stats_partials"
    if not partials_dir.is_dir():
        return result

    for path in sorted(partials_dir.glob("episode_*.json")):
        try:
            episode_index = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        histograms = {
            video_key: np.asarray(counts, dtype=np.int64)
            for video_key, counts in payload.get("histograms", {}).items()
        }
        included_in_merge = bool(payload.get("included_in_merge", True))
        result[episode_index] = (histograms, included_in_merge)
    return result


def expected_channel_mass(
    ctx: ValidationContext, episode_index: int, video_key: str
) -> tuple[int | None, str | None]:
    """Returns `(expected_mass, reason)`: `expected_mass` is
    `frames_submitted * video.height * video.width`, where `frames_submitted`
    comes from that episode's `video_key` entry in
    `meta/encode_manifests/episode_{NNNNNN}.json` (D-14) and the frame
    dimensions from `meta/info.json`'s `features[video_key]["video_info"]`
    (`video.height`/`video.width`) — never a decoded MP4 video file and
    never a hardcoded frame size. Returns `(None, reason)` — never raises — when the
    manifest file, the camera's manifest entry, `meta/info.json`, or that
    feature's `video_info` is missing, AND ALSO when the manifest file or
    `meta/info.json` exists but is malformed (a `json.JSONDecodeError` or an
    `OSError` reading it) — both are caught and folded into a named reason
    naming the unreadable path, so this function's "never raises" claim is
    true as written rather than aspirational. The caller turns every one of
    these cases into a named failure, never a silent skip.
    """
    manifest_path = ctx.encode_manifest_path(episode_index)
    if not manifest_path.exists():
        return None, f"no encode manifest at {manifest_path}"
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"encode manifest at {manifest_path} is unreadable: {type(e).__name__}: {e}"
    cam_entry = manifest.get(video_key)
    if cam_entry is None:
        return None, f"encode manifest at {manifest_path} has no entry for {video_key!r}"
    if "frames_submitted" not in cam_entry:
        return None, f"encode manifest entry for {video_key!r} at {manifest_path} has no frames_submitted"
    frames_submitted = int(cam_entry["frames_submitted"])

    info_path = ctx.out_dir / "meta" / "info.json"
    if not info_path.exists():
        return None, f"no meta/info.json at {info_path}"
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"meta/info.json at {info_path} is unreadable: {type(e).__name__}: {e}"
    feature = info.get("features", {}).get(video_key)
    if feature is None:
        return None, f"meta/info.json has no features entry for {video_key!r}"
    video_info = feature.get("video_info", {})
    height = video_info.get("video.height")
    width = video_info.get("video.width")
    if height is None or width is None:
        return (
            None,
            f"meta/info.json features.{video_key}.video_info is missing video.height/video.width",
        )

    return frames_submitted * int(height) * int(width), None


def histogram_invariant_findings(
    counts: np.ndarray, expected_mass: int | None
) -> list[dict[str, Any]]:
    """One named finding per violated invariant on a SINGLE persisted
    histogram — a histogram violating two invariants reports both, because
    which ones fired together is itself diagnostic. Invariants, in order:
    bin count exactly `HISTOGRAM_BIN_COUNT` per channel; an integer dtype;
    every count non-negative; every channel's total mass identical to every
    other channel's; and (only when `expected_mass` is not `None` — the
    caller routes a `None` expectation to its own named failure instead of
    calling this with an unusable expectation) the mass equal to
    `expected_mass`. Every comparison is exact — no tolerance anywhere,
    because every value on both sides is an integer count.
    """
    findings: list[dict[str, Any]] = []

    if counts.ndim != 2 or counts.shape[1] != HISTOGRAM_BIN_COUNT:
        findings.append(
            {
                "invariant": "bin_count",
                "expected_bins_per_channel": HISTOGRAM_BIN_COUNT,
                "observed_shape": list(counts.shape),
            }
        )
        # A histogram with the wrong shape has no well-defined per-channel
        # mass to check further invariants against — report the structural
        # failure and stop, rather than guessing at a channel axis.
        return findings

    if not np.issubdtype(counts.dtype, np.integer):
        findings.append({"invariant": "integer_dtype", "observed_dtype": str(counts.dtype)})

    if counts.size and int(counts.min()) < 0:
        findings.append({"invariant": "non_negative", "observed_min": int(counts.min())})

    channel_masses = [int(m) for m in counts.sum(axis=1)]
    if channel_masses and any(m != channel_masses[0] for m in channel_masses):
        findings.append(
            {"invariant": "channel_agreement", "observed_channel_masses": channel_masses}
        )

    if expected_mass is not None and channel_masses:
        if any(m != expected_mass for m in channel_masses):
            findings.append(
                {
                    "invariant": "mass_identity",
                    "expected_mass": expected_mass,
                    "observed_channel_masses": channel_masses,
                }
            )

    return findings


def check_image_histogram_integrity(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset.

    For each episode index and each `video_key` in `ctx.emitted_video_keys`:
    reads the persisted histogram (`load_persisted_histograms`) and the
    expectation built from the encode manifest and `meta/info.json`
    (`expected_channel_mass`), then asserts every per-histogram invariant
    (`histogram_invariant_findings`). A partial carrying a histogram with no
    corresponding manifest entry, or a manifest entry with no corresponding
    persisted histogram, is a named failure identifying the episode and the
    camera — the state an interrupted writer leaves behind, never a crash
    and never a silent skip.

    Cross-episode: sums the per-episode per-channel masses over exactly the
    episodes whose partial carries `included_in_merge` true, folds the same
    episodes' histograms with `Uint8Histogram.merge` (integer addition,
    imported from `data/writers/stats.py`, never reimplemented), and asserts
    the merged result's per-channel mass equals that sum exactly.

    A dataset with no emitted cameras, or whose partials carry no
    histograms at all, yields `UNEVALUATED_KEY` true with a named reason —
    never a vacuous pass.
    """
    try:
        persisted = load_persisted_histograms(ctx)
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (CheckResult(CHECK_NAME, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),)

    emitted_video_keys = ctx.emitted_video_keys

    any_histogram = any(histograms for histograms, _ in persisted.values())
    if not emitted_video_keys or not any_histogram:
        reason = (
            "no emitted cameras declared on the profile"
            if not emitted_video_keys
            else "no persisted partial carries any histogram"
        )
        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {
                    UNEVALUATED_KEY: True,
                    SKIP_REASON_KEY: reason,
                    "emitted_video_keys": list(emitted_video_keys),
                },
            ),
        )

    findings: list[dict[str, Any]] = []

    try:
        per_channel_by_video_key: dict[str, list[np.ndarray]] = {vk: [] for vk in emitted_video_keys}
        merge_admitted_episodes: dict[str, list[int]] = {vk: [] for vk in emitted_video_keys}

        for episode_index in sorted(persisted):
            histograms, included_in_merge = persisted[episode_index]

            manifest: dict[str, Any] | None
            manifest_invariant = "manifest_missing"
            manifest_error: str | None = None
            try:
                manifest = json.loads(ctx.encode_manifest_path(episode_index).read_text())
            except FileNotFoundError:
                manifest = None
            except (json.JSONDecodeError, OSError) as e:
                # A malformed or unreadable manifest for THIS episode costs
                # this episode's affected cameras one named finding each,
                # distinct from the absent-file case, and the loop below
                # still runs (has_manifest_entry is False for every camera,
                # same as the absent-manifest branch) rather than aborting
                # the whole per-episode/per-camera loop.
                manifest = None
                manifest_invariant = "manifest_unreadable"
                manifest_error = f"{type(e).__name__}: {e}"

            try:
                for video_key in emitted_video_keys:
                    has_histogram = video_key in histograms
                    has_manifest_entry = manifest is not None and video_key in manifest

                    if has_histogram and not has_manifest_entry:
                        detail = (
                            f"the encode manifest for this episode is unreadable: {manifest_error}"
                            if manifest_invariant == "manifest_unreadable"
                            else (
                                "a persisted histogram exists for this camera but its encode "
                                "manifest (or the manifest entry) is absent — interrupted-writer "
                                "state"
                            )
                        )
                        findings.append(
                            {
                                "invariant": manifest_invariant,
                                "episode_index": episode_index,
                                "video_key": video_key,
                                "detail": detail,
                            }
                        )
                        continue
                    if has_manifest_entry and not has_histogram:
                        findings.append(
                            {
                                "invariant": "histogram_missing",
                                "episode_index": episode_index,
                                "video_key": video_key,
                                "detail": (
                                    "an encode manifest entry exists for this camera but its "
                                    "persisted histogram is absent — interrupted-writer state"
                                ),
                            }
                        )
                        continue
                    if not has_histogram:
                        # Neither side has this camera for this episode — nothing
                        # to check (e.g. a camera not emitted for this episode).
                        continue

                    counts = histograms[video_key]
                    expected_mass, mass_reason = expected_channel_mass(ctx, episode_index, video_key)
                    if expected_mass is None:
                        findings.append(
                            {
                                "invariant": "expectation_unavailable",
                                "episode_index": episode_index,
                                "video_key": video_key,
                                "detail": mass_reason,
                            }
                        )
                    else:
                        per_histogram_findings = histogram_invariant_findings(counts, expected_mass)
                        for finding in per_histogram_findings:
                            findings.append(
                                {"episode_index": episode_index, "video_key": video_key, **finding}
                            )

                    if (
                        included_in_merge
                        and counts.ndim == 2
                        and counts.shape[1] == HISTOGRAM_BIN_COUNT
                    ):
                        per_channel_by_video_key[video_key].append(counts)
                        merge_admitted_episodes[video_key].append(episode_index)
            except Exception as e:  # noqa: BLE001 - one episode's fault, not the run's
                findings.append(
                    {
                        "invariant": "episode_check_error",
                        "episode_index": episode_index,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

        merge_evidence: dict[str, dict[str, Any]] = {}
        for video_key in emitted_video_keys:
            per_episode_histograms = per_channel_by_video_key[video_key]
            if not per_episode_histograms:
                continue

            summed_masses = [int(h.sum(axis=1)[0]) for h in per_episode_histograms]
            expected_merged_mass = sum(summed_masses)

            merged = Uint8Histogram.empty(per_episode_histograms[0].shape[0])
            for h in per_episode_histograms:
                merged = merged.merge(Uint8Histogram(counts=h))
            merged_masses = [int(m) for m in merged.counts.sum(axis=1)]

            merge_evidence[video_key] = {
                "admitted_episode_indices": merge_admitted_episodes[video_key],
                "per_episode_masses": summed_masses,
                "expected_merged_mass": expected_merged_mass,
                "observed_merged_masses": merged_masses,
            }
            if any(m != expected_merged_mass for m in merged_masses):
                findings.append(
                    {
                        "invariant": "merge_additivity",
                        "video_key": video_key,
                        "expected_merged_mass": expected_merged_mass,
                        "observed_merged_masses": merged_masses,
                        "admitted_episode_indices": merge_admitted_episodes[video_key],
                    }
                )
    except Exception as e:  # noqa: BLE001 - reported not raised, never lossy
        # After the per-episode isolation above, this path should be
        # unreachable for any artifact-level fault — kept as a last-resort
        # boundary anyway, and made lossless so a future edit can never
        # silently reintroduce finding loss: whatever this check had already
        # accumulated survives alongside the error.
        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                False,
                {"error": f"{type(e).__name__}: {e}", "findings": findings},
            ),
        )

    return (
        CheckResult(
            CHECK_NAME,
            "fatal",
            not findings,
            {
                "findings": findings,
                "merge_evidence": merge_evidence,
                "emitted_video_keys": list(emitted_video_keys),
            },
        ),
    )
