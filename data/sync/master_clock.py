"""Master-clock synchronizer — the full per-field resample dispatch (02-04).

`MasterClockSynchronizer` builds the idealized `frame_index / fps` grid from
the master-clock camera's own buffered timestamps (never a raw capture
timestamp — see `CanonicalFrame.timestamp_ns`'s derivation below) and
resamples every state/action/secondary-camera series onto it, driven entirely
by each field's declared `resample` policy (`interp` / `zoh`) or a secondary
camera's `sync_policy` (`nearest`, with a declared gap bound).

Every nanosecond timestamp is kept as a Python/NumPy `int64` end to end and
only ever converted to `float64` **relative to a local reference point**
(the resampled series' own first sample timestamp, `sample_ts_ns[0]` after
duplicate-timestamp collapse — not the master camera's first frame)
immediately before an interpolation call — an absolute Unix nanosecond
timestamp (~1.7e18) exceeds `float64`'s ~2^53 exact-integer range, so
converting it directly to float anywhere would silently lose precision.
The *same* reference value is subtracted from both the spline's knot array
and its evaluation array; subtracting it from only one side does not raise,
but silently returns values wrong by ~1e16, so a future edit must keep both
sides referencing the one bound local name rather than recomputing either.

The grid rate is carried as a `fractions.Fraction` — constructed once, reused
for every derived timestamp — so no accumulated float step error appears
across an episode's ~1,200 ticks. Every emitted `CanonicalFrame.timestamp_ns`
derives from `frame_index / fps` using that Fraction; no raw capture
timestamp ever reaches an emitted frame.

New in this module (02-04, PROV-03): the populated `ClockReport` (grid rate,
integer-nanosecond skew with no rejection threshold, per-field source age,
resolved action source per field) and the secondary-camera nearest-match with
gap-bound rejection. Both live in `EpisodeHeader.global_meta`, alongside two
new companion keys the report doesn't itself carry a field for:
`global_meta["per_field_resample_mode"]` (the resampling mode that produced
each `per_field_source_age_ns` entry — `ClockReport` itself has no such
field, so it travels here) and `global_meta["skew_rounding_mode"]` /
`global_meta["single_stream"]` (the rounding convention and the
no-other-stream marker for the skew measurement). See this plan's own
SUMMARY for the exact final key names.

New in this module (04-04-PLAN.md, G-01/VERIFY-01 gap closure): this module
now trims the emitted grid to the coverage every constraining camera
actually has, and records the trim and its per-camera reasons in a named
`cameras.coverage_deficit` check (`CAMERA_COVERAGE_CHECK_NAME`) — see
`camera_coverage_report`/`coverage_check_result` below. It still never
repairs, gap-fills, or synthesizes a frame: a camera whose own recording
ends before the master clock's simply shrinks the emitted grid rather than
having a frame held or invented for it.

"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import make_interp_spline

from data.contracts.containers import (
    CanonicalEpisode,
    CanonicalFrame,
    CheckResult,
    ClockReport,
    EpisodeHeader,
)
from data.contracts.profile import EmbodimentProfile

_SKEW_ROUNDING_MODE = "floor"

CAMERA_COVERAGE_CHECK_NAME = "cameras.coverage_deficit"
"""G-01/VERIFY-01: the named check carrying every episode's camera-coverage
truncation annotation. Named for the mechanism (a coverage deficit measured
as a contiguous covered prefix), not G-01's illustrative example
(`trailing_coverage_deficit`) — the same arithmetic also catches an
interior hole, and the name must not promise a cause it cannot guarantee.
`data/adapters/galaxea.py::_enforce_camera_coverage` imports this constant
rather than duplicating the string literal, so a rename here cannot
silently disable the pre-write rejection."""


@dataclass(frozen=True, slots=True)
class CameraCoverage:
    """One camera's measured coverage of the master-clock grid.

    `covered_prefix_ticks` is the largest `k` for which this camera has a
    valid nearest-match (`nearest_camera_matches`) at every grid tick below
    `k` — the contiguous covered prefix, not merely a count of matched
    ticks (a single interior miss must trim from the miss, never leave a
    later frame silently shifted onto the wrong tick).

    `hole_kind` is `""` when the camera covered every tick,
    `"stream_ended_before_master"` when it has no buffered frame at or after
    the first uncovered tick's own timestamp, and
    `"gap_exceeded_max_sync_gap_ms"` when it does have a later frame (an
    interior hole, distinct from a stream that simply ended). `hole_gap_ns`
    is the measured nanosecond distance at the first uncovered tick — `-1`
    when there is no hole, matching `nearest_camera_matches`' own "no frames
    at all" sentinel convention (which also yields `-1` here, since a
    zero-frame camera's first uncovered tick is tick `0`) rather than
    inventing a second sentinel.
    """

    camera_name: str
    video_key: str
    covered_prefix_ticks: int
    hole_kind: str
    hole_gap_ns: int
    required: bool
    emitted: bool


@dataclass(frozen=True, slots=True)
class CameraCoverageReport:
    """The whole-episode coverage verdict `synchronize` computes once,
    before it trims anything.

    `kept_ticks` is `grid_ticks` when no camera constrains the grid,
    otherwise the minimum `covered_prefix_ticks` across every constraining
    camera — a `min` over integers, so two cameras with equal covered
    prefixes cannot produce a different outcome depending on iteration
    order. `within_tolerance` is `deficit_ticks <= tolerance_ticks` —
    inclusive at the boundary, deliberately: a future edit must not quietly
    flip this to a strict comparison.
    """

    grid_ticks: int
    kept_ticks: int
    deficit_ticks: int
    tolerance_ticks: int
    within_tolerance: bool
    shortfall_cameras: tuple[CameraCoverage, ...]


def camera_coverage_report(
    camera_matches: Mapping[str, tuple[np.ndarray, np.ndarray, list[Any], str, np.ndarray]],
    profile: EmbodimentProfile,
    grid_ts_ns: np.ndarray,
    tolerance_ticks: int,
) -> CameraCoverageReport:
    """Builds the coverage report from `synchronize`'s already-computed
    `camera_matches` mapping — `{cam_name: (matched_idx, gap_ns,
    frame_refs_cam, video_key, frame_ts_ns)}`. Never recomputes a nearest
    match itself; it only measures the contiguous covered prefix out of
    `matched_idx`, which `nearest_camera_matches` already produced.

    The constraining set is every entry in `camera_matches` whose profile
    `CameraSpec` is `required` or `emit` (the widening this plan states
    explicitly: an `emit: true, required: false` camera gets an MP4 and can
    therefore break the parity invariant, so it must constrain even though
    G-01's wording says "required") — and, when a camera is absent from
    `profile.cameras` entirely, it is treated as constraining too, matching
    this module's existing tolerant fallback for a missing `video_key`
    rather than silently exempting an unknown camera.

    All tick-count arithmetic below is NumPy/Python integer end to end —
    `np.argmax` over a boolean mask, integer array indexing, and a `min`
    over a Python list of `int`s — so no float ever enters the deficit or
    tolerance comparison.
    """
    grid_ticks = int(grid_ts_ns.shape[0])
    coverages: list[CameraCoverage] = []
    constraining_prefixes: list[int] = []

    for cam_name, (matched_idx, gap_ns, _frame_refs, video_key, frame_ts_ns) in (
        camera_matches.items()
    ):
        unmatched = matched_idx < 0
        if not bool(unmatched.any()):
            covered_prefix_ticks = grid_ticks
            hole_kind = ""
            hole_gap_ns = -1
        else:
            # First True in a boolean array is np.argmax's own first-match
            # convention — never a Python loop over ticks.
            covered_prefix_ticks = int(np.argmax(unmatched))
            hole_gap_ns = int(gap_ns[covered_prefix_ticks])
            first_uncovered_ts = grid_ts_ns[covered_prefix_ticks]
            has_later_frame = bool(np.any(frame_ts_ns >= first_uncovered_ts))
            hole_kind = (
                "gap_exceeded_max_sync_gap_ms"
                if has_later_frame
                else "stream_ended_before_master"
            )

        cam_profile = profile.cameras.get(cam_name)
        required = cam_profile.required if cam_profile is not None else True
        emitted = cam_profile.emit if cam_profile is not None else True
        is_constraining = cam_profile is None or cam_profile.required or cam_profile.emit

        coverages.append(
            CameraCoverage(
                camera_name=cam_name,
                video_key=video_key,
                covered_prefix_ticks=covered_prefix_ticks,
                hole_kind=hole_kind,
                hole_gap_ns=hole_gap_ns,
                required=required,
                emitted=emitted,
            )
        )
        if is_constraining:
            constraining_prefixes.append(covered_prefix_ticks)

    kept_ticks = min(constraining_prefixes) if constraining_prefixes else grid_ticks
    deficit_ticks = grid_ticks - kept_ticks
    within_tolerance = deficit_ticks <= tolerance_ticks
    shortfall_cameras = tuple(
        sorted(
            (c for c in coverages if c.covered_prefix_ticks < grid_ticks),
            key=lambda c: c.camera_name,
        )
    )

    return CameraCoverageReport(
        grid_ticks=grid_ticks,
        kept_ticks=kept_ticks,
        deficit_ticks=deficit_ticks,
        tolerance_ticks=tolerance_ticks,
        within_tolerance=within_tolerance,
        shortfall_cameras=shortfall_cameras,
    )


def coverage_check_result(report: CameraCoverageReport) -> CheckResult:
    """One `CheckResult` for `CAMERA_COVERAGE_CHECK_NAME`, severity `fatal`
    (INTEG-01's policy: a failing fatal check aborts the episode before
    encode — enforced by `data/adapters/galaxea.py::_enforce_camera_coverage`,
    which reads this result rather than recomputing anything). `passed` is
    `report.within_tolerance`. The evidence IS the truncation annotation:
    `ProvenanceRecorder.record` already copies `global_meta["checks"]`
    verbatim into the per-episode record and `data/cli.py`'s
    `_serialize_check` already copies evidence into the run report, so this
    is the whole annotation mechanism — no record-schema change, no parquet
    column (D-05)."""
    evidence: dict[str, Any] = {
        "grid_ticks": report.grid_ticks,
        "kept_ticks": report.kept_ticks,
        "deficit_ticks": report.deficit_ticks,
        "tolerance_ticks": report.tolerance_ticks,
        "truncated": report.deficit_ticks > 0,
        "shortfall_cameras": [
            {
                "camera_name": c.camera_name,
                "video_key": c.video_key,
                "covered_prefix_ticks": c.covered_prefix_ticks,
                "hole_kind": c.hole_kind,
                "hole_gap_ns": c.hole_gap_ns,
                "required": c.required,
                "emitted": c.emitted,
            }
            for c in report.shortfall_cameras
        ],
    }
    return CheckResult(CAMERA_COVERAGE_CHECK_NAME, "fatal", report.within_tolerance, evidence)


class MasterClockSynchronizerError(Exception):
    """Raised when the master-clock camera stream cannot be built into a
    grid (no master_clock camera in `raw_streams`, it produced zero or one
    frame with no declared target rate to fall back on, or a required
    action field has no populated source). Names the camera/topic/field
    involved."""


def _resample_series(
    sample_ts_ns: np.ndarray,
    sample_values: np.ndarray,
    grid_ts_ns: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Resamples one field's `(sample_ts_ns, sample_values)` onto
    `grid_ts_ns`. `sample_values` is `(N, J)`; returns `(len(grid_ts_ns), J)`
    `float32`.

    `mode == "interp"`: linear interpolation via
    `scipy.interpolate.make_interp_spline(..., k=1, axis=0)` — vectorized
    across all J joints in one call (never a per-joint Python loop). Grid
    timestamps outside the sample series' own span are clipped to that
    span's endpoints rather than extrapolated.

    `mode in ("zoh", "nearest")`: previous-or-nearest known sample, via
    `np.searchsorted` — zero-order hold. A grid tick before the first sample
    holds the first sample (never extrapolates backward into an undefined
    value).

    Caller never calls this with a field that has zero buffered samples —
    that case is filtered out before dispatch, so the field is absent from
    the emitted mapping entirely rather than zero-filled (never a silently
    manufactured measurement).
    """
    n_grid = grid_ts_ns.shape[0]
    n_joints = sample_values.shape[1] if sample_values.ndim == 2 else 0
    if sample_values.shape[0] == 0:
        return np.zeros((n_grid, n_joints), dtype=np.float32)

    if mode == "interp":
        # `make_interp_spline` rejects duplicate x-values outright — a
        # zero-timestamp-delta pair (INTEG-04's zero-delta case) would
        # otherwise crash the resample rather than being handled. Collapse
        # consecutive samples sharing an identical timestamp, keeping the
        # LAST value recorded at that instant (the more complete/final
        # measurement), before building the spline.
        unique_ts_ns = np.unique(sample_ts_ns)
        if unique_ts_ns.shape[0] != sample_ts_ns.shape[0]:
            last_value_by_ts: dict[int, np.ndarray] = {}
            for ts, val in zip(sample_ts_ns.tolist(), sample_values):
                last_value_by_ts[ts] = val
            sample_ts_ns = np.array(sorted(last_value_by_ts), dtype=sample_ts_ns.dtype)
            sample_values = np.stack([last_value_by_ts[ts] for ts in sample_ts_ns.tolist()])

        if sample_ts_ns.shape[0] == 1:
            return np.repeat(sample_values.astype(np.float32), n_grid, axis=0)
        # A real Unix-epoch nanosecond timestamp (~1.7e18) exceeds float64's
        # ~2^53 exact-integer range: float64's ULP at that magnitude is
        # ~256ns, so two distinct int64 timestamps closer together than that
        # would collapse to the identical float64 value and re-trigger the
        # duplicate-knot error the block above already de-duplicated against
        # (CR-01, 03-REVIEW.md). Bind one reference point after the
        # duplicate-collapse has run (so it names a real remaining knot) and
        # subtract it from BOTH the knots and the evaluation points before
        # casting to float64 — never recompute it, so the two sides cannot
        # desynchronize. `np.clip` stays in int64, before the subtraction:
        # clipping after the float64 cast would reintroduce the precision
        # loss this fix removes.
        reference_ts_ns = sample_ts_ns[0]
        spline = make_interp_spline(
            (sample_ts_ns - reference_ts_ns).astype(np.float64),
            sample_values.astype(np.float64),
            k=1,
            axis=0,
        )
        clipped = np.clip(grid_ts_ns, sample_ts_ns[0], sample_ts_ns[-1])
        return spline((clipped - reference_ts_ns).astype(np.float64)).astype(np.float32)

    # zoh / nearest: hold the most recent sample at or before each grid
    # tick; a tick before the first sample holds the first sample.
    idx = np.searchsorted(sample_ts_ns, grid_ts_ns, side="right") - 1
    idx = np.clip(idx, 0, sample_ts_ns.shape[0] - 1)
    return sample_values[idx].astype(np.float32)


def _source_age_ns(sample_ts_ns: np.ndarray, grid_ts_ns: np.ndarray, mode: str) -> np.ndarray:
    """Per-tick integer-nanosecond distance from each grid tick to the source
    sample the resampler actually consumed for `mode`.

    `interp`: the nearer of the two bracketing samples (interpolation blends
    both, but the nearer one's distance is the reported "age" — the plan's
    own definition). `zoh` / `nearest`: the held previous-or-nearest sample's
    distance. Never called with zero samples (caller filters those out).
    """
    n_grid = grid_ts_ns.shape[0]
    if sample_ts_ns.shape[0] == 0:
        return np.zeros(n_grid, dtype=np.int64)
    if sample_ts_ns.shape[0] == 1:
        return np.abs(grid_ts_ns - sample_ts_ns[0]).astype(np.int64)

    if mode == "interp":
        idx_right = np.clip(
            np.searchsorted(sample_ts_ns, grid_ts_ns, side="left"), 0, sample_ts_ns.shape[0] - 1
        )
        idx_left = np.clip(idx_right - 1, 0, sample_ts_ns.shape[0] - 1)
        gap_right = np.abs(sample_ts_ns[idx_right] - grid_ts_ns)
        gap_left = np.abs(sample_ts_ns[idx_left] - grid_ts_ns)
        return np.minimum(gap_right, gap_left).astype(np.int64)

    idx = np.searchsorted(sample_ts_ns, grid_ts_ns, side="right") - 1
    idx = np.clip(idx, 0, sample_ts_ns.shape[0] - 1)
    return np.abs(grid_ts_ns - sample_ts_ns[idx]).astype(np.int64)


def nearest_camera_matches(
    frame_ts_ns: np.ndarray, grid_ts_ns: np.ndarray, max_gap_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized nearest-match, with a gap bound, of a secondary camera's
    own frame timestamps onto every grid tick at once.

    Returns `(matched_idx, gap_ns)`, both `int64`, length `len(grid_ts_ns)`.
    `matched_idx[i] == -1` whenever no frame is within `max_gap_ns` of tick
    `i` — never a stale frame silently attached. `gap_ns[i] == -1` when the
    camera has zero buffered frames at all (distinguishing "no frames exist"
    from a genuine zero-distance match, which is a real, valid measurement).

    Deliberately public (D-12, Phase 3 plan 03-03): `data/validation/
    frame_integrity.py`'s `videos.spot_decode_content` check calls this
    directly to recompute the frame-to-source mapping it verifies against.
    A second, independent implementation of this arithmetic inside the
    validator would risk reproducing this function's own bugs and then
    certifying them as correct — the check is only meaningful if it shares
    this one implementation, never a copy of it.
    """
    n_grid = grid_ts_ns.shape[0]
    if frame_ts_ns.shape[0] == 0:
        return (
            np.full(n_grid, -1, dtype=np.int64),
            np.full(n_grid, -1, dtype=np.int64),
        )
    idx_right = np.searchsorted(frame_ts_ns, grid_ts_ns, side="left")
    idx_right_c = np.clip(idx_right, 0, frame_ts_ns.shape[0] - 1)
    idx_left_c = np.clip(idx_right - 1, 0, frame_ts_ns.shape[0] - 1)
    gap_right = np.abs(frame_ts_ns[idx_right_c] - grid_ts_ns)
    gap_left = np.abs(frame_ts_ns[idx_left_c] - grid_ts_ns)
    use_right = gap_right <= gap_left
    best_idx = np.where(use_right, idx_right_c, idx_left_c)
    best_gap = np.where(use_right, gap_right, gap_left)
    matched = best_gap <= max_gap_ns
    out_idx = np.where(matched, best_idx, -1)
    return out_idx.astype(np.int64), best_gap.astype(np.int64)


def _detect_native_fps(grid_ts_ns: np.ndarray, topic: str) -> Fraction:
    """The master camera's measured native rate as an exact `Fraction`,
    derived from the mean inter-frame interval over the whole buffered span
    — never a raw per-pair jitter value, and never a rounded float.

    A stream with fewer than two frames has no detectable rate: this raises
    rather than substituting any rate literal (SYNC-01). Only called when
    the profile declares no `target_fps` — a declared rate never needs
    detection at all, so a one-frame stream is not an error in that case.
    """
    n = grid_ts_ns.shape[0]
    if n < 2:
        raise MasterClockSynchronizerError(
            f"master-clock camera {topic!r} produced {n} frame(s) — a rate cannot be "
            "detected from fewer than two frames, and no target_fps is declared"
        )
    span_ns = int(grid_ts_ns[-1] - grid_ts_ns[0])
    if span_ns <= 0:
        raise MasterClockSynchronizerError(
            f"master-clock camera {topic!r}'s frames span {span_ns} ns — a rate cannot "
            "be detected from a non-positive span"
        )
    return Fraction(10**9 * (n - 1), span_ns)


def _resolve_action_sources(
    action_streams: Mapping[str, Any], profile: EmbodimentProfile
) -> dict[str, str]:
    """Resolves `resolved_action_source` for every `action_topics` key.

    `action_streams` (`raw_streams["action"]`) already carries, per action
    key present, the single winning source's topic under `"topic"` — the
    ordering mechanism itself already ran upstream of this module (the
    caller only ever buffers the first source in declared order with any
    messages). This function's job is narrower and independently
    verifiable: for every action key the profile declares, name the winning
    topic when one resolved, record an empty-string resolution for an
    optional field that resolved to nothing, and abort — naming the field
    and every source topic it tried — for a *required* field that resolved
    to nothing. A field is "resolved to nothing" when it is absent from
    `action_streams`, or present with zero buffered samples (never treated
    as a phantom zero-filled resolution).
    """
    resolved: dict[str, str] = {}
    missing_required: list[str] = []
    for action_key, action_field in profile.action_topics.items():
        entry = action_streams.get(action_key)
        if entry is not None and entry.get("samples"):
            resolved[action_key] = entry["topic"]
        elif action_field.required:
            tried = ", ".join(s.topic for s in action_field.sources)
            missing_required.append(f"{action_key} (tried: {tried})")
        else:
            resolved[action_key] = ""
    if missing_required:
        raise MasterClockSynchronizerError(
            "required action field(s) resolved to no populated source: "
            + "; ".join(sorted(missing_required))
        )
    return resolved


class MasterClockSynchronizer:
    """`BaseSynchronizer` (data/contracts/protocols.py) by structural
    typing — no import from, no inheritance of, that module."""

    def synchronize(
        self, raw_streams: Mapping[str, Any], profile: EmbodimentProfile
    ) -> CanonicalEpisode:
        """Resamples `raw_streams` onto the master-clock camera's own grid.

        `raw_streams` is the plain dict shape `data/adapters/galaxea.py`'s
        decode pass builds — see that module's `_run_decode_pass` docstring.
        This module additionally tolerates (does not require) secondary
        camera entries under `raw_streams["cameras"]` beyond the master —
        any entry whose key is not the master-clock camera's own key is
        treated as a `sync_policy: nearest` secondary camera, resolved
        against `profile.cameras` for `video_key`/`max_sync_gap_ms` when the
        entry itself omits them.
        """
        cameras = raw_streams["cameras"]
        master_name = next(
            (name for name, c in cameras.items() if c["role"] == "master_clock"),
            None,
        )
        if master_name is None:
            raise MasterClockSynchronizerError(
                "raw_streams carries no role=master_clock camera entry"
            )
        master = cameras[master_name]
        frames_sorted = sorted(master["frames"], key=lambda pair: pair[0])
        if not frames_sorted:
            raise MasterClockSynchronizerError(
                f"master-clock camera {master['topic']!r} produced zero frames "
                "— cannot build a grid with no ticks"
            )

        grid_ts_ns = np.array([ts for ts, _ in frames_sorted], dtype=np.int64)
        master_frame_refs = [ref for _, ref in frames_sorted]
        n = grid_ts_ns.shape[0]

        if profile.master_clock.target_fps is not None:
            fps_fraction = Fraction(profile.master_clock.target_fps)
        else:
            fps_fraction = _detect_native_fps(grid_ts_ns, master["topic"])

        # Action-source resolution — raises before any resampling work if a
        # required field has no populated source (fail fast).
        resolved_action_source = _resolve_action_sources(
            raw_streams.get("action", {}), profile
        )

        state_arrays: dict[str, np.ndarray] = {}
        per_field_source_age_ns: dict[str, int] = {}
        per_field_resample_mode: dict[str, str] = {}
        skew_components: list[np.ndarray] = []

        for name, entry in raw_streams.get("state", {}).items():
            samples = sorted(entry["samples"], key=lambda pair: pair[0])
            if not samples:
                # Never zero-filled: a declared-optional field with no
                # source samples is simply absent, from every mapping.
                continue
            sample_ts = np.array([ts for ts, _ in samples], dtype=np.int64)
            sample_vals = np.stack([v for _, v in samples]).astype(np.float32)
            mode = entry["resample"]
            resampled = _resample_series(sample_ts, sample_vals, grid_ts_ns, mode)

            state_field = profile.state_topics.get(name) or profile.extra_state_topics.get(
                name
            )
            include = state_field.include_in_observation_state if state_field else True
            if include:
                state_arrays[entry["state_key"]] = resampled

            age = _source_age_ns(sample_ts, grid_ts_ns, mode)
            key = f"state.{entry['state_key']}"
            per_field_source_age_ns[key] = int(age.sum() // age.size) if age.size else 0
            per_field_resample_mode[key] = mode
            skew_components.append(age)

        action_arrays: dict[str, np.ndarray] = {}
        for action_key, entry in raw_streams.get("action", {}).items():
            samples = sorted(entry["samples"], key=lambda pair: pair[0])
            if not samples:
                continue
            sample_ts = np.array([ts for ts, _ in samples], dtype=np.int64)
            sample_vals = np.stack([v for _, v in samples]).astype(np.float32)
            mode = entry["resample"]
            action_arrays[action_key] = _resample_series(
                sample_ts, sample_vals, grid_ts_ns, mode
            )

            age = _source_age_ns(sample_ts, grid_ts_ns, mode)
            key = f"action.{action_key}"
            per_field_source_age_ns[key] = int(age.sum() // age.size) if age.size else 0
            per_field_resample_mode[key] = mode
            skew_components.append(age)

        # Secondary cameras: any cameras-dict entry that is not the master.
        # Tolerated, not required — a caller (galaxea.py) that never
        # populates one simply yields an episode with no secondary-camera
        # entries at all, identical to today's behavior.
        camera_matches: dict[str, tuple[np.ndarray, np.ndarray, list[Any], str, np.ndarray]] = {}
        for cam_name, cam in cameras.items():
            if cam_name == master_name:
                continue
            cam_profile = profile.cameras.get(cam_name)
            frames_cam = sorted(cam.get("frames", []), key=lambda pair: pair[0])
            frame_ts_ns = np.array([ts for ts, _ in frames_cam], dtype=np.int64)
            frame_refs_cam = [ref for _, ref in frames_cam]
            max_gap_ms = cam.get("max_sync_gap_ms")
            if max_gap_ms is None and cam_profile is not None:
                max_gap_ms = cam_profile.max_sync_gap_ms
            max_gap_ns = int((max_gap_ms or 0) * 1_000_000)
            video_key = (
                cam.get("video_key")
                or (cam_profile.video_key if cam_profile is not None else None)
                or cam_name
            )
            matched_idx, gap_ns = nearest_camera_matches(frame_ts_ns, grid_ts_ns, max_gap_ns)
            camera_matches[cam_name] = (matched_idx, gap_ns, frame_refs_cam, video_key, frame_ts_ns)

            key = f"camera.{cam_name}"
            valid_gaps = gap_ns[matched_idx >= 0]
            per_field_source_age_ns[key] = (
                int(valid_gaps.sum() // valid_gaps.size) if valid_gaps.size else 0
            )
            per_field_resample_mode[key] = "nearest"
            if frame_ts_ns.shape[0] > 0:
                skew_components.append(np.abs(gap_ns[gap_ns >= 0]))

        if skew_components:
            all_ages = np.concatenate(skew_components)
            max_skew_ns = int(all_ages.max())
            mean_skew_ns = int(all_ages.sum() // all_ages.size)
            single_stream = False
        else:
            max_skew_ns = 0
            mean_skew_ns = 0
            single_stream = True

        clock_report = ClockReport(
            float(fps_fraction),
            max_skew_ns,
            mean_skew_ns,
            dict(sorted(per_field_source_age_ns.items())),
            dict(sorted(resolved_action_source.items())),
        )

        # G-01/VERIFY-01: measure every constraining camera's coverage of
        # this grid and trim the emitted iteration to the common coverage.
        # `grid_ts_ns`, the resampled state/action arrays above, and every
        # per-field source age are deliberately left computed over the FULL
        # grid (`n`) — trimming only the frame-generator's own iteration
        # count is what shrinks the parquet, and re-slicing every array
        # would be a second place the trim could disagree with itself.
        coverage_report = camera_coverage_report(
            camera_matches,
            profile,
            grid_ts_ns,
            profile.master_clock.max_camera_coverage_deficit_ticks,
        )
        kept = coverage_report.kept_ticks
        existing_checks = tuple(raw_streams.get("meta", {}).get("checks", ()))

        def _frames() -> Any:
            for frame_index in range(kept):
                if fps_fraction != 0:
                    timestamp_ns = int(
                        round(Fraction(frame_index) * Fraction(10**9) / fps_fraction)
                    )
                else:
                    timestamp_ns = int(grid_ts_ns[frame_index] - grid_ts_ns[0])

                images: dict[str, Any] = {
                    master["video_key"]: master_frame_refs[frame_index]
                }
                camera_gaps: dict[str, int] = {}
                for cam_name, (matched_idx, gap_ns, frame_refs_cam, video_key, _frame_ts_ns) in (
                    camera_matches.items()
                ):
                    idx = int(matched_idx[frame_index])
                    if idx >= 0:
                        images[video_key] = frame_refs_cam[idx]
                    else:
                        camera_gaps[cam_name] = int(gap_ns[frame_index])
                frame_meta: dict[str, Any] = {}
                if camera_gaps:
                    frame_meta["secondary_camera_gaps_ns"] = camera_gaps

                yield CanonicalFrame(
                    timestamp_ns=timestamp_ns,
                    frame_index=frame_index,
                    images=images,
                    state={key: arr[frame_index] for key, arr in state_arrays.items()},
                    action={key: arr[frame_index] for key, arr in action_arrays.items()},
                    meta=frame_meta,
                )

        meta = dict(raw_streams.get("meta", {}))
        meta["clock_report"] = clock_report
        meta["skew_rounding_mode"] = _SKEW_ROUNDING_MODE
        meta["single_stream"] = single_stream
        meta["per_field_resample_mode"] = dict(sorted(per_field_resample_mode.items()))
        # G-01/VERIFY-01: append, never insert -- preserves the check
        # registry's existing declared order (INTEG-01).
        meta["checks"] = existing_checks + (coverage_check_result(coverage_report),)

        header = EpisodeHeader(
            episode_id=meta.get("episode_id", "episode"),
            fps=float(fps_fraction),
            robot_type=profile.robot_type,
            task_prompt=None,
            frame_count=kept,
            global_meta=meta,
        )
        return CanonicalEpisode(header=header, frames=_frames())
