"""The single inspectable fatal/advisory ingest-time check registry
(INTEG-01) and every check function it registers.

`CheckRegistry.default()` is the one place INTEG-01 requires the
fatal-vs-advisory policy to live. Fatal checks are resolved entirely from
index-pass facts (`FatalCheckContext`) — before any payload decode — and a
failing fatal check aborts the episode before encode. Advisory checks are
resolved from decode-pass facts (`AdvisoryCheckContext`) and only ever
record evidence and downgrade `Disposition`; nothing here trims, repairs,
or interpolates over an anomaly.

Every check name is a dotted `{category}.{check}` drawn from the
LeRobot-ecosystem diagnostics vocabulary (`metadata`, `temporal`, `videos`,
`consistency`, `anomalies`, `statistics`, `kinematics`) — see this plan's
`<external_reference_decision>`: the taxonomy is adopted, the tool itself is
not.

This module never imports from `data.adapters.galaxea` — `galaxea.py`
builds `FatalCheckContext`/`AdvisoryCheckContext` from its own index/decode
pass results and calls into this module, not the other way around, so the
two modules stay free of a circular dependency.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import numpy as np

from data.contracts.containers import CheckResult


def compute_payload_digest(data: bytes) -> str:
    """The one place `videos.frozen_duplicate_frames` (INTEG-03) decides
    what "identical payload" means: a full `hashlib.sha256` hex digest —
    never a truncated prefix, a byte length, or a non-cryptographic
    checksum. `galaxea.py`'s index pass calls this on each camera message's
    extracted image payload; this module owns the digest algorithm even
    though the extraction happens elsewhere, so the check's contract for
    "same bytes" lives in exactly one place."""
    return hashlib.sha256(data).hexdigest()

_STATIC_SCENE_BLIND_SPOT_MARKER = "re_encoded_static_scene_not_detected"
"""D-02's documented, deliberate blind spot: a genuinely static scene that
gets re-encoded per frame is byte-different (different JPEG bytes) but
pixel-identical — frozen-frame detection by compressed-payload hash cannot
see it. Recorded in every frozen/duplicate-frame CheckResult's evidence,
never silently passed over. Closed on the other side, post-write, by
`data/validation/frame_integrity.py::check_decoded_pixel_uniqueness`
(D-10, Phase 3 plan 03-03), which works on decoded pixels instead of
compressed payload bytes — this marker's text names that check so a reader
of an ingest-time record knows where the other half of the coverage lives."""

_DEFAULT_DISCONTINUITY_THRESHOLD = 0.5
"""Soft curation knob (D-07): a per-sample joint-position jump beyond this
magnitude (profile units, e.g. radians) is flagged as a discontinuity.
Tuning this leaves `profile_hash` stable — it is a check-function
parameter, never a profile field."""

_DEFAULT_IDLE_VELOCITY_THRESHOLD = 0.01
"""Soft curation knob (D-07): a per-sample joint velocity below this
magnitude counts as "idle". Also a check-function parameter, not a profile
field."""


# ---------------------------------------------------------------------------
# Check input contracts — owned by this module, built by galaxea.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FatalCheckContext:
    """Everything a fatal check needs, resolved entirely from the index
    pass — never a decoded payload."""

    scan_error: str | None
    non_monotonic_topics: frozenset[str]
    schema_encodings: tuple[tuple[str, str], ...]
    """`(schema_name, encoding)` pairs, one per unique schema record seen."""
    master_camera_topic: str
    master_camera_message_count: int


@dataclass(frozen=True)
class JointSeries:
    """One state/action field's raw (pre-resample) decoded series — the
    advisory checks that must measure on the source signal, never the
    resampled grid, read this directly."""

    name: str
    """The topic this series was decoded from — the name every check's
    evidence attributes a flag to."""
    timestamps_ns: np.ndarray
    """`int64`, sorted ascending."""
    values: np.ndarray
    """`(N, J)` float32."""
    max_velocity: float | None = None
    """The profile's declared `StateField.max_velocity`, if any. Always
    `None` for an action field — `ActionField` declares no such field."""
    max_acceleration: float | None = None


@dataclass(frozen=True)
class AdvisoryCheckContext:
    """Everything an advisory check needs, resolved from the decode pass."""

    camera_payload_digests: Mapping[str, list[str]]
    """topic -> ordered sha256 hex digest list, gathered in the index pass
    at zero decode cost (D-02) — reused here unchanged."""
    master_camera_topic: str
    joint_series: tuple[JointSeries, ...]
    cross_stream_timestamps: Mapping[str, np.ndarray]
    """topic -> sorted `int64` timestamp array, for every decoded stream
    (camera + state + action) — INTEG-02's skew/source-age measurement."""


# ---------------------------------------------------------------------------
# Fatal checks — run before any payload decode.
# ---------------------------------------------------------------------------


def _check_mcap_crc(ctx: FatalCheckContext) -> CheckResult:
    """INTEG-01's named CRC regression check (02-RESEARCH.md Pitfall 1):
    fails whenever the index pass's CRC-validating scan hit any error —
    a corrupted chunk/data-section CRC, or any other failure surfaced while
    reading the file with `validate_crcs=True`."""
    if ctx.scan_error is not None:
        return CheckResult("metadata.mcap_crc", "fatal", False, {"error": ctx.scan_error})
    return CheckResult("metadata.mcap_crc", "fatal", True, {})


def _check_schema_decodability(ctx: FatalCheckContext) -> CheckResult:
    """Rejects any schema whose encoding is not `ros2msg` by name, rather
    than reinterpreting it as some other valid type (INTEG-01)."""
    bad = sorted({name for name, encoding in ctx.schema_encodings if encoding != "ros2msg"})
    if bad:
        return CheckResult(
            "metadata.schema_decodability", "fatal", False, {"undecodable_schemas": bad}
        )
    return CheckResult(
        "metadata.schema_decodability",
        "fatal",
        True,
        {"schema_count": len(ctx.schema_encodings)},
    )


def _check_monotonic_timestamps(ctx: FatalCheckContext) -> CheckResult:
    """Names every topic whose log-timestamp sequence went non-monotonic."""
    if ctx.non_monotonic_topics:
        return CheckResult(
            "temporal.monotonic_timestamps",
            "fatal",
            False,
            {"topics": sorted(ctx.non_monotonic_topics)},
        )
    return CheckResult("temporal.monotonic_timestamps", "fatal", True, {})


def _check_master_camera_frames(ctx: FatalCheckContext) -> CheckResult:
    """A required master-clock camera producing zero frames cannot build a
    grid — fatal, naming the camera's topic."""
    if ctx.master_camera_message_count == 0:
        return CheckResult(
            "videos.master_camera_frames", "fatal", False, {"topic": ctx.master_camera_topic}
        )
    return CheckResult(
        "videos.master_camera_frames",
        "fatal",
        True,
        {"topic": ctx.master_camera_topic, "message_count": ctx.master_camera_message_count},
    )


# ---------------------------------------------------------------------------
# Advisory checks — run after every fatal check passes, in the decode pass.
# ---------------------------------------------------------------------------


def _check_frozen_duplicate_frames(ctx: AdvisoryCheckContext) -> CheckResult:
    """Frozen/duplicate-frame detection over full sha256 digests of the
    compressed payload bytes gathered in the index pass — zero decode
    cost, and catches the failure that actually occurs: a stalled driver
    republishing an identical buffer (D-02). Iterates **every** camera
    topic present in `ctx.camera_payload_digests`, not only the master
    camera's (WR-01 — this became reachable once 02-05 activated
    secondary-camera decode: a stalled wrist-camera driver went undetected
    until this fix). Still returns exactly one `CheckResult` — the
    registry's contract is one result per registered check — with
    per-topic evidence: for each topic, its `pairs_compared` and
    `duplicate_count`, or the same explicit `unevaluated` marker the
    single-camera branch used to carry for the whole check, now applied per
    topic. Always carries the static-scene blind-spot marker. `passed` is
    true only when at least one topic had enough digests to evaluate, and
    every topic that did has a zero duplicate count — mirroring the
    single-camera implementation's own rule that a check with nothing to
    evaluate is never silently a pass."""
    per_topic: dict[str, dict[str, Any]] = {}
    any_evaluated = False
    all_zero_duplicates = True
    for topic, digests in sorted(ctx.camera_payload_digests.items()):
        pairs_compared = max(len(digests) - 1, 0)
        if pairs_compared == 0:
            per_topic[topic] = {"pairs_compared": 0, "unevaluated": True}
            continue
        any_evaluated = True
        duplicate_count = sum(1 for a, b in zip(digests, digests[1:]) if a == b)
        per_topic[topic] = {"pairs_compared": pairs_compared, "duplicate_count": duplicate_count}
        if duplicate_count != 0:
            all_zero_duplicates = False
    return CheckResult(
        "videos.frozen_duplicate_frames",
        "advisory",
        any_evaluated and all_zero_duplicates,
        {
            "per_topic": per_topic,
            "blind_spot": _STATIC_SCENE_BLIND_SPOT_MARKER,
        },
    )


def _check_joint_discontinuity(
    ctx: AdvisoryCheckContext, *, threshold: float = _DEFAULT_DISCONTINUITY_THRESHOLD
) -> CheckResult:
    """Measured on the raw decoded series, before resampling (per D-01's
    placement of advisory checks ahead of the grid) — a discontinuity in
    the source is a sensor-fault signal; a resample-induced one would be a
    synchronizer bug, out of this check's scope."""
    flagged: dict[str, float] = {}
    for series in ctx.joint_series:
        if series.values.shape[0] < 2:
            continue
        deltas = np.abs(np.diff(series.values, axis=0))
        max_delta = float(np.max(deltas)) if deltas.size else 0.0
        if max_delta > threshold:
            flagged[series.name] = max_delta
    return CheckResult(
        "consistency.joint_discontinuity",
        "advisory",
        not flagged,
        {
            "threshold": threshold,
            "measured_on": "raw_decoded_series_pre_resample",
            "flagged": dict(sorted(flagged.items())),
        },
    )


def _check_velocity_acceleration_bounds(ctx: AdvisoryCheckContext) -> CheckResult:
    """D-07/D-08: always computes and records measured velocity/acceleration
    extremes; compares against a declared profile limit when one exists and
    echoes the resolved limit into evidence; when no limit is declared for
    a field, that field carries an explicit `no_limit_declared` marker
    rather than reading as passing. Consecutive samples sharing a timestamp
    are counted, never divided into an infinity."""
    per_field: dict[str, dict[str, Any]] = {}
    any_flagged = False
    any_limit_declared = False

    for series in ctx.joint_series:
        if series.values.shape[0] < 2:
            per_field[series.name] = {
                "measured_max_velocity": 0.0,
                "measured_max_acceleration": 0.0,
                "zero_delta_sample_count": 0,
                "no_limit_declared": [
                    n
                    for n, v in (
                        ("max_velocity", series.max_velocity),
                        ("max_acceleration", series.max_acceleration),
                    )
                    if v is None
                ],
            }
            continue

        dt_ns = np.diff(series.timestamps_ns)
        zero_delta_count = int(np.sum(dt_ns == 0))
        dt_s = dt_ns.astype(np.float64) / 1e9
        valid = dt_s > 0

        if np.any(valid):
            velocity = np.diff(series.values, axis=0)[valid] / dt_s[valid, None]
            measured_max_velocity = float(np.max(np.abs(velocity))) if velocity.size else 0.0
        else:
            velocity = np.zeros((0, series.values.shape[1]), dtype=np.float64)
            measured_max_velocity = 0.0

        measured_max_acceleration = 0.0
        if velocity.shape[0] >= 2:
            dt_v_ns = dt_ns[valid][1:]
            dt_v_s = dt_v_ns.astype(np.float64) / 1e9
            valid_v = dt_v_s > 0
            if np.any(valid_v):
                acceleration = np.diff(velocity, axis=0)[valid_v] / dt_v_s[valid_v, None]
                measured_max_acceleration = (
                    float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
                )

        no_limit_declared: list[str] = []
        field_evidence: dict[str, Any] = {
            "measured_max_velocity": measured_max_velocity,
            "measured_max_acceleration": measured_max_acceleration,
            "zero_delta_sample_count": zero_delta_count,
        }
        flagged = False

        if series.max_velocity is not None:
            any_limit_declared = True
            field_evidence["resolved_max_velocity_limit"] = series.max_velocity
            if measured_max_velocity > series.max_velocity:
                flagged = True
        else:
            no_limit_declared.append("max_velocity")

        if series.max_acceleration is not None:
            any_limit_declared = True
            field_evidence["resolved_max_acceleration_limit"] = series.max_acceleration
            if measured_max_acceleration > series.max_acceleration:
                flagged = True
        else:
            no_limit_declared.append("max_acceleration")

        if no_limit_declared:
            field_evidence["no_limit_declared"] = no_limit_declared
        if flagged:
            any_flagged = True
        per_field[series.name] = field_evidence

    # D-08: with no limit declared anywhere, this check stays visibly
    # unevaluated rather than reading as passing.
    passed = any_limit_declared and not any_flagged
    return CheckResult(
        "kinematics.velocity_acceleration_bounds",
        "advisory",
        passed,
        {"per_field": dict(sorted(per_field.items())), "any_limit_declared": any_limit_declared},
    )


def _check_idle_span(
    ctx: AdvisoryCheckContext, *, velocity_threshold: float = _DEFAULT_IDLE_VELOCITY_THRESHOLD
) -> CheckResult:
    """Records per-field idle-sample counts (velocity magnitude below the
    soft threshold). Informational only — idle spans never fail this
    check; a future curation pass decides what to do with the count."""
    idle_counts: dict[str, int] = {}
    for series in ctx.joint_series:
        if series.values.shape[0] < 2:
            continue
        dt_ns = np.diff(series.timestamps_ns)
        dt_s = dt_ns.astype(np.float64) / 1e9
        valid = dt_s > 0
        if not np.any(valid):
            continue
        velocity_norm = np.linalg.norm(np.diff(series.values, axis=0)[valid], axis=1) / dt_s[
            valid
        ]
        idle_count = int(np.sum(velocity_norm < velocity_threshold))
        if idle_count:
            idle_counts[series.name] = idle_count
    return CheckResult(
        "temporal.idle_span",
        "advisory",
        True,
        {"velocity_threshold": velocity_threshold, "idle_sample_counts": dict(sorted(idle_counts.items()))},
    )


def _check_cross_stream_skew(ctx: AdvisoryCheckContext) -> CheckResult:
    """INTEG-02: cross-stream skew (max/mean over the pairwise per-tick
    timestamp differences against the master camera's own tick times) and
    per-field source age, in integer nanoseconds throughout, with no
    rejection threshold applied — the recorded distribution is what a
    future threshold gets set from. A single-stream episode (nothing to
    compare the master camera against) reports zero skew with an explicit
    marker rather than an absent measurement."""
    streams = ctx.cross_stream_timestamps
    if len(streams) < 2 or ctx.master_camera_topic not in streams:
        return CheckResult(
            "temporal.cross_stream_skew",
            "advisory",
            True,
            {
                "max_skew_ns": 0,
                "mean_skew_ns": 0,
                "single_stream": True,
                "rounding_mode": "floor",
            },
        )

    master_ts = streams[ctx.master_camera_topic]
    all_diffs: list[int] = []
    per_field_source_age: dict[str, int] = {}
    for topic, ts in streams.items():
        if topic == ctx.master_camera_topic or ts.shape[0] == 0:
            continue
        idx = np.clip(np.searchsorted(ts, master_ts), 0, ts.shape[0] - 1)
        diffs = np.abs(ts[idx] - master_ts)
        all_diffs.extend(int(d) for d in diffs)
        per_field_source_age[topic] = int(diffs[-1]) if diffs.size else 0

    if not all_diffs:
        return CheckResult(
            "temporal.cross_stream_skew",
            "advisory",
            True,
            {
                "max_skew_ns": 0,
                "mean_skew_ns": 0,
                "single_stream": True,
                "rounding_mode": "floor",
            },
        )

    max_skew = max(all_diffs)
    mean_skew = sum(all_diffs) // len(all_diffs)  # integer floor division, named below
    return CheckResult(
        "temporal.cross_stream_skew",
        "advisory",
        True,
        {
            "max_skew_ns": max_skew,
            "mean_skew_ns": mean_skew,
            "rounding_mode": "floor",
            "per_field_source_age_ns": dict(sorted(per_field_source_age.items())),
        },
    )


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One registered check: its dotted name, severity, and callable."""

    name: str
    severity: Literal["fatal", "advisory"]
    fn: Callable[[Any], CheckResult]


@dataclass(frozen=True)
class CheckRegistry:
    """The single inspectable place INTEG-01 requires the fatal-vs-advisory
    policy to live. Checks run in the declared, stable order below —
    `run_fatal`/`run_advisory` preserve that order in the returned tuple, so
    two runs over the same fixture produce identical check sequences."""

    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def fatal(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.severity == "fatal")

    @property
    def advisory(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.severity == "advisory")

    @classmethod
    def default(cls) -> "CheckRegistry":
        return cls(
            (
                Check("metadata.mcap_crc", "fatal", _check_mcap_crc),
                Check("metadata.schema_decodability", "fatal", _check_schema_decodability),
                Check("temporal.monotonic_timestamps", "fatal", _check_monotonic_timestamps),
                Check("videos.master_camera_frames", "fatal", _check_master_camera_frames),
                Check(
                    "videos.frozen_duplicate_frames", "advisory", _check_frozen_duplicate_frames
                ),
                Check("consistency.joint_discontinuity", "advisory", _check_joint_discontinuity),
                Check(
                    "kinematics.velocity_acceleration_bounds",
                    "advisory",
                    _check_velocity_acceleration_bounds,
                ),
                Check("temporal.idle_span", "advisory", _check_idle_span),
                Check("temporal.cross_stream_skew", "advisory", _check_cross_stream_skew),
            )
        )

    def run_fatal(self, ctx: FatalCheckContext) -> tuple[CheckResult, ...]:
        return tuple(c.fn(ctx) for c in self.fatal)

    def run_advisory(self, ctx: AdvisoryCheckContext) -> tuple[CheckResult, ...]:
        return tuple(c.fn(ctx) for c in self.advisory)
