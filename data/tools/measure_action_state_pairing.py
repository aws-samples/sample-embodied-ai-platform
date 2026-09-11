#!/usr/bin/env python3
"""Permanent pairing-measurement CLI — Defect B of D-GAP-01 (VALID-03).

Answers, by measurement against a real Galaxea recording rather than a
session assertion, whether `consistency.action_state_lag`'s action/state
pairing (`data/validation/correlation.py`) is correct: is the profile's
declared action source an absolute position target or an achieved delta,
what lag (if any) a correlation search actually finds on real motion, and
whether the target stream leads or lags the achieved feedback stream. See
`03-11-PLAN.md`'s `<measured_grounding>` for the findings this tool exists
to re-confirm, and `data/validation/correlation.py`'s module docstring for
the decided consequence.

Standing tooling (D-05), in the same tier as `data/tools/inspect_mcap.py` —
re-runnable against every new recording batch, writes nothing to disk beyond
stdout/JSON. Imports neither `cv2` nor `av`: this tool has no reason to
decode a single pixel, and `01-RESEARCH.md`'s Common Pitfall #3 measured a
macOS-only dylib symbol collision between those two libraries, so keeping
both out means every command below runs identically on the macOS dev
machine and the Linux GPU workstation.

Reuses, never reimplements, the engine's own alignment and lag primitives
(D-12 discipline): `align_series` from `data.tools.inspect_mcap` and
`peak_lag_timesteps`/`MAX_LAG_SEARCH` from `data.validation.correlation` — a
second copy of either could reproduce its own bug and then certify it as
correct.

Every decode path is wrapped in its own exception boundary and reports a
named error rather than crashing: this tool exists to run against
recordings that have not been validated yet, so a malformed file is its
normal case, not an exceptional one.
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from data.adapters.galaxea import GalaxeaAdapterError, iter_decoded_messages
from data.contracts.profile import ActionField, EmbodimentProfile, StateField, load_profile
from data.tools.inspect_mcap import align_series
from data.validation.correlation import MAX_LAG_SEARCH, peak_lag_timesteps

CANDIDATE_PAIRINGS: tuple[str, ...] = (
    "delta_vs_action",
    "delta_vs_delta",
    "delta_vs_tracking_error",
)
"""The three candidate correlation pairings Defect B's measurement compares:
`diff(state)[t]` against `action[:-1]` (the shipped pairing), against
`diff(action)`, and against `(action - state)[:-1]`."""

CLOCK_SOURCES: tuple[str, ...] = ("log_time", "header_stamp")
"""`log_time` is the MCAP record's own write time (what
`iter_decoded_messages` yields today, and therefore what every existing
`<measured_grounding>` finding was taken against). `header_stamp` is the
publisher's `header.stamp` on the decoded message — available on every
state and action message this tool reads, since `sensor_msgs/msg/JointState`
and `hdas_msg/msg/MotorControl` both begin with a `std_msgs/Header header`
field."""

_GRID_PERIOD_NS = round(1_000_000_000 / 30)
"""The persisted camera-grid period this dataset's writer actually targets
(~30 Hz — `galaxea_r1_pro.yaml`'s measured native rates across all three
sample recordings). Used only to express a clock skew in units directly
comparable to one camera tick; it has no effect on the pairing measurement
itself, and no `--decimate` value changes it."""


def _header_stamp_ns(msg: Any) -> int | None:
    """`sec * 1e9 + nanosec` from `msg.header.stamp`, or `None` when
    `header`/`stamp` is absent, or when the stamp is the all-zero sentinel a
    publisher that never populated it would leave behind — a real,
    reportable fact about the recording, never fabricated as zero."""
    header = getattr(msg, "header", None)
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    if sec == 0 and nanosec == 0:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def _resolve_ts(msg: Any, log_time_ns: int, clock: str) -> int | None:
    """The message's timestamp under `clock` — `None` when `clock ==
    "header_stamp"` and no usable stamp exists (see `_header_stamp_ns`),
    never `log_time_ns` as a silent fallback: mixing clocks within one
    measurement is exactly the ambiguity this tool exists to avoid."""
    if clock == "log_time":
        return log_time_ns
    return _header_stamp_ns(msg)


class _Pair:
    """One `action_topics` key resolved against its paired
    `state_topics` entry, or a named reason it could not be."""

    __slots__ = ("state_field", "action_field", "skip_reason")

    def __init__(
        self,
        *,
        state_field: StateField | None = None,
        action_field: ActionField | None = None,
        skip_reason: str | None = None,
    ) -> None:
        self.state_field = state_field
        self.action_field = action_field
        self.skip_reason = skip_reason


def _derive_pairs(profile: EmbodimentProfile) -> dict[str, _Pair]:
    """For every `action_topics` entry, the `state_topics` entry whose
    `state_key` equals that action key — derived from the profile, never
    hardcoded. An `action_topics` key with no matching `state_topics` entry,
    or whose `joint_names` length disagrees with its paired state field's,
    is reported as a skipped pair with a named reason rather than raising:
    Defect A's channel-mismatch case is data this tool must describe, not
    reject.
    """
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    pairs: dict[str, _Pair] = {}
    for action_field in profile.action_topics.values():
        action_key = action_field.action_key
        state_field = by_state_key.get(action_key)
        if state_field is None:
            pairs[action_key] = _Pair(
                skip_reason=f"no state_topics entry has state_key == {action_key!r}"
            )
            continue
        if len(state_field.joint_names) != len(action_field.joint_names):
            pairs[action_key] = _Pair(
                skip_reason=(
                    "joint_names length mismatch: state has "
                    f"{len(state_field.joint_names)}, action has "
                    f"{len(action_field.joint_names)}"
                )
            )
            continue
        pairs[action_key] = _Pair(state_field=state_field, action_field=action_field)
    return pairs


def _field_map(pairs: dict[str, _Pair]) -> dict[str, str]:
    """`{topic: attribute_name}` for every state topic and every candidate
    action source topic across every non-skipped pair — the exact topic
    list this tool requests, derived from the profile, containing no camera
    topic (state/action fields never name one)."""
    field_by_topic: dict[str, str] = {}
    for pair in pairs.values():
        if pair.state_field is None or pair.action_field is None:
            continue
        field_by_topic[pair.state_field.topic] = pair.state_field.field
        for source in pair.action_field.sources:
            field_by_topic[source.topic] = source.field
    return field_by_topic


def _stream_buffers(
    path: str,
    field_by_topic: dict[str, str],
    *,
    max_samples: int,
    skip_samples: int,
    clock: str,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Streams every topic in `field_by_topic` from `path` via
    `iter_decoded_messages`, returning `{topic: [(ts, value_array), ...]}`.

    Per topic: the first `skip_samples` raw messages are discarded (clearing
    the idle prefix real recordings open with — Finding 5), then up to
    `max_samples` rows are buffered, each keyed by its resolved `clock`
    timestamp. A message whose `clock` timestamp is unavailable (an absent
    `header.stamp` under `clock="header_stamp"`) is skipped, never
    fabricated.

    Breaks out of the iterator once every requested topic has reached
    `max_samples` buffered rows, bounding wall time to a prefix of the file
    (measured: 0.4 s for a 600-sample cap over 4 topics in a 322 MB
    recording). A topic that is genuinely empty for the whole file (e.g.
    `/motion_target/target_joint_state_arm_left` in every real sample —
    DERISK-02) can never reach that cap on its own, so a fixed safety
    ceiling on total messages processed (`max_samples * topic_count * 4`)
    backstops termination in that case too (T-03-11-02).
    """
    topics = list(field_by_topic)
    buffers: dict[str, list[tuple[int, np.ndarray]]] = {t: [] for t in topics}
    if not topics:
        return buffers

    raw_seen: dict[str, int] = {t: 0 for t in topics}
    safety_limit = max_samples * len(topics) * 4
    total_processed = 0

    for topic, log_time_ns, msg in iter_decoded_messages(path, topics):
        total_processed += 1
        raw_seen[topic] += 1
        if raw_seen[topic] > skip_samples and len(buffers[topic]) < max_samples:
            ts = _resolve_ts(msg, log_time_ns, clock)
            if ts is not None:
                values = np.asarray(getattr(msg, field_by_topic[topic]), dtype=np.float64)
                buffers[topic].append((ts, values))
        if all(len(buffers[t]) >= max_samples for t in topics):
            break
        if total_processed >= safety_limit:
            break

    for buf in buffers.values():
        buf.sort(key=lambda pair: pair[0])
    return buffers


def tracking_error_profile(
    state: np.ndarray, action: np.ndarray, *, max_lag: int
) -> dict[int, float]:
    """The median `|action[t+L] - state[t]|` over the overlap, per integer
    lag `L` in `-max_lag..+max_lag`, skipping any lag whose overlap is under
    2 rows. `action` must already be aligned onto `state`'s own timestamps
    (via `align_series`) — this shifts the two ALIGNED arrays by whole rows,
    exactly `data.validation.correlation.lag_score`'s own shift convention
    (`lag >= 0` compares `state[t]` against `action[t + lag]`; `lag < 0` the
    mirror), so a lag reported here is directly comparable to a
    `CANDIDATE_PAIRINGS` `peak_lag`.

    Unlike `lag_score`, this is a plain magnitude functional with no
    variance requirement — it is well-defined even over a degenerate
    (zero-variance) channel, which is exactly why Finding 4 could measure a
    moving optimum on real data where every `CANDIDATE_PAIRINGS` correlation
    search found none.
    """
    n = min(state.shape[0], action.shape[0])
    profile: dict[int, float] = {}
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
        profile[lag] = float(np.median(np.abs(a - s)))
    return profile


def clock_skew_ns(
    path: str,
    profile: EmbodimentProfile,
    *,
    max_samples: int = 1200,
    skip_samples: int = 0,
) -> dict[str, Any]:
    """Per requested topic (every state topic and every candidate action
    source topic across every non-skipped pair), the median/p05/p95/max of
    `log_time_ns - header_stamp_ns` over up to `max_samples` samples (after
    discarding `skip_samples`), plus the count of samples whose
    `header.stamp` was absent or the all-zero sentinel.

    This is the load-bearing new number Finding 7 needs: `log_time` includes
    transport, driver buffering and recorder write latency; `header_stamp`
    does not. If the two disagree by an amount comparable to one camera
    tick, part of Finding 4's measured lag may be a recording artifact of
    this pipeline rather than the robot's own actuation lag.
    """
    pairs = _derive_pairs(profile)
    field_by_topic = _field_map(pairs)
    topics = sorted(field_by_topic)

    if not topics:
        return {"grid_period_ns": _GRID_PERIOD_NS, "topics": {}}

    raw_seen: dict[str, int] = {t: 0 for t in topics}
    skew_samples: dict[str, list[int]] = {t: [] for t in topics}
    absent_counts: dict[str, int] = {t: 0 for t in topics}
    sample_counts: dict[str, int] = {t: 0 for t in topics}
    safety_limit = max_samples * len(topics) * 4
    total_processed = 0

    try:
        for topic, log_time_ns, msg in iter_decoded_messages(path, topics):
            total_processed += 1
            raw_seen[topic] += 1
            if raw_seen[topic] > skip_samples and sample_counts[topic] < max_samples:
                sample_counts[topic] += 1
                header_ns = _header_stamp_ns(msg)
                if header_ns is None:
                    absent_counts[topic] += 1
                else:
                    skew_samples[topic].append(log_time_ns - header_ns)
            if all(sample_counts[t] >= max_samples for t in topics):
                break
            if total_processed >= safety_limit:
                break
    except GalaxeaAdapterError as e:  # noqa: BLE001 - reported not raised
        return {"error": str(e), "grid_period_ns": _GRID_PERIOD_NS, "topics": {}}

    topics_report: dict[str, Any] = {}
    for topic in topics:
        values = skew_samples[topic]
        median_ns = p05_ns = p95_ns = max_ns = None
        if values:
            arr = np.array(values, dtype=np.float64)
            median_ns = float(np.median(arr))
            p05_ns = float(np.percentile(arr, 5))
            p95_ns = float(np.percentile(arr, 95))
            max_ns = float(np.max(arr))
        topics_report[topic] = {
            "sample_count": sample_counts[topic],
            "absent_or_zero_stamp_count": absent_counts[topic],
            "median_ns": median_ns,
            "p05_ns": p05_ns,
            "p95_ns": p95_ns,
            "max_ns": max_ns,
            "median_grid_periods": median_ns / _GRID_PERIOD_NS if median_ns is not None else None,
            "p05_grid_periods": p05_ns / _GRID_PERIOD_NS if p05_ns is not None else None,
            "p95_grid_periods": p95_ns / _GRID_PERIOD_NS if p95_ns is not None else None,
            "max_grid_periods": max_ns / _GRID_PERIOD_NS if max_ns is not None else None,
        }

    return {"grid_period_ns": _GRID_PERIOD_NS, "topics": topics_report}


def measure_pairings(
    path: str,
    profile: EmbodimentProfile,
    *,
    max_samples: int = 1200,
    skip_samples: int = 0,
    decimate: int = 1,
    clock: str = "log_time",
) -> dict[str, Any]:
    """The measurement. See the module docstring and `03-11-PLAN.md`'s
    `<measured_grounding>`/Task 1 `<action>` for the full statistical
    contract. Returns `{"clock": <resolved clock name>, "action_keys":
    {action_key: <entry>}}`, where `<entry>` is either `{"skip_reason":
    str}` or the full statistics dict described in the module's own
    `<action>` text (aligned_rows, per-magnitude medians, a `pairings` dict
    keyed by `CANDIDATE_PAIRINGS`, and a `tracking_error_profile` dict).
    """
    if clock not in CLOCK_SOURCES:
        raise ValueError(f"clock must be one of {CLOCK_SOURCES!r}, got {clock!r}")
    if decimate < 1:
        raise ValueError(f"decimate must be >= 1, got {decimate}")

    pairs = _derive_pairs(profile)
    field_by_topic = _field_map(pairs)

    try:
        buffers = _stream_buffers(
            path, field_by_topic, max_samples=max_samples, skip_samples=skip_samples, clock=clock
        )
    except GalaxeaAdapterError as e:  # noqa: BLE001 - reported not raised
        return {"clock": clock, "error": str(e), "action_keys": {}}

    action_keys: dict[str, Any] = {}
    for action_key, pair in pairs.items():
        if pair.skip_reason is not None:
            action_keys[action_key] = {"skip_reason": pair.skip_reason}
            continue
        assert pair.state_field is not None and pair.action_field is not None

        state_buf = buffers.get(pair.state_field.topic, [])
        if len(state_buf) < 2:
            action_keys[action_key] = {
                "skip_reason": (
                    f"fewer than 2 buffered samples for state topic "
                    f"{pair.state_field.topic!r}"
                )
            }
            continue

        decimated = state_buf[::decimate]
        if len(decimated) < 2:
            action_keys[action_key] = {
                "skip_reason": f"decimate={decimate} leaves fewer than 2 state rows"
            }
            continue

        winning_source = None
        for source in pair.action_field.sources:
            if len(buffers.get(source.topic, [])) > 0:
                winning_source = source
                break
        if winning_source is None:
            action_keys[action_key] = {
                "skip_reason": "no populated action source within the sampled window"
            }
            continue

        action_buf = buffers[winning_source.topic]
        state_ts = np.array([t for t, _ in decimated], dtype=np.float64)
        state_vals = np.stack([v for _, v in decimated])
        action_ts = np.array([t for t, _ in action_buf], dtype=np.float64)
        action_vals = np.stack([v for _, v in action_buf])

        aligned_action = align_series(state_ts, state_vals, action_ts, action_vals)

        aligned_rows = int(state_ts.shape[0])
        duration_s = (state_ts[-1] - state_ts[0]) / 1e9 if aligned_rows > 1 else 0.0
        state_rate_hz = (aligned_rows - 1) / duration_s if duration_s > 0 else None

        state_delta = np.diff(state_vals, axis=0)
        action_trim = aligned_action[:-1]
        tracking_error_lag0 = np.abs(aligned_action - state_vals)

        pairing_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "delta_vs_action": (state_delta, action_trim),
            "delta_vs_delta": (state_delta, np.diff(aligned_action, axis=0)),
            "delta_vs_tracking_error": (state_delta, (aligned_action - state_vals)[:-1]),
        }

        pairing_results: dict[str, Any] = {}
        for name in CANDIDATE_PAIRINGS:
            s_arr, a_arr = pairing_inputs[name]
            max_lag = min(MAX_LAG_SEARCH, s_arr.shape[0] - 1) if s_arr.shape[0] > 1 else 0
            if max_lag < 1:
                pairing_results[name] = {
                    "peak_lag": None,
                    "scores": {},
                    "margin": None,
                    "reason": "too few rows for any non-zero lag to be scoreable",
                }
                continue
            peak_lag, scores = peak_lag_timesteps(s_arr, a_arr, max_lag)
            if peak_lag is None:
                pairing_results[name] = {
                    "peak_lag": None,
                    "scores": {},
                    "margin": None,
                    "reason": (
                        "no aligned channel carries variance on both sides at any "
                        "searched lag — degenerate (zero-variance) signal"
                    ),
                }
                continue
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else None
            pairing_results[name] = {
                "peak_lag": peak_lag,
                "scores": {str(k): v for k, v in scores.items()},
                "margin": margin,
                "reason": None,
            }

        te_max_lag = min(MAX_LAG_SEARCH, aligned_rows - 1) if aligned_rows > 1 else 0
        te_scores = (
            tracking_error_profile(state_vals, aligned_action, max_lag=te_max_lag)
            if te_max_lag >= 1
            else {}
        )
        if te_scores:
            argmin_lag: int | None = min(te_scores, key=lambda lag: te_scores[lag])
            sorted_te = sorted(te_scores.values())
            runner_up_ratio: float | None = (
                sorted_te[1] / sorted_te[0] if len(sorted_te) > 1 and sorted_te[0] > 0 else None
            )
        else:
            argmin_lag = None
            runner_up_ratio = None

        action_keys[action_key] = {
            "resolved_action_source": winning_source.topic,
            "aligned_rows": aligned_rows,
            "state_rate_hz": state_rate_hz,
            "median_abs_state": float(np.median(np.abs(state_vals))),
            "median_abs_action": float(np.median(np.abs(aligned_action))),
            "median_abs_state_delta": (
                float(np.median(np.abs(state_delta))) if state_delta.size else None
            ),
            "median_abs_tracking_error": float(np.median(np.abs(tracking_error_lag0))),
            "pairings": pairing_results,
            "tracking_error_profile": {
                "scores": {str(k): v for k, v in te_scores.items()},
                "argmin_lag": argmin_lag,
                "runner_up_ratio": runner_up_ratio,
            },
        }

    return {"clock": clock, "action_keys": action_keys}


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_action_state_pairing.py",
        description=(
            "Measures whether consistency.action_state_lag's action/state "
            "pairing holds on real Galaxea recordings (Defect B of "
            "D-GAP-01). Read-only diagnostic. Imports neither cv2 nor av. "
            "Writes nothing but stdout — standing tooling (D-05)."
        ),
    )
    parser.add_argument("mcap_path", nargs="+", metavar="mcap-path")
    parser.add_argument(
        "--profile",
        default="data/profiles/galaxea_r1_pro.yaml",
        help="Embodiment profile YAML path (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1200,
        help="Per-topic buffered-sample cap after --skip-samples (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Per-topic leading raw messages discarded before buffering (default: %(default)s).",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Take every n-th state sample (default: %(default)s, i.e. no decimation).",
    )
    parser.add_argument(
        "--clock",
        choices=list(CLOCK_SOURCES),
        default="log_time",
        help="Timestamp source driving ordering, decimation and alignment (default: %(default)s).",
    )
    parser.add_argument(
        "--clock-skew",
        action="store_true",
        help=(
            "Report clock_skew_ns for the requested topics and exit without "
            "running the pairing measurement."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON array of per-file objects instead of a human-readable table.",
    )
    return parser


def _print_table(results: list[dict[str, Any]], *, clock_skew: bool) -> None:
    for obj in results:
        print(f"# {obj['path']}")
        if obj.get("error") is not None:
            print(f"  error: {obj['error']}")
            continue
        if clock_skew:
            for topic, row in obj["topics"].items():
                print(
                    f"  {topic:<55} n={row['sample_count']} "
                    f"absent={row['absent_or_zero_stamp_count']} "
                    f"median_ns={row['median_ns']} "
                    f"median_grid_periods={row['median_grid_periods']}"
                )
            continue
        print(f"  clock={obj['clock']}")
        for action_key, entry in obj["action_keys"].items():
            if "skip_reason" in entry:
                print(f"  [{action_key}] skipped: {entry['skip_reason']}")
                continue
            print(
                f"  [{action_key}] aligned_rows={entry['aligned_rows']} "
                f"median_abs_state={entry['median_abs_state']:.6f} "
                f"median_abs_action={entry['median_abs_action']:.6f} "
                f"median_abs_state_delta={entry['median_abs_state_delta']}"
            )
            for name, pairing in entry["pairings"].items():
                print(f"      {name}: peak_lag={pairing['peak_lag']} margin={pairing['margin']}")
            te = entry["tracking_error_profile"]
            print(f"      tracking_error_profile: argmin_lag={te['argmin_lag']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for p in args.mcap_path:
        if not os.path.exists(p):
            print(f"error: no such file: {p}", file=sys.stderr)
            return 2

    try:
        profile = load_profile(args.profile)
    except Exception as e:  # noqa: BLE001 - reported not raised
        print(
            f"error: failed to load profile {args.profile!r}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2

    results: list[dict[str, Any]] = []
    exit_code = 0
    for path in args.mcap_path:
        try:
            if args.clock_skew:
                report = clock_skew_ns(
                    path, profile, max_samples=args.max_samples, skip_samples=args.skip_samples
                )
            else:
                report = measure_pairings(
                    path,
                    profile,
                    max_samples=args.max_samples,
                    skip_samples=args.skip_samples,
                    decimate=args.decimate,
                    clock=args.clock,
                )
        except Exception as e:  # noqa: BLE001 - reported not raised
            print(f"error: {path}: {type(e).__name__}: {e}", file=sys.stderr)
            exit_code = 1
            continue
        if report.get("error") is not None:
            exit_code = 1
        results.append({"path": path, **report})

    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, clock_skew=args.clock_skew)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
