#!/usr/bin/env python3
"""Read-only MCAP preflight diagnostic CLI.

This is standing preflight tooling (D-05), not a one-off script — re-run it
against every new Galaxea recording batch to confirm the schema-encoding,
DoF-layout, and action-source assumptions this data engine was built against
still hold. It writes nothing to disk (stdout/JSON only) and it deliberately
imports neither `cv2` nor `av`: 01-RESEARCH.md's Common Pitfall #3 measured a
macOS-only Objective-C dylib symbol collision when those two libraries are
imported into the same process, and this tool has no reason to decode a
single pixel to answer any of its three questions. Keeping both out means
every command below runs identically on the macOS dev machine and the Linux
GPU workstation.

Subcommands:
    schemas   Full sequential scan of every Schema record — answers "is this
               file's schema.encoding something mcap-ros2-support can decode?"
    topics    Per-topic message count, DoF (`len(position)`), and raw
               `msg.name` reality — answers "what does this recording's state
               layout actually look like?"
    compare   Time-aligned numeric overlay of two topic/field pairs — answers
               "do these two candidate action sources agree?"

Every decode path is wrapped in its own exception boundary and reports a
named error rather than crashing: this tool exists specifically to run
against recordings that have not been validated yet, so a malformed or
truncated file is its normal case, not an exceptional one.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from mcap.records import Channel, Header, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2.reader import read_ros2_messages

_JOINT_NAME_NOTE = (
    "per-joint identity comes from the embodiment profile's declared "
    "joint_names, not from this report; msg.name and len(position) below are "
    "reported for diagnosis only, never as ground truth"
)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


@dataclass
class _ScanResult:
    """The three record kinds a full sequential MCAP scan needs to answer
    every question `schemas` and `topics` ask."""

    schemas: list[Schema] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    header: Header | None = None


def _scan_mcap_records(path: str) -> _ScanResult:
    """Full sequential scan of `path` — not the footer summary index.

    Uses `mcap.stream_reader.StreamReader`, which decompresses every chunk
    and yields every record actually present in the file, so a Schema record
    living inside a compressed chunk is still found (DERISK-01).

    Raises:
        Exception: whatever the underlying `mcap` reader raises on a file
            that is not a valid MCAP (bad magic, truncated stream, etc.).
            Callers are expected to catch this and report the file name
            rather than let a traceback escape.
    """
    schemas: list[Schema] = []
    channels: list[Channel] = []
    header: Header | None = None
    with open(path, "rb") as f:
        for record in StreamReader(f, skip_magic=False, validate_crcs=False).records:
            if isinstance(record, Schema):
                schemas.append(record)
            elif isinstance(record, Channel):
                channels.append(record)
            elif isinstance(record, Header):
                header = record
    return _ScanResult(schemas=schemas, channels=channels, header=header)


def enumerate_schemas(path: str) -> dict[int, tuple[str, str]]:
    """Full sequential scan of every Schema record in `path`.

    Args:
        path: Path to an MCAP file.

    Returns:
        `{schema_id: (name, encoding)}` — one entry per distinct schema id
        seen in the file.
    """
    scan = _scan_mcap_records(path)
    return {s.id: (s.name, s.encoding) for s in scan.schemas}


def _require_paths_exist(paths: Iterable[str]) -> int | None:
    """Returns exit code 2 (and prints the offending path to stderr) for the
    first missing/unreadable path, or None if every path exists."""
    import os

    for p in paths:
        if not os.path.exists(p):
            print(f"error: no such file: {p}", file=sys.stderr)
            return 2
    return None


def cmd_schemas(paths: list[str], as_json: bool) -> int:
    """Implements the `schemas` subcommand. See module docstring."""
    missing = _require_paths_exist(paths)
    if missing is not None:
        return missing

    exit_code = 0
    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            scan = _scan_mcap_records(path)
        except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
            print(
                f"error: {path}: not a valid MCAP ({type(e).__name__}: {e})",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        dedup: dict[tuple[int, str], str] = {}
        for s in scan.schemas:
            dedup[(s.id, s.name)] = s.encoding
        unsorted_rows: list[dict[str, Any]] = [
            {"id": schema_id, "name": name, "encoding": encoding}
            for (schema_id, name), encoding in dedup.items()
        ]
        rows = sorted(unsorted_rows, key=lambda r: int(r["id"]))
        message_encodings = sorted({c.message_encoding for c in scan.channels})
        writer_library = scan.header.library if scan.header is not None else None

        obj = {
            "path": path,
            "schema_records_scanned": len(scan.schemas),
            "unique_schemas": len(rows),
            "schemas": rows,
            "message_encodings": message_encodings,
            "writer_library": writer_library,
        }
        results.append(obj)

        if obj["unique_schemas"] == 0:
            print(f"error: {path}: zero unique schemas found in a valid MCAP file", file=sys.stderr)
            exit_code = 1

        for row in rows:
            if row["encoding"] != "ros2msg":
                print(
                    f"error: {path}: schema {row['name']!r} (id={row['id']}) has "
                    f"encoding {row['encoding']!r}, not 'ros2msg' — "
                    "mcap-ros2-support cannot decode this schema; the rosbags "
                    "fallback is the contingency path for this case",
                    file=sys.stderr,
                )
                exit_code = 1

    if as_json:
        print(json.dumps(results))
    else:
        _print_schemas_table(results)
    return exit_code


def _print_schemas_table(results: list[dict[str, Any]]) -> None:
    for obj in results:
        print(f"# {obj['path']}")
        print(
            f"  schema_records_scanned={obj['schema_records_scanned']}  "
            f"unique_schemas={obj['unique_schemas']}  "
            f"writer_library={obj['writer_library']!r}"
        )
        print(f"  message_encodings={obj['message_encodings']}")
        for row in obj["schemas"]:
            print(f"    id={row['id']:>3}  encoding={row['encoding']:<10}  name={row['name']}")


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------


def _default_topic_reader(path: str, topic: str) -> Iterator[Any]:
    return read_ros2_messages(path, topics=[topic], log_time_order=True)


def measure_topic(
    path: str,
    topic: str,
    *,
    reader: Callable[[str, str], Iterable[Any]] = _default_topic_reader,
) -> dict[str, Any]:
    """Measure one topic's message count, DoF, and joint-name reality.

    `reader` is injectable so a raising iterator can be exercised in a test
    with no real MCAP file — see `data/tests/test_inspect_mcap.py`. It
    defaults to `mcap_ros2.reader.read_ros2_messages` restricted to `topic`.

    Args:
        path: Path to an MCAP file.
        topic: The topic to measure.
        reader: `(path, topic) -> Iterable[McapROS2Message]`. Defaults to the
            real MCAP decoder.

    Returns:
        A dict with keys `topic`, `message_count`, `min_len_position`,
        `max_len_position`, `len_position_varies`, `name_field`,
        `schema_name`, `measured_hz`, `error`, `note`. On a decode failure,
        `error` is a non-empty string and `message_count` is `None` — never
        `0` — because a corrupt stream and a legitimately empty stream are
        different facts.
    """
    result: dict[str, Any] = {
        "topic": topic,
        "message_count": None,
        "min_len_position": None,
        "max_len_position": None,
        "len_position_varies": False,
        "name_field": None,
        "schema_name": None,
        "measured_hz": None,
        "error": None,
        "note": _JOINT_NAME_NOTE,
    }
    try:
        count = 0
        first_ts: int | None = None
        last_ts: int | None = None
        min_len: int | None = None
        max_len: int | None = None
        name_field: list[Any] | None = None
        schema_name: str | None = None

        for msg in reader(path, topic):
            count += 1
            ts = msg.log_time_ns
            ros_msg = msg.ros_msg
            if first_ts is None:
                first_ts = ts
                if hasattr(ros_msg, "name"):
                    name_field = list(ros_msg.name)
                schema_name = msg.schema.name
            last_ts = ts
            if hasattr(ros_msg, "position"):
                length = len(ros_msg.position)
                if min_len is None or max_len is None:
                    min_len, max_len = length, length
                else:
                    min_len = min(min_len, length)
                    max_len = max(max_len, length)

        result["message_count"] = count
        result["min_len_position"] = min_len
        result["max_len_position"] = max_len
        result["len_position_varies"] = min_len is not None and min_len != max_len
        result["name_field"] = name_field
        result["schema_name"] = schema_name
        if count >= 2 and first_ts is not None and last_ts is not None and last_ts > first_ts:
            duration_s = (last_ts - first_ts) / 1e9
            result["measured_hz"] = (count - 1) / duration_s
    except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
        result["error"] = f"{type(e).__name__}: {e}"
        result["message_count"] = None
        result["min_len_position"] = None
        result["max_len_position"] = None
        result["len_position_varies"] = False
        result["name_field"] = None
        result["schema_name"] = None
        result["measured_hz"] = None
    return result


def cmd_topics(paths: list[str], topics_filter: list[str] | None, as_json: bool) -> int:
    """Implements the `topics` subcommand. See module docstring."""
    missing = _require_paths_exist(paths)
    if missing is not None:
        return missing

    exit_code = 0
    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            scan = _scan_mcap_records(path)
        except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
            print(
                f"error: {path}: not a valid MCAP ({type(e).__name__}: {e})",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        available_topics = {c.topic for c in scan.channels}
        requested = list(topics_filter) if topics_filter else sorted(available_topics)

        rows: list[dict[str, Any]] = []
        for topic in requested:
            if topic not in available_topics:
                rows.append(
                    {
                        "topic": topic,
                        "status": "absent",
                        "message_count": None,
                        "min_len_position": None,
                        "max_len_position": None,
                        "len_position_varies": False,
                        "name_field": None,
                        "schema_name": None,
                        "measured_hz": None,
                        "error": None,
                        "note": _JOINT_NAME_NOTE,
                    }
                )
                continue

            row = measure_topic(path, topic)
            if row["error"] is not None:
                row["status"] = "error"
                exit_code = 1
            elif row["message_count"] == 0:
                row["status"] = "empty"
            else:
                row["status"] = "ok"
            rows.append(row)

        rows.sort(key=lambda r: r["topic"])
        results.append({"path": path, "topics": rows})

    if as_json:
        print(json.dumps(results))
    else:
        _print_topics_table(results)
    return exit_code


def _print_topics_table(results: list[dict[str, Any]]) -> None:
    for obj in results:
        print(f"# {obj['path']}")
        for row in obj["topics"]:
            print(
                f"  [{row['status']:>5}] {row['topic']:<55} "
                f"count={row['message_count']} "
                f"len_position=[{row['min_len_position']},{row['max_len_position']}] "
                f"name={row['name_field']} "
                f"schema={row['schema_name']} "
                f"hz={row['measured_hz']} "
                f"error={row['error']}"
            )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _parse_topic_field(spec: str, arg_name: str) -> tuple[str, str]:
    """Splits a `TOPIC:FIELD` spec. Raises ValueError naming `spec` if it is
    not in that form."""
    if ":" not in spec:
        raise ValueError(
            f"{arg_name} value {spec!r} is not in TOPIC:FIELD form (no colon found)"
        )
    topic, _, field_name = spec.partition(":")
    if not topic or not field_name:
        raise ValueError(
            f"{arg_name} value {spec!r} is not in TOPIC:FIELD form "
            "(topic or field is empty)"
        )
    return topic, field_name


def align_series(a_t: np.ndarray, a_v: np.ndarray, b_t: np.ndarray, b_v: np.ndarray) -> np.ndarray:
    """Linearly interpolates `b_v` (sampled at `b_t`) onto `a_t`'s timestamps.

    At an `a_t` timestamp that exactly coincides with a `b_t` sample, returns
    that source sample's value exactly — `numpy.interp`'s behavior at a
    matching x-value, with no rounding error introduced.

    Args:
        a_t: 1-D array of the target timestamps (float, e.g. nanoseconds).
        a_v: Unused for alignment itself (kept for a symmetric call signature
            with `compare_series`); present only so callers can pass the same
            four arrays to both functions.
        b_t: 1-D array of `b_v`'s sample timestamps. Must not be empty.
        b_v: `b_v`'s values — 1-D (single joint) or 2-D (`(N, joints)`).

    Returns:
        The interpolated series at `a_t`'s timestamps, same ndim as `b_v`.
    """
    del a_v  # unused — kept for signature symmetry with compare_series
    a_t = np.asarray(a_t, dtype=np.float64)
    b_t = np.asarray(b_t, dtype=np.float64)
    b_v = np.asarray(b_v, dtype=np.float64)

    if b_v.ndim == 1:
        return np.interp(a_t, b_t, b_v)

    n_joints = b_v.shape[1]
    aligned = np.empty((len(a_t), n_joints), dtype=np.float64)
    for j in range(n_joints):
        aligned[:, j] = np.interp(a_t, b_t, b_v[:, j])
    return aligned


def compare_series(
    a_t: np.ndarray, a_v: np.ndarray, b_t: np.ndarray, b_v: np.ndarray
) -> dict[str, Any]:
    """Per-joint residual/correlation statistics of aligning `b_v` onto `a_t`.

    Args:
        a_t: 1-D array of series A's timestamps.
        a_v: Series A's values — 1-D or 2-D `(N, joints)`.
        b_t: 1-D array of series B's timestamps.
        b_v: Series B's values, same joint-count convention as `a_v`.

    Returns:
        `{"pair_count": int, "joints": [{"joint_index", "mean_residual",
        "std_residual", "min_a", "max_a", "min_b", "max_b", "pearson_r"},
        ...]}`. `pearson_r` is `None` for a joint whose values are constant
        (a Pearson correlation is undefined, not zero, in that case).

    Raises:
        ValueError: if either series has zero samples, naming which one —
        never returns a `nan` correlation for an empty series, which would
        read as a number instead of an explicit refusal. Also raised if the
        two series carry a different number of joints/columns.
    """
    a_t = np.asarray(a_t, dtype=np.float64)
    a_v = np.asarray(a_v, dtype=np.float64)
    b_t = np.asarray(b_t, dtype=np.float64)
    b_v = np.asarray(b_v, dtype=np.float64)

    if a_t.shape[0] == 0:
        raise ValueError(
            "compare_series: series 'a' has zero samples — refusing to "
            "compute a correlation over an empty series"
        )
    if b_t.shape[0] == 0:
        raise ValueError(
            "compare_series: series 'b' has zero samples — refusing to "
            "compute a correlation over an empty series"
        )

    a_v2 = a_v.reshape(-1, 1) if a_v.ndim == 1 else a_v
    b_v2 = b_v.reshape(-1, 1) if b_v.ndim == 1 else b_v
    if a_v2.shape[1] != b_v2.shape[1]:
        raise ValueError(
            "compare_series: joint-count mismatch — series 'a' has "
            f"{a_v2.shape[1]} columns, series 'b' has {b_v2.shape[1]}"
        )

    aligned_b = align_series(a_t, a_v2, b_t, b_v2)
    if aligned_b.ndim == 1:
        aligned_b = aligned_b.reshape(-1, 1)

    residual = a_v2 - aligned_b
    n_joints = a_v2.shape[1]
    joints: list[dict[str, Any]] = []
    for j in range(n_joints):
        col_a = a_v2[:, j]
        col_b = aligned_b[:, j]
        col_residual = residual[:, j]
        if np.std(col_a) == 0 or np.std(col_b) == 0:
            pearson_r = None
        else:
            pearson_r = float(np.corrcoef(col_a, col_b)[0, 1])
        joints.append(
            {
                "joint_index": j,
                "mean_residual": float(np.mean(col_residual)),
                "std_residual": float(np.std(col_residual)),
                "min_a": float(np.min(col_a)),
                "max_a": float(np.max(col_a)),
                "min_b": float(np.min(col_b)),
                "max_b": float(np.max(col_b)),
                "pearson_r": pearson_r,
            }
        )

    return {"pair_count": int(a_t.shape[0]), "joints": joints}


def cmd_compare(path: str, a_spec: str, b_spec: str, as_json: bool) -> int:
    """Implements the `compare` subcommand. See module docstring."""
    try:
        a_topic, a_field = _parse_topic_field(a_spec, "--a")
        b_topic, b_field = _parse_topic_field(b_spec, "--b")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    missing = _require_paths_exist([path])
    if missing is not None:
        return missing

    a_t: list[int] = []
    a_v: list[list[float]] = []
    b_t: list[int] = []
    b_v: list[list[float]] = []
    try:
        for msg in read_ros2_messages(path, topics=[a_topic, b_topic], log_time_order=True):
            topic = msg.channel.topic
            if topic == a_topic:
                a_t.append(msg.log_time_ns)
                a_v.append(list(getattr(msg.ros_msg, a_field)))
            elif topic == b_topic:
                b_t.append(msg.log_time_ns)
                b_v.append(list(getattr(msg.ros_msg, b_field)))
    except Exception as e:  # noqa: BLE001 - deliberately broad, reported not raised
        print(f"error: {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if len(a_t) == 0:
        print(
            f"error: topic {a_topic!r} has zero messages — refusing to "
            "compute a correlation over an empty series",
            file=sys.stderr,
        )
        return 1
    if len(b_t) == 0:
        print(
            f"error: topic {b_topic!r} has zero messages — refusing to "
            "compute a correlation over an empty series",
            file=sys.stderr,
        )
        return 1

    try:
        stats = compare_series(
            np.array(a_t, dtype=np.float64),
            np.array(a_v, dtype=np.float64),
            np.array(b_t, dtype=np.float64),
            np.array(b_v, dtype=np.float64),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    output = {
        "path": path,
        "a": a_spec,
        "b": b_spec,
        "pair_count": stats["pair_count"],
        "joints": stats["joints"],
    }
    if as_json:
        print(json.dumps([output]))
    else:
        _print_compare_table(output)
    return 0


def _print_compare_table(output: dict[str, Any]) -> None:
    print(f"# {output['path']}  a={output['a']}  b={output['b']}  pairs={output['pair_count']}")
    for joint in output["joints"]:
        print(
            f"  joint {joint['joint_index']}: "
            f"mean_residual={joint['mean_residual']:.6f} "
            f"std_residual={joint['std_residual']:.6f} "
            f"a=[{joint['min_a']:.4f},{joint['max_a']:.4f}] "
            f"b=[{joint['min_b']:.4f},{joint['max_b']:.4f}] "
            f"pearson_r={joint['pearson_r']}"
        )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argparse parser with its three subcommands."""
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON array of per-file objects instead of a "
        "human-readable table.",
    )

    parser = argparse.ArgumentParser(
        prog="inspect_mcap.py",
        description=(
            "Read-only MCAP preflight diagnostic. Writes nothing. Imports "
            "neither cv2 nor av. Standing tooling — re-run against every "
            "new recording batch (D-05)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_schemas = sub.add_parser(
        "schemas",
        parents=[json_parent],
        help="Report schema.encoding for every schema record in one or more MCAP files.",
    )
    p_schemas.add_argument("paths", nargs="+", metavar="mcap-path")

    p_topics = sub.add_parser(
        "topics",
        parents=[json_parent],
        help="Report per-topic message count, DoF, and joint-name reality.",
    )
    p_topics.add_argument("paths", nargs="+", metavar="mcap-path")
    p_topics.add_argument(
        "--topic",
        dest="topics",
        action="append",
        default=None,
        metavar="TOPIC",
        help="Restrict to this topic (repeatable). Default: every channel in the file.",
    )

    p_compare = sub.add_parser(
        "compare",
        parents=[json_parent],
        help="Time-align two topic/field pairs and report per-joint residual/correlation stats.",
    )
    p_compare.add_argument("path", metavar="mcap-path")
    p_compare.add_argument("--a", required=True, metavar="TOPIC:FIELD")
    p_compare.add_argument("--b", required=True, metavar="TOPIC:FIELD")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "schemas":
        return cmd_schemas(args.paths, as_json=args.json)
    if args.command == "topics":
        return cmd_topics(args.paths, topics_filter=args.topics, as_json=args.json)
    if args.command == "compare":
        return cmd_compare(args.path, args.a, args.b, as_json=args.json)

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - parser.error() always raises SystemExit


if __name__ == "__main__":
    sys.exit(main())
