"""D-05/D-06: `python -m data.serve_dataset --root data/output` — a
stdlib-only static file server exposing a locally converted LeRobot v2.1
dataset directory at the URL shape `lerobot-dataset-visualizer` fetches, so
the (optional, non-gating) container can read a converted dataset with no
HuggingFace Hub account and no upload. Serves GET and HEAD, with identical
path-confinement and header behaviour (see "HEAD requests." below).

This tool is explicitly non-gating (see `data/tools/visualizer/README.md`):
it gates no automated check in `data/validation/`, blocks no verification
step, and adds no Python runtime dependency — it imports only the standard
library.

Route shape. Every request path is matched against `DATASET_ROUTE_PREFIX`
(`/{repo_id}/resolve/main/{path}`). `{repo_id}` is accepted and ignored —
the visualizer app supplies whatever string the user typed; there is only
one served prefix. `{repo_id}` may span multiple path segments: the
visualizer's own route is `[org]/[dataset]/[episode]` and its landing page
pushes the raw typed value with a bare `router.push(value)`, so the id it
sends is always at least two segments (`org/dataset`), never one. The match
is non-greedy so a `{path}` that itself contains a later `/resolve/main/`
substring still splits at the first occurrence. `{path}` is resolved
against `--root`.

Path confinement (implemented here — there is no prior HTTP server in this
repo to copy). The URL path is never string-concatenated into a filesystem
path: `resolve_dataset_path` percent-decodes the `{path}` segment, rejects
any request whose decoded segment is absolute or contains a
parent-directory component, joins the remainder under the root, calls
`Path.resolve()` on both the candidate and the root, and serves only when
the resolved candidate `is_relative_to` the resolved root. `Path.resolve()`
follows symlinks before that comparison, which is what closes the
symlink-escape case as well. Anything that fails this check gets a 4xx and
never opens a file.

Binding. `DEFAULT_BIND_HOST` is the loopback address, so this server cannot
expose a local dataset tree to the LAN unless the caller explicitly passes
a non-loopback `--host`.

Range support. Single-range `Range: bytes=start-end` (and the `start-` /
`-suffix` open-ended forms) returns 206 with a `Content-Range` header. Any
request with no `Range` header, or a `Range` header this shim does not
parse (including a genuine multi-range request, e.g.
`bytes=0-9,20-29`), falls back to a full 200 response — a media player
that asked for two disjoint slices still gets the whole file and can pick
out what it needs client-side; the alternative (parsing and returning a
multipart/byteranges response) is not needed for a single local developer
browsing one dataset, and D-05/D-06 makes this explicitly a non-gating
convenience tool, not a general-purpose HTTP server.

HEAD requests. `do_HEAD` shares `_resolve_target` with `do_GET` — the only
place in this module that calls `resolve_dataset_path` against the bound
dataset root — so there is exactly one path-confinement decision and a
future confinement fix cannot land for one verb and not the other. It also shares
`_send_entity_headers` with `do_GET`, so the two verbs cannot diverge on
`Content-Type`, `Content-Length`, `Accept-Ranges` or
`Access-Control-Allow-Origin` either. Three deliberate decisions:
(1) a `Range` header on HEAD is ignored — status is always 200 and
`Content-Length` always reports the *full* entity length, never a partial
one. RFC 9110 explicitly permits a server to ignore `Range`; the reason a
client sends HEAD here in the first place is to learn the full length so it
can compute the byte-range GETs it is about to issue, so answering with a
partial length would defeat the request's purpose, and a bodyless 206 would
be strictly less useful and harder to reason about than this module's
existing conservative full-200 fallback for any `Range` header it does not
act on. `Accept-Ranges: bytes` is still advertised, so the client knows the
ranged GET that follows will be honoured. (2) HEAD never opens the file —
`stat()` alone is sufficient and strictly cheaper — which means an
existing-but-unreadable file answers 200 to HEAD where GET would fail at
open time; `do_GET` has no handling for that case either, so this is an
accepted, not a new, gap. (3) 4xx responses route through the same
`send_error` calls `do_GET` uses, with identical status codes and message
strings; `BaseHTTPRequestHandler` already suppresses the response body for
HEAD while still reporting the error entity's `Content-Length`, which is
correct — that length describes the entity the equivalent GET would have
returned. `do_HEAD` was absent up to now because every prior verification of
this module used `curl` without `-I`, i.e. GET only, and a GET-only matrix
cannot detect a broken HEAD path.
"""

import argparse
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

# Documents the shape the route matcher below implements; not itself parsed
# into the regex (translating a format string into a regex is more trouble
# than it saves for a fixed three-segment shape).
DATASET_ROUTE_PREFIX = "/{repo_id}/resolve/main/{path}"

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

_ROUTE_RE = re.compile(r"^/(?P<repo_id>.+?)/resolve/main/(?P<path>.*)$")
_SINGLE_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/json",
    ".parquet": "application/octet-stream",
    ".mp4": "video/mp4",
}


def resolve_dataset_path(root: Path, url_path: str) -> Path | None:
    """Resolves a raw HTTP request path against `root`, or returns `None`
    when the request does not match `DATASET_ROUTE_PREFIX`'s shape or its
    decoded `{path}` segment would escape `root` (absolute segment, a
    parent-directory component, or a symlink that resolves outside `root`).

    Pure and side-effect-free (no socket, no exception on a malformed
    request) so the confinement decision is unit-testable on its own, per
    this task's own framing.
    """
    match = _ROUTE_RE.match(url_path)
    if match is None:
        return None

    raw_segment = match.group("path")
    try:
        decoded = urllib.parse.unquote(raw_segment, errors="strict")
    except UnicodeDecodeError:
        return None

    if decoded == "" or decoded.startswith("/"):
        return None

    segments = decoded.split("/")
    if any(segment in ("", "..") for segment in segments):
        return None

    root_resolved = root.resolve()
    candidate = root_resolved / decoded
    resolved = candidate.resolve()

    if not resolved.is_relative_to(root_resolved):
        return None

    return resolved


def _content_type_for(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _parse_single_range(header_value: str, file_size: int) -> tuple[int, int] | None:
    """Parses a `Range` header value into an inclusive `(start, end)` byte
    pair clamped to `file_size`. Returns `None` for an absent, malformed,
    unsatisfiable, or multi-range header — callers fall back to a full 200
    response in every one of those cases."""
    if "," in header_value:
        return None  # multi-range: fall back to a full response.

    match = _SINGLE_RANGE_RE.match(header_value.strip())
    if match is None:
        return None

    start_str, end_str = match.groups()
    if start_str == "" and end_str == "":
        return None

    if start_str == "":
        suffix_length = int(end_str)
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str != "" else file_size - 1

    if file_size == 0 or start >= file_size or start > end:
        return None

    end = min(end, file_size - 1)
    return start, end


class DatasetRequestHandler(BaseHTTPRequestHandler):
    """Serves exactly one dataset prefix (`dataset_root`) at
    `DATASET_ROUTE_PREFIX`. `dataset_root` is bound per-server by
    `build_handler` rather than passed per-request, since this shim only
    ever serves one dataset directory per invocation."""

    dataset_root: ClassVar[Path] = Path(".")

    def _resolve_target(self) -> Path | None:
        """Resolves this request's path against `dataset_root`, sending the
        403/404 error response itself when resolution fails. Callers must
        `return` immediately when this returns `None` — the response has
        already been sent. Shared by `do_GET` and `do_HEAD` so there is
        exactly one path-confinement decision point in this module."""
        url_path = urllib.parse.urlsplit(self.path).path
        resolved = resolve_dataset_path(self.dataset_root, url_path)
        if resolved is None:
            self.send_error(403, "Forbidden: path escapes the served root")
            return None
        if not resolved.is_file():
            self.send_error(404, "Not Found")
            return None
        return resolved

    def _send_entity_headers(
        self,
        *,
        status: int,
        content_type: str,
        content_length: int,
        content_range: str | None = None,
    ) -> None:
        """Sends the status line and the full header set shared by the 200
        response, the 206 response and the HEAD response. Header order
        matches the pre-extraction 206 branch's order, so this is a pure
        extraction."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        resolved = self._resolve_target()
        if resolved is None:
            return

        file_size = resolved.stat().st_size
        content_type = _content_type_for(resolved)
        range_header = self.headers.get("Range")
        byte_range = (
            _parse_single_range(range_header, file_size)
            if range_header is not None
            else None
        )

        with resolved.open("rb") as fh:
            if byte_range is not None:
                start, end = byte_range
                length = end - start + 1
                self._send_entity_headers(
                    status=206,
                    content_type=content_type,
                    content_length=length,
                    content_range=f"bytes {start}-{end}/{file_size}",
                )
                fh.seek(start)
                self.wfile.write(fh.read(length))
            else:
                self._send_entity_headers(
                    status=200,
                    content_type=content_type,
                    content_length=file_size,
                )
                self.wfile.write(fh.read())

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's naming convention
        resolved = self._resolve_target()
        if resolved is None:
            return

        # Range is deliberately ignored on HEAD — see the module docstring's
        # "HEAD requests." paragraph (RFC 9110 permits this; the client's
        # whole reason for sending HEAD is to learn the *full* length).
        # stat() only: never open the file for HEAD.
        self._send_entity_headers(
            status=200,
            content_type=_content_type_for(resolved),
            content_length=resolved.stat().st_size,
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # This is a local developer tool; stay quiet by default.


def build_handler(root: Path) -> type[DatasetRequestHandler]:
    """Binds `root` to a fresh `DatasetRequestHandler` subclass, since
    `HTTPServer` takes a handler *class*, not an instance."""

    class _BoundDatasetRequestHandler(DatasetRequestHandler):
        dataset_root = root.resolve()

    return _BoundDatasetRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve_dataset.py",
        description=(
            "Local, stdlib-only static file server exposing a converted "
            "LeRobot v2.1 dataset directory at the URL shape "
            f"{DATASET_ROUTE_PREFIX!r} that lerobot-dataset-visualizer "
            "fetches. Optional, non-gating tooling (D-05/D-06)."
        ),
    )
    parser.add_argument(
        "--root",
        default="data/output",
        metavar="PATH",
        help="Dataset output directory to serve. Default: data/output",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        metavar="HOST",
        help=(
            f"Bind address. Default: {DEFAULT_BIND_HOST} (loopback only). "
            "There is no other way to expose this server beyond localhost "
            "than passing an explicit non-loopback address here."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="PORT",
        help=f"Bind port. Default: {DEFAULT_PORT}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 1

    handler_cls = build_handler(root)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"Serving {root} at http://{args.host}:{server.server_port}{DATASET_ROUTE_PREFIX}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
