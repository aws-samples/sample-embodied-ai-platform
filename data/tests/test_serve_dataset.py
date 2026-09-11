"""Tests for `data/serve_dataset.py` — path confinement, the
`{repo_id}/resolve/main/{path}` route shape, single-range support, and
HEAD/GET parity (Task 2, plan 03-02; quick task 260910-d3z).

`resolve_dataset_path` is exercised directly for the traversal/symlink
negative cases so the security decision is unit-testable without binding a
socket (per the plan's own framing). The request-level behaviours (exact
status codes, exact bytes, `Content-Range`) are exercised against a live
server bound to port 0 in a background thread.
"""

import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from data.serve_dataset import (
    DEFAULT_BIND_HOST,
    build_handler,
    build_parser,
    resolve_dataset_path,
)


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_bytes(b'{"codebase_version": "v2.1"}')
    (root / "video.mp4").write_bytes(bytes(range(50)))
    return root


@pytest.fixture()
def running_server(dataset_root: Path) -> Iterator["object"]:
    from http.server import ThreadingHTTPServer

    handler_cls = build_handler(dataset_root)
    server = ThreadingHTTPServer((DEFAULT_BIND_HOST, 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _url(server: "object", path: str) -> str:
    host, port = server.server_address[:2]  # type: ignore[attr-defined]
    return f"http://{host}:{port}{path}"


# --- resolve_dataset_path: pure confinement decision, no socket needed ---


def test_resolve_dataset_path_returns_path_under_root(dataset_root: Path) -> None:
    result = resolve_dataset_path(dataset_root, "/anything/resolve/main/meta/info.json")
    assert result == (dataset_root / "meta" / "info.json").resolve()


def test_resolve_dataset_path_ignores_repo_id(dataset_root: Path) -> None:
    a = resolve_dataset_path(dataset_root, "/repo-one/resolve/main/meta/info.json")
    b = resolve_dataset_path(dataset_root, "/an-entirely-different-repo/resolve/main/meta/info.json")
    assert a == b


def test_dotdot_traversal_rejected(tmp_path: Path) -> None:
    assert resolve_dataset_path(tmp_path, "/repo/resolve/main/../../etc/passwd") is None


def test_percent_encoded_dotdot_traversal_rejected(tmp_path: Path) -> None:
    assert (
        resolve_dataset_path(tmp_path, "/repo/resolve/main/%2e%2e/%2e%2e/etc/passwd")
        is None
    )


def test_absolute_path_segment_rejected(tmp_path: Path) -> None:
    assert resolve_dataset_path(tmp_path, "/repo/resolve/main//etc/passwd") is None


def test_symlink_escaping_root_rejected(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    (root / "escape").symlink_to(outside)

    assert resolve_dataset_path(root, "/repo/resolve/main/escape/secret.txt") is None


def test_unmatched_route_rejected(tmp_path: Path) -> None:
    assert resolve_dataset_path(tmp_path, "/not-the-right-shape") is None


# --- Task 1 (D1): repo_id must span multiple segments, non-greedy ---


def test_two_segment_repo_id_resolves(dataset_root: Path) -> None:
    result = resolve_dataset_path(
        dataset_root, "/local/dataset/resolve/main/meta/info.json"
    )
    assert result == (dataset_root / "meta" / "info.json").resolve()


def test_two_segment_and_single_segment_repo_id_resolve_identically(
    dataset_root: Path,
) -> None:
    two_segment = resolve_dataset_path(
        dataset_root, "/local/dataset/resolve/main/meta/info.json"
    )
    single_segment = resolve_dataset_path(
        dataset_root, "/single/resolve/main/meta/info.json"
    )
    assert two_segment == single_segment


def test_three_segment_repo_id_resolves(dataset_root: Path) -> None:
    result = resolve_dataset_path(
        dataset_root, "/a/b/c/resolve/main/meta/info.json"
    )
    assert result == (dataset_root / "meta" / "info.json").resolve()


def test_repo_id_binds_non_greedy_at_first_resolve_main_occurrence(
    dataset_root: Path,
) -> None:
    (dataset_root / "resolve").mkdir()
    (dataset_root / "resolve" / "main").mkdir()
    (dataset_root / "resolve" / "main" / "x.json").write_bytes(b"{}")

    result = resolve_dataset_path(
        dataset_root, "/org/ds/resolve/main/resolve/main/x.json"
    )
    assert result == (dataset_root / "resolve" / "main" / "x.json").resolve()


def test_live_two_segment_repo_id_returns_exact_bytes(
    running_server: object, dataset_root: Path
) -> None:
    with urllib.request.urlopen(
        _url(running_server, "/local/dataset/resolve/main/meta/info.json")
    ) as resp:
        assert resp.status == 200
        assert resp.read() == (dataset_root / "meta" / "info.json").read_bytes()


def test_two_segment_repo_id_dotdot_traversal_rejected(tmp_path: Path) -> None:
    assert (
        resolve_dataset_path(
            tmp_path, "/local/dataset/resolve/main/../../etc/passwd"
        )
        is None
    )


def test_two_segment_repo_id_percent_encoded_dotdot_traversal_rejected(
    tmp_path: Path,
) -> None:
    assert (
        resolve_dataset_path(
            tmp_path,
            "/local/dataset/resolve/main/%2e%2e/%2e%2e/etc/passwd",
        )
        is None
    )


# --- Test 1: metadata fetch returns exact bytes ---


def test_metadata_fetch_returns_exact_bytes(running_server: object, dataset_root: Path) -> None:
    with urllib.request.urlopen(
        _url(running_server, "/some-repo/resolve/main/meta/info.json")
    ) as resp:
        assert resp.status == 200
        assert resp.read() == (dataset_root / "meta" / "info.json").read_bytes()


# --- Test 2: traversal attempts (raw and percent-encoded) return 4xx ---


def test_live_traversal_request_returns_4xx(running_server: object) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_url(running_server, "/repo/resolve/main/../../etc/passwd"))
    assert 400 <= excinfo.value.code < 500


def test_live_percent_encoded_traversal_request_returns_4xx(running_server: object) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(
            _url(running_server, "/repo/resolve/main/%2e%2e/%2e%2e/etc/passwd")
        )
    assert 400 <= excinfo.value.code < 500


# --- Test 3: symlink escape inside root, exercised live too ---


def test_live_symlink_escape_returns_4xx(dataset_root: Path) -> None:
    from http.server import ThreadingHTTPServer

    outside = dataset_root.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    (dataset_root / "escape").symlink_to(outside)

    handler_cls = build_handler(dataset_root)
    server = ThreadingHTTPServer((DEFAULT_BIND_HOST, 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(
                _url(server, "/repo/resolve/main/escape/secret.txt")
            )
        assert 400 <= excinfo.value.code < 500
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- Test 4: missing path returns 404 ---


def test_missing_path_returns_404(running_server: object) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_url(running_server, "/repo/resolve/main/does/not/exist.json"))
    assert excinfo.value.code == 404


# --- Test 5: single Range request ---


def test_range_request_returns_partial_content(
    running_server: object, dataset_root: Path
) -> None:
    request = urllib.request.Request(
        _url(running_server, "/repo/resolve/main/video.mp4"),
        headers={"Range": "bytes=0-9"},
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 206
        body = resp.read()
        assert len(body) == 10
        assert body == (dataset_root / "video.mp4").read_bytes()[0:10]
        assert resp.headers["Content-Range"] == f"bytes 0-9/{50}"


def test_no_range_header_returns_full_body_200(
    running_server: object, dataset_root: Path
) -> None:
    with urllib.request.urlopen(_url(running_server, "/repo/resolve/main/video.mp4")) as resp:
        assert resp.status == 200
        assert resp.read() == (dataset_root / "video.mp4").read_bytes()


def test_multi_range_falls_back_to_full_body(
    running_server: object, dataset_root: Path
) -> None:
    request = urllib.request.Request(
        _url(running_server, "/repo/resolve/main/video.mp4"),
        headers={"Range": "bytes=0-9,20-29"},
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 200
        assert resp.read() == (dataset_root / "video.mp4").read_bytes()


# --- Test 6: default bind host is loopback; --host required to change it ---


def test_default_bind_host_is_loopback() -> None:
    args = build_parser().parse_args([])
    assert args.host == DEFAULT_BIND_HOST
    assert DEFAULT_BIND_HOST in ("127.0.0.1", "localhost")


def test_host_override_via_flag() -> None:
    args = build_parser().parse_args(["--host", "0.0.0.0"])
    assert args.host == "0.0.0.0"


# --- Task 1 (quick-260910-d3z): HEAD parity with GET ---


def test_head_on_real_file_returns_200_content_length_and_empty_body(
    running_server: object, dataset_root: Path
) -> None:
    request = urllib.request.Request(
        _url(running_server, "/repo/resolve/main/meta/info.json"), method="HEAD"
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 200
        real_length = (dataset_root / "meta" / "info.json").stat().st_size
        assert int(resp.headers["Content-Length"]) == real_length
        assert resp.read() == b""


def test_head_headers_match_get_headers_on_real_file(running_server: object) -> None:
    url = _url(running_server, "/repo/resolve/main/video.mp4")
    with urllib.request.urlopen(url) as get_resp:
        get_body = get_resp.read()
        get_headers = get_resp.headers

    head_request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head_request) as head_resp:
        head_body = head_resp.read()
        head_headers = head_resp.headers

    for header in (
        "Content-Type",
        "Content-Length",
        "Accept-Ranges",
        "Access-Control-Allow-Origin",
    ):
        assert head_headers[header] == get_headers[header]

    assert get_body == bytes(range(50))
    assert head_body == b""


def test_head_missing_file_returns_404(running_server: object) -> None:
    request = urllib.request.Request(
        _url(running_server, "/repo/resolve/main/does/not/exist.json"), method="HEAD"
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request)
    assert excinfo.value.code == 404


def test_head_dotdot_traversal_returns_4xx_matching_get(running_server: object) -> None:
    path = "/repo/resolve/main/../../etc/passwd"

    with pytest.raises(urllib.error.HTTPError) as get_excinfo:
        urllib.request.urlopen(_url(running_server, path))

    head_request = urllib.request.Request(_url(running_server, path), method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as head_excinfo:
        urllib.request.urlopen(head_request)

    assert head_excinfo.value.code == get_excinfo.value.code
    assert 400 <= head_excinfo.value.code < 500


def test_head_percent_encoded_dotdot_traversal_returns_4xx_matching_get(
    running_server: object,
) -> None:
    path = "/repo/resolve/main/%2e%2e/%2e%2e/etc/passwd"

    with pytest.raises(urllib.error.HTTPError) as get_excinfo:
        urllib.request.urlopen(_url(running_server, path))

    head_request = urllib.request.Request(_url(running_server, path), method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as head_excinfo:
        urllib.request.urlopen(head_request)

    assert head_excinfo.value.code == get_excinfo.value.code
    assert 400 <= head_excinfo.value.code < 500


def test_head_route_mismatch_returns_4xx_matching_get(running_server: object) -> None:
    path = "/not-the-right-shape"

    with pytest.raises(urllib.error.HTTPError) as get_excinfo:
        urllib.request.urlopen(_url(running_server, path))

    head_request = urllib.request.Request(_url(running_server, path), method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as head_excinfo:
        urllib.request.urlopen(head_request)

    assert head_excinfo.value.code == get_excinfo.value.code
    assert 400 <= head_excinfo.value.code < 500


def test_head_error_response_writes_no_body(running_server: object) -> None:
    path = "/repo/resolve/main/does/not/exist.json"

    with pytest.raises(urllib.error.HTTPError) as get_excinfo:
        urllib.request.urlopen(_url(running_server, path))
    get_content_length = get_excinfo.value.headers["Content-Length"]

    head_request = urllib.request.Request(_url(running_server, path), method="HEAD")
    with pytest.raises(urllib.error.HTTPError) as head_excinfo:
        urllib.request.urlopen(head_request)

    assert head_excinfo.value.read() == b""
    assert head_excinfo.value.headers["Content-Length"] == get_content_length


def test_head_with_range_header_reports_full_entity_length_200(
    running_server: object,
) -> None:
    request = urllib.request.Request(
        _url(running_server, "/repo/resolve/main/video.mp4"),
        headers={"Range": "bytes=0-9"},
        method="HEAD",
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 200
        assert int(resp.headers["Content-Length"]) == 50
        assert resp.headers.get("Content-Range") is None
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert resp.read() == b""
