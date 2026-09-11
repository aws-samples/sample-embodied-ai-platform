"""Sample-independent tests for `data.tools.inspect_mcap`.

Every test in this module passes with no MCAP file present under
`data/samples/` — the suite is green on a fresh clone. CLI exit-code
contracts are driven as a subprocess via `sys.executable`; the pure numeric
functions (`align_series`, `compare_series`) and `measure_topic`'s error path
are imported and exercised directly, with no real MCAP file needed.
"""

import subprocess
import sys

import numpy as np
import pytest

from data.tools.inspect_mcap import align_series, compare_series, measure_topic

_INSPECT_MCAP = "data/tools/inspect_mcap.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Runs the CLI as a subprocess via `sys.executable`, from the repo root."""
    return subprocess.run(
        [sys.executable, _INSPECT_MCAP, *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# --help contracts
# ---------------------------------------------------------------------------


def test_top_level_help_exits_zero_and_lists_all_three_subcommands() -> None:
    result = _run("--help")
    assert result.returncode == 0
    for name in ("schemas", "topics", "compare"):
        assert name in result.stdout


@pytest.mark.parametrize("subcommand", ["schemas", "topics", "compare"])
def test_subcommand_help_exits_zero(subcommand: str) -> None:
    result = _run(subcommand, "--help")
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Path / argument error contracts
# ---------------------------------------------------------------------------


def test_nonexistent_path_exits_2_and_names_the_path_on_stderr() -> None:
    result = _run("schemas", "/nonexistent/x.mcap")
    assert result.returncode == 2
    assert "/nonexistent/x.mcap" in result.stderr


def test_random_bytes_file_exits_nonzero_names_file_and_has_no_traceback(tmp_path) -> None:
    bad_file = tmp_path / "not_an_mcap.bin"
    bad_file.write_bytes(bytes(range(256)) * 4)

    result = _run("schemas", str(bad_file))

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert str(bad_file) in combined
    assert "Traceback" not in combined


def test_compare_bad_spec_with_no_colon_exits_2_naming_the_argument() -> None:
    result = _run("compare", "unused.mcap", "--a", "badspec", "--b", "x:y")
    assert result.returncode == 2
    assert "badspec" in result.stderr


def test_inspect_mcap_imports_neither_cv2_nor_av() -> None:
    with open(_INSPECT_MCAP) as f:
        source = f.read()
    for forbidden in ("import cv2", "import av", "from cv2", "from av"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# measure_topic's error path (injected raising iterator — no MCAP needed)
# ---------------------------------------------------------------------------


def test_measure_topic_raising_iterator_reports_error_never_zero_count() -> None:
    def raising_reader(path: str, topic: str):
        def gen():
            raise RuntimeError("decode boom")
            yield  # pragma: no cover - unreachable; makes this a generator function

        return gen()

    row = measure_topic("unused.mcap", "/some/topic", reader=raising_reader)

    assert row["error"] is not None
    assert "decode boom" in row["error"]
    assert row["message_count"] is None  # never 0 for a decode failure


def test_measure_topic_empty_iterator_reports_zero_not_error() -> None:
    def empty_reader(path: str, topic: str):
        return iter(())

    row = measure_topic("unused.mcap", "/some/topic", reader=empty_reader)

    assert row["error"] is None
    assert row["message_count"] == 0
    assert row["min_len_position"] is None
    assert row["max_len_position"] is None


# ---------------------------------------------------------------------------
# align_series / compare_series pure-function properties
# ---------------------------------------------------------------------------


def test_align_series_returns_source_value_exactly_at_coinciding_timestamp() -> None:
    b_t = np.array([0.0, 5.0, 10.0])
    b_v = np.array([1.0, 2.0, 3.0])
    a_t = np.array([5.0])

    aligned = align_series(a_t, np.array([0.0]), b_t, b_v)

    assert aligned[0] == pytest.approx(2.0)


def test_compare_series_reports_zero_residual_at_coinciding_timestamp() -> None:
    b_t = np.array([0.0, 5.0, 10.0])
    b_v = np.array([1.0, 2.0, 3.0])
    a_t = np.array([5.0])
    a_v = np.array([2.0])  # deliberately equal to b_v's value at the coinciding timestamp

    stats = compare_series(a_t, a_v, b_t, b_v)

    assert stats["joints"][0]["mean_residual"] == 0.0
    assert stats["joints"][0]["std_residual"] == 0.0


def test_compare_series_empty_b_series_raises_instead_of_nan_correlation() -> None:
    a_t = np.array([0.0, 1.0, 2.0])
    a_v = np.array([1.0, 2.0, 3.0])
    b_t = np.array([])
    b_v = np.array([])

    with pytest.raises(ValueError):
        compare_series(a_t, a_v, b_t, b_v)


def test_compare_series_empty_a_series_raises_naming_series_a() -> None:
    a_t = np.array([])
    a_v = np.array([])
    b_t = np.array([0.0, 1.0])
    b_v = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="'a'"):
        compare_series(a_t, a_v, b_t, b_v)


def test_compare_series_repeated_calls_with_duplicate_timestamps_are_equal() -> None:
    a_t = np.array([0.0, 1.0, 1.0, 2.0])
    a_v = np.array([1.0, 2.0, 2.5, 3.0])
    b_t = np.array([0.0, 1.0, 1.0, 2.0])
    b_v = np.array([1.1, 2.1, 2.1, 3.1])

    first = compare_series(a_t, a_v, b_t, b_v)
    second = compare_series(a_t, a_v, b_t, b_v)

    assert first == second


def test_compare_series_joint_count_mismatch_raises() -> None:
    a_t = np.array([0.0, 1.0])
    a_v = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 joints
    b_t = np.array([0.0, 1.0])
    b_v = np.array([1.0, 2.0])  # 1 joint

    with pytest.raises(ValueError):
        compare_series(a_t, a_v, b_t, b_v)
