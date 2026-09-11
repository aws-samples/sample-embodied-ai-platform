"""Asserts the installed environment matches data/requirements.txt exactly.

Parses data/requirements.txt at test time (never duplicating version strings here)
and checks each pinned distribution's installed version via importlib.metadata.
A single drifted package fails its own parametrized case, naming itself in the
failure output. Also asserts the interpreter itself is Python 3.14, so running
this suite under the wrong venv fails loudly instead of silently testing nothing.
"""

import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def _parse_pins(path: Path) -> list[tuple[str, str]]:
    """Parse a flat `name==version` requirements file into (name, version) pairs.

    Args:
        path: Path to the requirements file.

    Returns:
        One (distribution_name, pinned_version) tuple per non-blank line.
    """
    pins: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, pinned = line.partition("==")
        pins.append((name, pinned))
    return pins


PINS = _parse_pins(REQUIREMENTS_PATH)
assert PINS, (
    f"{REQUIREMENTS_PATH} parsed to zero pins — environment drift guard is not running"
)


@pytest.mark.parametrize("name,pinned", PINS, ids=[name for name, _ in PINS])
def test_installed_version_matches_pin(name: str, pinned: str) -> None:
    """Each pinned distribution's installed version must match requirements.txt exactly."""
    actual = installed_version(name)
    assert actual == pinned, (
        f"{name} is installed at {actual!r} but data/requirements.txt pins {pinned!r}"
    )


def test_python_version_is_3_14() -> None:
    """Running under a different interpreter must fail loudly, not silently."""
    assert sys.version_info[:2] == (3, 14), (
        f"Expected Python 3.14, but running under {sys.version_info[0]}.{sys.version_info[1]}"
    )


def test_simplejpeg_is_importable_and_pin_matches() -> None:
    """`simplejpeg` is the default JPEG-decoder backend (data/contracts/decoders.py).

    Import-plus-pin-parity, same shape as the parametrized case above, plus one
    extra assertion specific to this package: `decode_jpeg` must still accept a
    `colorspace` keyword. That keyword is what lets the decoder hand back RGB
    directly, removing the BGR->RGB fixup step before PyAV's encoder — a silent
    API change here would surface as wrong-colored video, not as an import error.
    """
    import inspect

    import simplejpeg

    pinned = dict(PINS)["simplejpeg"]
    assert installed_version("simplejpeg") == pinned, (
        f"simplejpeg is installed at {installed_version('simplejpeg')!r} but "
        f"data/requirements.txt pins {pinned!r}"
    )
    assert callable(simplejpeg.decode_jpeg)
    assert "colorspace" in inspect.signature(simplejpeg.decode_jpeg).parameters
