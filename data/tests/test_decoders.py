"""Tests for the JPEG-decoder backend seam (data/decoders.py).

Human-directed addition at the 02-01 Task 1 checkpoint (not in the original
plan): production ingestion later runs in a Linux container, where the right
JPEG decoder may differ from macOS's `simplejpeg`. These tests prove the seam
is swappable via a fake backend registered by name, without installing or
implementing a second real decoder.
"""

import numpy as np
import pytest

from data.contracts import JpegDecoderBackend
from data.decoders import (
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    SimplejpegBackend,
    get_decoder_backend,
    register_decoder_backend,
)


class FakeBackend:
    """No-inheritance `JpegDecoderBackend` stub proving the seam admits a
    backend `data/decoders.py` never imported at authoring time — the same
    no-inheritance, mypy-checked-conformance convention as `protocols.py`'s
    other stub implementers."""

    def decode(self, data: bytes) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)


def test_default_backend_is_simplejpeg() -> None:
    assert isinstance(get_decoder_backend(), SimplejpegBackend)


def test_default_backend_name_resolves_to_the_same_instance() -> None:
    assert get_decoder_backend(DEFAULT_BACKEND) is get_decoder_backend()


def test_fake_backend_is_registerable_and_resolvable_by_name() -> None:
    register_decoder_backend("fake", FakeBackend())
    backend = get_decoder_backend("fake")
    assert isinstance(backend, FakeBackend)
    assert backend.decode(b"unused").shape == (1, 1, 3)


def test_fake_backend_conforms_to_protocol_under_mypy() -> None:
    # mypy checks this annotated assignment; the statement just needs to be
    # valid Python at runtime — same convention as test_contracts.py's stubs.
    conformance: JpegDecoderBackend = FakeBackend()
    assert conformance.decode(b"unused").shape == (1, 1, 3)


def test_env_var_selects_backend_when_no_name_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_decoder_backend("fake", FakeBackend())
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    assert isinstance(get_decoder_backend(), FakeBackend)


def test_explicit_name_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    register_decoder_backend("fake", FakeBackend())
    monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
    assert isinstance(get_decoder_backend(DEFAULT_BACKEND), SimplejpegBackend)


def test_unknown_backend_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_decoder_backend("no-such-backend")


def test_simplejpeg_backend_decodes_real_jpeg_bytes_to_rgb() -> None:
    import simplejpeg as _simplejpeg

    tiny_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    tiny_rgb[:, :, 0] = 255
    encoded = _simplejpeg.encode_jpeg(tiny_rgb, colorspace="rgb")

    decoded = SimplejpegBackend().decode(encoded)

    assert decoded.shape == (2, 2, 3)
    assert decoded.dtype == np.uint8
