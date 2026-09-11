"""Pluggable JPEG-decoder backend seam.

Human-directed addition at the 02-01 Task 1 checkpoint, not in the original
plan: `simplejpeg` is the macOS-compatible decoder that checkpoint approved,
but production ingestion later runs in a Linux container (a later phase), where
the right decoder may differ. This module is the seam that lets a future
backend (e.g. an OpenCV-based one) be registered without editing any call site
that already resolved a `JpegDecoderBackend` by name.

Deliberately minimal: a name -> backend registry, one real backend
(`simplejpeg`, the `DEFAULT_BACKEND`), and name-based resolution. Not a plugin
framework — there is no discovery, no entry-points scanning, no priority
ordering. Adding a second real backend later means one
`register_decoder_backend(...)` call, not a change to `get_decoder_backend`.
"""

import os

import numpy as np
import simplejpeg

from data.contracts import JpegDecoderBackend

DEFAULT_BACKEND = "simplejpeg"
"""The backend name `get_decoder_backend()` resolves to when neither an
explicit name nor `BACKEND_ENV_VAR` is set."""

BACKEND_ENV_VAR = "DATA_ENGINE_JPEG_BACKEND"
"""Deploy-time knob: set this to select a non-default backend without a code
change, matching this repo's existing convention of env vars for
Batch/deploy-time runtime knobs (see CLAUDE.md's training-knob env vars)."""


class SimplejpegBackend:
    """Decodes JPEG bytes to an HxWx3 RGB uint8 array via `simplejpeg.decode_jpeg`.

    Explicit `colorspace="rgb"` is the whole point of choosing this library
    over `cv2.imdecode`: no BGR->RGB fixup is needed before
    `av.VideoFrame.from_ndarray(..., format="rgb24")`.
    """

    def decode(self, data: bytes) -> np.ndarray:
        return simplejpeg.decode_jpeg(data, colorspace="rgb")


_REGISTRY: dict[str, JpegDecoderBackend] = {
    DEFAULT_BACKEND: SimplejpegBackend(),
}


def register_decoder_backend(name: str, backend: JpegDecoderBackend) -> None:
    """Register `backend` under `name`, overwriting any existing entry.

    This is the seam a future backend (OpenCV or otherwise) uses to become
    selectable by name — no existing call site changes.
    """
    _REGISTRY[name] = backend


def get_decoder_backend(name: str | None = None) -> JpegDecoderBackend:
    """Resolve a decoder backend by name.

    Resolution order: explicit `name` argument, then the `BACKEND_ENV_VAR`
    environment variable, then `DEFAULT_BACKEND`.

    Raises:
        KeyError: if the resolved name has no registered backend.
    """
    resolved = name or os.environ.get(BACKEND_ENV_VAR) or DEFAULT_BACKEND
    try:
        return _REGISTRY[resolved]
    except KeyError:
        raise KeyError(
            f"no JPEG decoder backend registered under {resolved!r}; "
            f"registered: {sorted(_REGISTRY)}"
        ) from None
