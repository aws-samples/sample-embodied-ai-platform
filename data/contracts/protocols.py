"""Structural protocols for the physical-AI data engine's five pluggable layers.

Every protocol here is a `typing.Protocol` — structural typing. An implementation
satisfies a protocol purely by having matching method signatures; it never imports
or inherits from this module. `mypy`/`pyright` are what prove conformance, via an
annotated assignment (`impl: BaseInputAdapter = MyAdapter()`), not a runtime
`isinstance` check and not a base-class import.
"""

from typing import Any, Mapping, Protocol

import numpy as np

from data.contracts.containers import (
    CanonicalEpisode,
    EnrichmentPayload,
    EpisodeHeader,
    EpisodeWriteResult,
)
from data.contracts.profile import EmbodimentProfile


class BaseInputAdapter(Protocol):
    """Reads one source recording and produces one `CanonicalEpisode`."""

    def parse(self, path: str, profile: EmbodimentProfile) -> CanonicalEpisode:
        """Parse `path` into a `CanonicalEpisode` per `profile`'s topic mapping."""
        ...


class BaseDatasetWriter(Protocol):
    """Streams a `CanonicalEpisode` to an on-disk dataset, one frame at a time."""

    def write_episode(self, episode: CanonicalEpisode) -> EpisodeWriteResult:
        """Write one episode, returning a receipt (CONT-03)."""
        ...

    def finalize(self) -> None:
        """Close the dataset — flush encoders, write meta files."""
        ...


class BaseSynchronizer(Protocol):
    """Builds the master-clock grid and yields a single-pass `CanonicalFrame`
    stream from a set of raw, per-topic streams."""

    def synchronize(
        self, raw_streams: Mapping[str, Any], profile: EmbodimentProfile
    ) -> CanonicalEpisode:
        """Resample `raw_streams` onto the master clock per `profile`."""
        ...


class BaseEnricher(Protocol):
    """Produces a Layer-0 quality verdict for one episode.

    Takes an `EpisodeHeader`, deliberately not a `CanonicalEpisode`.
    `EpisodeHeader` is fully materialized and cheap, so it can be read
    repeatedly; handing this protocol the episode would invite the exact
    double-iteration bug that silently empties the frame stream (see
    `containers.py`'s `CanonicalEpisode` docstring). An implementation needing
    frame-level inspection reads pre-computed summaries from
    `EpisodeHeader.global_meta` (which the synchronizer populates during its
    single pass), or requests a fresh generator by re-parsing — it never
    assumes `episode.frames` is re-iterable.
    """

    def enrich(self, header: EpisodeHeader) -> EnrichmentPayload:
        """Score or flag one episode from its header alone."""
        ...


class BaseFilter(Protocol):
    """Decides whether to keep one episode.

    Takes an `EpisodeHeader`, deliberately not a `CanonicalEpisode`, for the
    exact same reason as `BaseEnricher` above: an implementation needing
    frame-level inspection reads pre-computed summaries from
    `EpisodeHeader.global_meta` (which the synchronizer populates during its
    single pass), or requests a fresh generator by re-parsing — it never
    assumes `episode.frames` is re-iterable.
    """

    def keep(self, header: EpisodeHeader) -> bool:
        """Return True to keep the episode, False to drop it."""
        ...


class JpegDecoderBackend(Protocol):
    """Decodes compressed JPEG bytes to an HxWx3 uint8 RGB array.

    The pluggable seam behind `ImageRef.decode()` for the one specific
    compressed format `sensor_msgs/CompressedImage` carries. Selected by name
    via `data.decoders.get_decoder_backend`, never by import — this is what
    lets a future backend (e.g. an OpenCV-based one, for a Linux deployment
    target this project has not yet built) register itself under a new name
    without any call site changing. `data/decoders.py` ships the one real
    backend (`simplejpeg`, the default) plus the by-name registry; this
    module declares only the shape a backend must satisfy.
    """

    def decode(self, data: bytes) -> np.ndarray:
        """Decode `data` (raw JPEG bytes) to an HxWx3 uint8 RGB array."""
        ...
