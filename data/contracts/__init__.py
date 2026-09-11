"""Single import point for the data engine's cross-layer contract (CONT-01).

Import every canonical container, pluggable-layer protocol, and profile symbol from
`data.contracts` — never from a submodule directly. This module's `__all__` is the
literal enumeration of what CONT-01 promises is importable from one place.
"""

from data.contracts.containers import (
    RECORD_SCHEMA_VERSION,
    CanonicalEpisode,
    CanonicalFrame,
    CheckResult,
    ClockReport,
    Disposition,
    EnrichmentPayload,
    EpisodeHeader,
    EpisodeRecord,
    EpisodeWriteResult,
    ImageRef,
    IngestOutcome,
    StreamInfo,
)
from data.contracts.profile import EmbodimentProfile, load_profile, profile_hash
from data.contracts.protocols import (
    BaseDatasetWriter,
    BaseEnricher,
    BaseFilter,
    BaseInputAdapter,
    BaseSynchronizer,
    JpegDecoderBackend,
)

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "CanonicalFrame",
    "CanonicalEpisode",
    "EpisodeHeader",
    "EnrichmentPayload",
    "ImageRef",
    "EpisodeWriteResult",
    "EpisodeRecord",
    "StreamInfo",
    "ClockReport",
    "CheckResult",
    "Disposition",
    "IngestOutcome",
    "BaseInputAdapter",
    "BaseDatasetWriter",
    "BaseSynchronizer",
    "BaseEnricher",
    "BaseFilter",
    "JpegDecoderBackend",
    "EmbodimentProfile",
    "load_profile",
    "profile_hash",
]
