"""`ValidationContext`/`BaselineConfig` — everything a post-write validation
check needs, resolved up front, nothing derived lazily inside a check body.

Shaped after `data/adapters/checks.py`'s `FatalCheckContext`/
`AdvisoryCheckContext` (D-01): a plain frozen dataclass, no methods that do
real work beyond cheap path arithmetic. Every path this context resolves is
built from the writer's own `_DATA_PATH_TEMPLATE`/`_VIDEO_PATH_TEMPLATE`/
`ENCODE_MANIFEST_DIRNAME` constants (`data/writers/lerobot_v21.py`) — never a
second copy of those strings, so a future template change cannot silently
desync the validator from the writer.

No `data/validation/__init__.py` exists: `data/adapters/`, `data/writers/`,
`data/sync/` and `data/provenance/` are all plain namespace directories under
this repo's `mypy.ini` (`namespace_packages = True`,
`explicit_package_bases = True`), and `data/validation/` follows them.
"""

import dataclasses
from pathlib import Path

from data.contracts.profile import EmbodimentProfile
from data.writers.lerobot_v21 import (
    _CHUNK_INDEX,
    _DATA_PATH_TEMPLATE,
    _VIDEO_PATH_TEMPLATE,
    ENCODE_MANIFEST_DIRNAME,
)


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    """D-17's four facts a baseline pointer must record for REGR-01
    (regression-against-golden-prefix, plan 03-05). Declared here so
    `validate_dataset`'s keyword-only `baseline` parameter is Phase 4's
    forward contract from this plan onward; loading, hash verification and
    the named-skip semantics are REGR-01's own logic (plan 03-05), not this
    plan's."""

    golden_prefix: Path
    source_sha256: tuple[str, ...]
    pipeline_version: str
    profile_hash: str


@dataclasses.dataclass(frozen=True)
class ValidationContext:
    """Everything a post-write validation check needs. `out_dir` is the
    written dataset prefix under validation; `profile` is the embodiment
    profile that produced it; `baseline`, when present, is REGR-01's golden
    reference (unused by any check this plan registers)."""

    out_dir: Path
    profile: EmbodimentProfile
    baseline: BaselineConfig | None = None
    use_local_golden: bool = False
    """The explicit opt-in `check_golden_fingerprint` requires before it will
    resolve the local, gitignored `data/goldens/baseline.json` pointer
    (`DEFAULT_BASELINE_PATH`) on `ctx.baseline`'s behalf. Additive, defaults
    `False`, so every existing `ValidationContext(out_dir=..., profile=...)`
    construction stays valid unchanged. This flag gates ONLY the ambient
    local-pointer fallback — an explicit `ctx.baseline` supplied by the
    caller is always honoured with no opt-in required (WINDOWS.md id 3,
    03-08-PLAN.md's decision record, layer 1)."""

    @property
    def emitted_video_keys(self) -> tuple[str, ...]:
        """Every `cameras` entry with `emit: true`, by `video_key`, in
        profile declaration order."""
        return tuple(
            cam.video_key
            for cam in self.profile.cameras.values()
            if cam.emit and cam.video_key is not None
        )

    @property
    def episode_indices(self) -> tuple[int, ...]:
        """Every episode index discovered from `meta/data_engine/
        episode_{NNNNNN}.json` provenance records — deliberately NOT from
        `data/chunk-000/episode_*.parquet`: a zero-frame episode never
        writes a parquet file, but `ProvenanceRecorder.record` still writes
        a provenance record for it (`build_baseline_dataset` always calls
        `.record()`), so scanning the provenance-record directory is what
        lets an episode-scoped check see a zero-frame episode and report it
        as `unevaluated` rather than never being invoked for it at all."""
        data_engine_dir = self.out_dir / "meta" / "data_engine"
        if not data_engine_dir.is_dir():
            return ()
        indices: list[int] = []
        for path in data_engine_dir.glob("episode_*.json"):
            try:
                indices.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return tuple(sorted(indices))

    def parquet_path(self, episode_index: int) -> Path:
        return self.out_dir / _DATA_PATH_TEMPLATE.format(
            episode_chunk=_CHUNK_INDEX, episode_index=episode_index
        )

    def video_path(self, episode_index: int, video_key: str) -> Path:
        return self.out_dir / _VIDEO_PATH_TEMPLATE.format(
            episode_chunk=_CHUNK_INDEX, video_key=video_key, episode_index=episode_index
        )

    def encode_manifest_path(self, episode_index: int) -> Path:
        return (
            self.out_dir
            / "meta"
            / ENCODE_MANIFEST_DIRNAME
            / f"episode_{episode_index:06d}.json"
        )

    def provenance_record_path(self, episode_index: int) -> Path:
        return self.out_dir / "meta" / "data_engine" / f"episode_{episode_index:06d}.json"
