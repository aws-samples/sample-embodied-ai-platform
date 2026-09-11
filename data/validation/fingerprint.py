"""REGR-01/D-15/D-16/D-17: golden-fingerprint regression machinery, with the
gitignored `data/goldens/baseline.json` swap seam.

Phase 3 builds the fingerprint and compare MACHINERY only; the golden
artifacts themselves and their public hosting are a later phase. Nothing
binary or dataset-shaped is committed by this module — `data/goldens/
baseline.json` (the real local pointer) is gitignored, and only
`data/goldens/baseline.example.json` (a placeholder) is committed.

**The digest algorithm.** `hashlib.sha256` exclusively — the project's one
approved content-identity digest, matching
`data/provenance/recorder.py::_hash_file_sha256`, `data/contracts/profile.py
::profile_hash`, and `data/adapters/checks.py::compute_payload_digest`.
Never a second algorithm, and never a perceptual hash: perceptual hashing
was rejected for the spot-decode check (`data/validation/frame_integrity.py`
::check_spot_decode_content, D-11) for being too insensitive to a real
difference, and exact-fingerprint regression detection needs MORE
sensitivity, not less.

**The swap seam (D-17).** `data/goldens/baseline.json` records the golden
prefix path, the tuple of source-recording content hashes it was built
from, and the pipeline/profile versions it was built under. It does NOT
separately record the source recordings' paths — those are discovered from
the golden prefix's OWN provenance records (`meta/data_engine/
episode_*.json`'s `source_uri` field), so there is exactly one place a
source recording's identity is recorded, never two copies that could drift.
`load_baseline_config` re-hashes those source recordings with
`_hash_file_sha256` (the provenance recorder's own bounded-chunk digest) and
refuses with a distinct named reason if the file, the prefix, or a matching
hash is absent — rejecting a bare CLI argument or convention-only trust was
deliberate: a golden built from a different recording or under an older
profile is otherwise indistinguishable from a valid one, which is exactly
the silent-drift class this harness exists to catch.

**Where the report says what it was compared against.** `validate_dataset`
(`data/validation/validate.py`) already threads its keyword-only `baseline`
parameter into the `ValidationContext` this check reads (plan 03-01);
`check_golden_fingerprint` puts the RESOLVED baseline identity (whichever
of `ctx.baseline` or the local `data/goldens/baseline.json` supplied it)
into its own `CheckResult.evidence` under `baseline_used`, which
`write_validation_report` already serializes verbatim into
`validation_report.json` — so the report itself says what it was compared
against without any change to the report-writing code path.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data.contracts.containers import CheckResult
from data.contracts.profile import EmbodimentProfile
from data.provenance.recorder import _hash_file_sha256
from data.validation.context import BaselineConfig, ValidationContext
from data.validation.diff import _first_provenance_field
from data.validation.frame_integrity import _decode_frames_at, _decoded_frame_count
from data.validation.registry import SKIP_REASON_KEY, UNEVALUATED_KEY

CHECK_NAME = "regression.golden_fingerprint"

BASELINE_FILENAME = "baseline.json"

DEFAULT_BASELINE_PATH = Path("data") / "goldens" / BASELINE_FILENAME
"""The local gitignored baseline pointer `check_golden_fingerprint` resolves
when `ctx.baseline` is `None` — a module-level attribute (not a default
argument value) so a test can `monkeypatch.setattr` it to a `tmp_path`
location without touching the real repo-relative file."""

FINGERPRINT_FRAME_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
"""The sampled-frame index policy: mapped to `floor(f * (row_count - 1))`
and deduplicated ascending — the identical policy
`data/validation/frame_integrity.py`'s `SPOT_CHECK_FRACTIONS` uses, so a
reader comparing this fingerprint report against a spot-decode report is
looking at the same frames."""

NUMERIC_COLUMN_NAMES: tuple[str, ...] = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
"""Every numeric parquet column `fingerprint_prefix` digests, matching
`data/writers/lerobot_v21.py::_build_table`'s declared schema exactly."""


def hash_frame_pixels(rgb: np.ndarray) -> str:
    """The one content-identity digest for a decoded RGB frame:
    `hashlib.sha256` over a contiguous `uint8` buffer — matching
    `data/validation/frame_integrity.py::_digest_decoded_frames`'s existing
    convention, never a second algorithm and never a perceptual hash."""
    return hashlib.sha256(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def hash_numeric_column(values: np.ndarray, dtype: Any) -> str:
    """`hashlib.sha256` over the canonicalized array
    `np.ascontiguousarray(values, dtype=dtype).tobytes()` — folding the
    column's own declared dtype into the canonicalization so a dtype change
    (e.g. a future float32-to-float64 regression) changes the digest rather
    than being absorbed by an implicit cast."""
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _numpy_dtype_for_arrow_type(arrow_type: Any) -> np.dtype:
    value_type = arrow_type.value_type if pa.types.is_fixed_size_list(arrow_type) else arrow_type
    if pa.types.is_floating(value_type):
        return np.dtype(np.float32) if value_type.bit_width == 32 else np.dtype(np.float64)
    if pa.types.is_integer(value_type):
        return np.dtype(np.int64) if value_type.bit_width == 64 else np.dtype(np.int32)
    raise TypeError(f"unsupported arrow type for fingerprinting: {arrow_type!r}")


def _read_numeric_column(table: Any, column_name: str) -> tuple[np.ndarray, np.dtype]:
    field_type = table.schema.field(column_name).type
    dtype = _numpy_dtype_for_arrow_type(field_type)
    if pa.types.is_fixed_size_list(field_type):
        width = field_type.list_size
        rows = table.column(column_name).to_pylist()
        if not rows:
            return np.zeros((0, width), dtype=dtype), dtype
        return np.asarray(rows, dtype=dtype).reshape(len(rows), width), dtype
    values = table.column(column_name).to_pylist()
    return np.asarray(values, dtype=dtype), dtype


def fingerprint_prefix(prefix: str | Path, profile: EmbodimentProfile) -> dict[str, Any]:
    """Produces a deterministic fingerprint document over `prefix` (an
    already-written LeRobot v2.1 dataset directory): for each episode in
    ascending index order, per emitted camera the sha256 hex digest of
    each sampled decoded frame's pixels (at the fixed fractional indices
    `FINGERPRINT_FRAME_FRACTIONS`, deduplicated ascending — reusing
    `data/validation/frame_integrity.py`'s own sequential MP4 decode
    helpers rather than opening the container in a second style), and per
    numeric parquet column the sha256 hex digest of the canonicalized
    array at that column's own declared dtype.

    Two `fingerprint_prefix` calls over the SAME unmodified prefix produce
    byte-identical documents when serialized with `json.dump(...,
    sort_keys=True)` (the caller's responsibility) — every episode is
    visited in ascending index order and every per-episode dict entry
    is written directly, never re-ordered later.
    """
    ctx = ValidationContext(out_dir=Path(prefix), profile=profile)
    episodes_doc: list[dict[str, Any]] = []

    for episode_index in ctx.episode_indices:
        parquet_path = ctx.parquet_path(episode_index)
        if not parquet_path.exists():
            continue  # a zero-frame episode wrote no parquet — nothing to fingerprint

        table = pq.read_table(parquet_path)

        cameras_doc: dict[str, Any] = {}
        for video_key in sorted(ctx.emitted_video_keys):
            video_path = ctx.video_path(episode_index, video_key)
            if not video_path.exists():
                cameras_doc[video_key] = {"error": f"video file absent: {video_path}"}
                continue
            decoded_frame_count = _decoded_frame_count(video_path)
            if decoded_frame_count == 0:
                cameras_doc[video_key] = {"decoded_frame_count": 0, "sampled_frame_hashes": {}}
                continue
            spot_indices = sorted(
                {
                    int((f * (decoded_frame_count - 1)) // 1)
                    for f in FINGERPRINT_FRAME_FRACTIONS
                }
            )
            decoded_frames = _decode_frames_at(video_path, set(spot_indices))
            cameras_doc[video_key] = {
                "decoded_frame_count": decoded_frame_count,
                "sampled_frame_hashes": {
                    str(idx): hash_frame_pixels(decoded_frames[idx]) for idx in spot_indices
                },
            }

        numeric_columns_doc: dict[str, str] = {}
        for column_name in NUMERIC_COLUMN_NAMES:
            if column_name not in table.schema.names:
                continue
            values, dtype = _read_numeric_column(table, column_name)
            numeric_columns_doc[column_name] = hash_numeric_column(values, dtype)

        episodes_doc.append(
            {
                "episode_index": episode_index,
                "cameras": cameras_doc,
                "numeric_columns": numeric_columns_doc,
            }
        )

    return {"episodes": episodes_doc}


def _golden_source_uris(golden_prefix: Path) -> list[str]:
    """Every `source_uri` recorded across `golden_prefix`'s own provenance
    records, in ascending episode-index order — discovered from the
    golden's own on-disk data, never a second, independently-maintained
    path list in `baseline.json` that could drift from it."""
    data_engine_dir = golden_prefix / "meta" / "data_engine"
    if not data_engine_dir.is_dir():
        return []
    indexed_uris: list[tuple[int, str]] = []
    for path in sorted(data_engine_dir.glob("episode_*.json")):
        try:
            episode_index = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        source_uri = payload.get("source_uri")
        if source_uri:
            indexed_uris.append((episode_index, source_uri))
    indexed_uris.sort(key=lambda pair: pair[0])
    return [uri for _, uri in indexed_uris]


def load_baseline_config(path: str | Path) -> tuple[BaselineConfig | None, str | None]:
    """Reads a `baseline.json`-shaped document and returns
    `(BaselineConfig, None)` on success, or `(None, reason)` with a
    DISTINCT named reason for each of FOUR cases: the file itself is
    absent, the named golden prefix does not exist, the golden prefix
    carries zero provenance-recorded source recordings, or re-hashing the
    golden prefix's own recorded source recordings does not reproduce the
    hashes `baseline.json` carries. The hash re-verification is the whole
    point of recording them: a baseline built from a different recording or
    under an older profile is otherwise indistinguishable from a valid one.

    The third case (zero provenance-recorded source recordings) is NOT
    merely redundant with the fourth (hash mismatch), even though both
    ultimately reject an unbacked golden — an empty `source_uris` list
    makes `missing_uris == []` and `actual_source_sha256 == ()`. If a
    hand-authored `baseline.json` also carries `"source_sha256": []`
    (trivial to do by mistake, or unavoidable if the golden was built
    without a `ProvenanceRecorder`), the empty tuple compares equal to the
    empty tuple and the hash-mismatch guard below evaluates `False` —
    reporting SUCCESS for a golden with no verified provenance backing at
    all. The bug this case closes is that an empty comparison is a
    vacuous one, not that the comparison itself is wrong; rejecting the
    empty-`source_uris` case explicitly, before the hash comparison ever
    runs, is what makes a golden with nothing to verify against never
    resolve as valid regardless of what `source_sha256` the file claims.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        return None, f"no baseline config found at {resolved_path}"

    with open(resolved_path, encoding="utf-8") as f:
        payload = json.load(f)

    golden_prefix = Path(payload["golden_prefix"])
    recorded_source_sha256 = tuple(payload["source_sha256"])
    pipeline_version = payload["pipeline_version"]
    profile_hash_value = payload["profile_hash"]

    if not golden_prefix.exists():
        return None, f"golden prefix does not exist: {golden_prefix}"

    source_uris = _golden_source_uris(golden_prefix)
    if not source_uris:
        return None, (
            f"golden prefix has no provenance-recorded source recordings: {golden_prefix}"
        )

    missing_uris = [uri for uri in source_uris if not Path(uri).exists()]
    actual_source_sha256 = tuple(
        _hash_file_sha256(uri) for uri in source_uris if Path(uri).exists()
    )

    if missing_uris or actual_source_sha256 != recorded_source_sha256:
        return None, (
            f"source recording hash mismatch for baseline at {resolved_path}: "
            f"recorded={recorded_source_sha256!r} actual={actual_source_sha256!r} "
            f"missing={missing_uris!r}"
        )

    return (
        BaselineConfig(
            golden_prefix=golden_prefix,
            source_sha256=recorded_source_sha256,
            pipeline_version=pipeline_version,
            profile_hash=profile_hash_value,
        ),
        None,
    )


def _compare_fingerprints(
    actual_doc: dict[str, Any], golden_doc: dict[str, Any]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    actual_by_index = {ep["episode_index"]: ep for ep in actual_doc["episodes"]}
    golden_by_index = {ep["episode_index"]: ep for ep in golden_doc["episodes"]}

    actual_indices = set(actual_by_index)
    golden_indices = set(golden_by_index)
    if actual_indices != golden_indices:
        mismatches.append(
            {
                "kind": "episode_set_mismatch",
                "actual_episode_indices": sorted(actual_indices),
                "golden_episode_indices": sorted(golden_indices),
            }
        )

    for episode_index in sorted(actual_indices & golden_indices):
        actual_ep = actual_by_index[episode_index]
        golden_ep = golden_by_index[episode_index]

        camera_keys = set(actual_ep["cameras"]) | set(golden_ep["cameras"])
        for video_key in sorted(camera_keys):
            actual_hashes = actual_ep["cameras"].get(video_key, {}).get(
                "sampled_frame_hashes", {}
            )
            golden_hashes = golden_ep["cameras"].get(video_key, {}).get(
                "sampled_frame_hashes", {}
            )
            frame_keys = set(actual_hashes) | set(golden_hashes)
            for frame_key in sorted(frame_keys, key=int):
                actual_digest = actual_hashes.get(frame_key)
                golden_digest = golden_hashes.get(frame_key)
                if actual_digest != golden_digest:
                    mismatches.append(
                        {
                            "kind": "frame_pixel_mismatch",
                            "episode_index": episode_index,
                            "video_key": video_key,
                            "frame_index": int(frame_key),
                            "actual_digest": actual_digest,
                            "golden_digest": golden_digest,
                        }
                    )

        column_names = set(actual_ep["numeric_columns"]) | set(golden_ep["numeric_columns"])
        for column_name in sorted(column_names):
            actual_digest = actual_ep["numeric_columns"].get(column_name)
            golden_digest = golden_ep["numeric_columns"].get(column_name)
            if actual_digest != golden_digest:
                mismatches.append(
                    {
                        "kind": "numeric_column_mismatch",
                        "episode_index": episode_index,
                        "column": column_name,
                        "actual_digest": actual_digest,
                        "golden_digest": golden_digest,
                    }
                )

    return mismatches


def golden_applies_to(
    ctx: ValidationContext, baseline: BaselineConfig
) -> tuple[bool, str | None, dict[str, Any]]:
    """The `adjacency` subject-scoping probe (D-17, layer 2 of
    03-08-PLAN.md's decision record): an opted-in golden must only be
    applied to a dataset it is genuinely about, never to one that merely
    happens to be validated while the golden is configured. Compares the
    dataset-under-validation's own provenance-recorded `profile_hash` and
    `pipeline_version` against the values `baseline` (from `baseline.json`)
    carries, reading them via `data/validation/diff.py`'s existing
    `_first_provenance_field` — imported, not reimplemented, so this
    reader can never independently drift from REGR-02's own.

    Returns `(applies, reason, subject_match)`:
    - `applies` is `True` only when BOTH fields match exactly.
    - `reason` is `None` when applicable; otherwise a string naming which
      field differs (or that provenance is absent) and both observed
      values — never silently treated as a match.
    - `subject_match` always carries both sides' `profile_hash` and
      `pipeline_version`, applicable or not, so the report says exactly
      what was compared.

    An absent provenance value — no provenance records at all (a synthetic
    fixture written without a `ProvenanceRecorder`), or a record missing the
    field — is treated as NOT matching, never as a wildcard match: a golden
    silently applying to a dataset whose subject cannot be established is
    the same class of unearned trust D-17 rejected for the golden itself.
    """
    dataset_profile_hash = _first_provenance_field(ctx.out_dir, "profile_hash")
    dataset_pipeline_version = _first_provenance_field(ctx.out_dir, "pipeline_version")

    subject_match = {
        "dataset_profile_hash": dataset_profile_hash,
        "golden_profile_hash": baseline.profile_hash,
        "dataset_pipeline_version": dataset_pipeline_version,
        "golden_pipeline_version": baseline.pipeline_version,
    }

    if dataset_profile_hash is None or dataset_pipeline_version is None:
        return (
            False,
            (
                f"dataset under validation at {ctx.out_dir} has no "
                "provenance-recorded profile_hash/pipeline_version "
                f"(profile_hash={dataset_profile_hash!r}, "
                f"pipeline_version={dataset_pipeline_version!r}) — a golden "
                "cannot apply to a dataset whose subject cannot be established"
            ),
            subject_match,
        )

    if dataset_profile_hash != baseline.profile_hash:
        return (
            False,
            (
                "golden does not apply: profile_hash differs "
                f"(dataset={dataset_profile_hash!r}, golden={baseline.profile_hash!r})"
            ),
            subject_match,
        )

    if dataset_pipeline_version != baseline.pipeline_version:
        return (
            False,
            (
                "golden does not apply: pipeline_version differs "
                f"(dataset={dataset_pipeline_version!r}, golden={baseline.pipeline_version!r})"
            ),
            subject_match,
        )

    return True, None, subject_match


NOT_REQUESTED_REASON = (
    "local golden resolution was not requested (ValidationContext.use_local_golden "
    "is False) — pass use_local_golden=True to validate_dataset to opt in to "
    "resolving the local data/goldens/baseline.json pointer. This is WINDOWS.md "
    "id 3's poisoning fix: without the opt-in, this check never touches the "
    "filesystem, so a real golden left on disk cannot affect this run's verdict."
)
"""The exact reason `check_golden_fingerprint` reports when `ctx.baseline` is
`None` and `ctx.use_local_golden` is `False` — the default, inert path. A
distinct string from every one of `load_baseline_config`'s three named
resolution-failure reasons, so a reader of `validation_report.json` can tell
"not asked" apart from "asked and failed"."""


def check_golden_fingerprint(ctx: ValidationContext) -> tuple[CheckResult, ...]:
    """One `CheckResult` for the whole dataset. Resolution and applicability
    is a four-way order, evaluated top to bottom:

    1. **Explicit `ctx.baseline`.** Honoured as-is, no opt-in required and NO
       subject-scoping applied — a caller who constructed a `BaselineConfig`
       directly has already been explicit about what it names (D-17's
       subject-scoping rule, layer 2 of 03-08-PLAN.md's decision record,
       gates only the ambient local-pointer fallback below, per its own
       "an opted-in run additionally compares" framing).
    2. **`ctx.baseline is None` and `ctx.use_local_golden` is `False`**
       (the default). Returns `UNEVALUATED_KEY` true with
       `NOT_REQUESTED_REASON` and `golden_opt_in` false in evidence. Does
       NOT touch the filesystem at all on this path — not even an
       `exists()` probe on `DEFAULT_BASELINE_PATH` — so the verdict cannot
       depend on what happens to be on disk. This is WINDOWS.md id 3's fix:
       a real `data/goldens/baseline.json` left on disk from a prior
       developer workflow session cannot poison an unrelated run that never
       opted in.
    3. **`ctx.baseline is None`, `ctx.use_local_golden` is `True`, resolved
       baseline is not applicable to this dataset.** Resolves
       `DEFAULT_BASELINE_PATH` through `load_baseline_config` exactly as
       before, preserving its three distinct named failure reasons and the
       never-a-pass rule. Once a baseline resolves, `golden_applies_to` runs
       BEFORE any fingerprinting work — fingerprinting decodes video and
       hashes every numeric column, so a dataset the golden does not apply
       to never pays that cost. On either a resolution failure or a
       non-applicable subject, returns `UNEVALUATED_KEY` true with the named
       reason, `golden_opt_in` true, and (when a baseline did resolve) a
       `subject_match` evidence key naming both sides' `profile_hash`/
       `pipeline_version`.
    4. **`ctx.baseline is None`, `ctx.use_local_golden` is `True`, resolved
       baseline applies.** Fingerprints both the prefix under validation and
       the golden prefix and compares digest by digest, failing on any
       difference and naming the artifact kind, the camera and frame index
       or the column name, and both hex digests.
    """
    try:
        baseline: BaselineConfig | None
        if ctx.baseline is not None:
            baseline = ctx.baseline
            baseline_source = "ValidationContext.baseline"
        elif not ctx.use_local_golden:
            return (
                CheckResult(
                    CHECK_NAME,
                    "fatal",
                    False,
                    {
                        UNEVALUATED_KEY: True,
                        SKIP_REASON_KEY: NOT_REQUESTED_REASON,
                        "golden_opt_in": False,
                    },
                ),
            )
        else:
            baseline, reason = load_baseline_config(DEFAULT_BASELINE_PATH)
            baseline_source = str(DEFAULT_BASELINE_PATH)
            if baseline is None:
                return (
                    CheckResult(
                        CHECK_NAME,
                        "fatal",
                        False,
                        {
                            UNEVALUATED_KEY: True,
                            SKIP_REASON_KEY: reason,
                            "baseline_source": baseline_source,
                            "golden_opt_in": True,
                        },
                    ),
                )

            applies, non_applicable_reason, subject_match = golden_applies_to(ctx, baseline)
            if not applies:
                return (
                    CheckResult(
                        CHECK_NAME,
                        "fatal",
                        False,
                        {
                            UNEVALUATED_KEY: True,
                            SKIP_REASON_KEY: non_applicable_reason,
                            "baseline_source": baseline_source,
                            "golden_opt_in": True,
                            "subject_match": subject_match,
                        },
                    ),
                )

        actual_doc = fingerprint_prefix(ctx.out_dir, ctx.profile)
        golden_doc = fingerprint_prefix(baseline.golden_prefix, ctx.profile)
        mismatches = _compare_fingerprints(actual_doc, golden_doc)

        return (
            CheckResult(
                CHECK_NAME,
                "fatal",
                not mismatches,
                {
                    "mismatches": mismatches,
                    "baseline_source": baseline_source,
                    "baseline_used": {
                        "golden_prefix": str(baseline.golden_prefix),
                        "pipeline_version": baseline.pipeline_version,
                        "profile_hash": baseline.profile_hash,
                    },
                    "golden_opt_in": ctx.use_local_golden,
                },
            ),
        )
    except Exception as e:  # noqa: BLE001 - reported not raised
        return (CheckResult(CHECK_NAME, "fatal", False, {"error": f"{type(e).__name__}: {e}"}),)
