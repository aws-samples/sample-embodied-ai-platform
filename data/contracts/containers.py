"""Canonical frozen containers for the physical-AI data engine's cross-layer contract.

Every container below is a frozen, slotted dataclass: `frozen=True` blocks
`__setattr__`, so a field's *binding* is set once at construction and can never be
reassigned (`frame.state = {...}` raises `FrozenInstanceError`). That guarantee is
shallow, not deep: it says nothing about mutating an already-referenced mutable
object reachable through a field. To close the gap that matters most in practice —
one component reassigning a key on a dict another component still holds a reference
to — every dict-shaped field below (`CanonicalFrame.images`/`.state`/`.action`/
`.meta`, `EpisodeHeader.global_meta`, `CheckResult.evidence`, `EnrichmentPayload.meta`,
`ClockReport.per_field_source_age_ns`/`.resolved_action_source`,
`EpisodeWriteResult.output_paths`) is typed `Mapping`, not `dict`: mypy rejects
`frame.state["x"] = ...` statically. This costs nothing at runtime — the object
handed to every consumer is still the same real `dict` built at construction, never
copied or wrapped — so no downstream layer needs a defensive copy. It is still only
a static, mypy-enforced convention, not a runtime-enforced one: nothing here stops
`frame.state["left_arm"][:] = ...` (mutating a `np.ndarray` value's contents in
place) or a caller that keeps its own `dict` alias and mutates through that. Treat
every value reachable through these containers as owned-and-not-mutated by
convention, exactly as if it were truly immutable, even though only the top-level
dict shape is statically protected. `CanonicalEpisode.frames` is a single-pass
generator (`Iterator[CanonicalFrame]`), not a list — iterating it a second time
yields nothing. See `protocols.py`'s `BaseEnricher`/`BaseFilter` docstrings for what
this means for stub implementations that need to inspect an episode.

Pydantic is deliberately absent from this module: it never runs in the per-frame hot
path (roughly 1,200 frames x 4 cameras per episode), where numpy arrays pass through
untouched. Pydantic v2 lives only in `profile.py`, for embodiment-profile loading and
validation.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Protocol

import numpy as np

RECORD_SCHEMA_VERSION = 2
"""Freezes the on-disk shape of `EpisodeRecord`. Bump this — and update every
already-written provenance record's expectations — before changing any field on that
container; do not change `EpisodeRecord`'s shape without also bumping this.

**v2 back-compat rule (04-09):** a record with no `ingest_outcome` key at all was
written under `RECORD_SCHEMA_VERSION` 1, before that field existed. Its written-ness
must be DERIVED from artifact presence at its own episode index, never read off
`disposition` — v1 predates the distinction this field exists to make. A v1 record
on an existing EFS prefix keeps loading and keeps resuming under this fallback; this
bump never requires migrating or rewriting an already-written v1 record.

**v2 also covers `superseded_by_episode_index` (04-11):** appended to the same
version-2 shape as `ingest_outcome`, not a second bump — both fields belong to the
same batch's version-2 record shape. A version-2 record written before this field
existed (i.e. by 04-09's code) simply lacks the key, which reads back as `None`,
exactly "not superseded" — no migration needed for that gap either."""


class ImageRef(Protocol):
    """A handle to one camera frame. Decoding is deferred until a consumer actually
    needs pixels — normally the Layer-4 writer, exactly once, immediately before
    feeding the video encoder. Because of this contract, no component ever holds
    more than one frame's worth of decoded pixels across all cameras at a time.
    """

    def decode(self) -> np.ndarray:
        """Decode this frame to an HxWxC uint8 array.

        Not cached by the ref itself — call once, use the result immediately, and
        let it go out of scope so memory stays bounded.
        """
        ...

    @property
    def timestamp_ns(self) -> int:
        """Capture timestamp of this frame, in nanoseconds."""
        ...


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    """One synchronized instant across every stream in an episode.

    `images` holds undecoded `ImageRef` handles, not pixels — see `ImageRef.decode`.
    `action` is a dict, not a single array, because multiple named action
    representations (joint-target, EE-pose, ...) can coexist side by side; none is
    privileged by this container. The same reasoning applies to `state`.
    """

    timestamp_ns: int
    frame_index: int
    images: Mapping[str, ImageRef]
    state: Mapping[str, np.ndarray]
    action: Mapping[str, np.ndarray]
    task_prompt: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeHeader:
    """A fully materialized, cheap episode-level summary.

    Safe to hold thousands of these in memory across a whole batch — this is what
    resumability, batch-level filtering, and manifest writing operate on.
    `global_meta` is where a Synchronizer records anything an Enricher/Filter might
    need, since neither of those may assume `CanonicalEpisode.frames` is
    re-iterable (see `protocols.py`).
    """

    episode_id: str
    fps: float
    robot_type: str
    task_prompt: str | None
    frame_count: int | None = None
    global_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    """One episode. `frames` is a single-pass generator — iterating it a second time
    yields nothing. Any component that needs to inspect frames more than once must
    request a fresh `CanonicalEpisode` (re-parse the source), never reuse this
    one's `frames` a second time.

    Deliberately not named for a "trajectory": `CanonicalFrame.state`/`.action`
    are unconstrained `Mapping`s and nothing on `EmbodimentProfile` requires a
    populated action, so a video-only, world-model-style episode (next-frame
    pixel prediction, no trajectory) fits this same structure. `CanonicalEpisode`
    pairs with the `CanonicalFrame` it contains and claims nothing about
    trajectories (D-12).
    """

    header: EpisodeHeader
    frames: Iterator[CanonicalFrame]

    def __iter__(self) -> Iterator[CanonicalFrame]:
        return self.frames


class Disposition(enum.Enum):
    """The training-disposition axis of `EpisodeRecord` (PROV-04 axis (c)).

    Deliberately separate from `outcome` (task success/failure) and from
    per-check technical validity — a demonstration can be technically perfect and
    still deliberately excluded from training, or vice versa.
    """

    PASS = "PASS"
    PASS_WITH_FLAGS = "PASS_WITH_FLAGS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class IngestOutcome(enum.Enum):
    """Whether an input ever produced a written episode — decided exactly
    once, at write time, and never revised (04-09).

    Deliberately separate from `Disposition`: `Disposition` is the
    training-disposition axis that post-write validation legitimately amends
    (`ProvenanceRecorder.amend_checks`) whenever a standing check fails after
    an episode was already written. `IngestOutcome` answers a narrower,
    permanent question — did this input's episode ever get written at all —
    and no code path amends it after the fact. Conflating the two (reading
    `Disposition.REJECT` as "never written") is the exact defect this type
    exists to close: a written episode that a later fatal check flags is
    still `WRITTEN`, even though its `disposition` reads `REJECT`.
    """

    WRITTEN = "WRITTEN"
    REJECTED_BEFORE_WRITE = "REJECTED_BEFORE_WRITE"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One ingest-time integrity check's outcome, with evidence.

    `severity` is the single inspectable place INTEG-01 requires the
    fatal-vs-advisory policy to live: a fatal check failing aborts the episode
    before encode; an advisory check failing only downgrades disposition.
    """

    check_name: str
    severity: Literal["fatal", "advisory"]
    passed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnrichmentPayload:
    """A Layer-0 quality verdict for one episode.

    The field is named `verdict`, not `action` — a Layer-0 quality verdict and a
    robot kinematic action (`CanonicalFrame.action`) must never share a field
    name, or a careless dict-merge could silently mix the two.
    """

    verdict: str
    score: float | None = None
    source: str = "unset"
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeWriteResult:
    """The receipt `BaseDatasetWriter.write_episode` returns for one episode.

    On a zero-frame write (`frame_count == 0`), `episode_index` is **not** a
    consumed index — writers advance their index counters only on the
    non-zero-frame branch, so this is the index the *next* real episode
    will reuse. A caller numbering a report row or a rejection record from
    a zero-frame result's `episode_index` would give two different inputs
    the same index; a zero-frame result must be treated as "no episode
    consumed" for accounting purposes, never as "episode N, zero frames."
    """

    episode_index: int
    frame_count: int
    output_paths: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One row of PROV-02's per-stream inventory — including streams that were
    seen but deliberately not emitted, so what was left on the table is recorded
    rather than forgotten.
    """

    topic: str
    schema_name: str
    message_count: int
    measured_hz: float | None
    emitted: bool


@dataclass(frozen=True, slots=True)
class ClockReport:
    """PROV-03's per-episode temporal-synchronization report.

    `resolved_action_source` maps each action field to the topic its ordered
    `sources` list actually resolved to — the audit trail CONT-07's promoted
    per-field ordered-source model needs.
    """

    grid_hz: float
    max_skew_ns: int
    mean_skew_ns: int
    per_field_source_age_ns: Mapping[str, int]
    resolved_action_source: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """The on-disk provenance schema written to
    `meta/data_engine/episode_{NNNNNN}.json` — one per episode.

    Technical validity (`checks`), task outcome (`outcome`/`failure_modes`/
    `outcome_labeled_by`), and training disposition (`disposition`) are three
    strictly separate notions, never aliased to one another: a successful
    conversion is not evidence of a successful demonstration, and a failed
    demonstration can be technically perfect and deliberately retained for
    training. The `outcome`/`data_origin`/`parent_episode_ids`/`action_source`
    fields are reserved and defaulted — Phase 1 never populates them, so a
    future phase can start using them without a schema migration.
    """

    record_schema_version: int
    episode_id: str
    source_uri: str
    source_sha256: str
    pipeline_version: str
    profile_hash: str
    git_commit: str
    stream_inventory: tuple[StreamInfo, ...]
    clock_report: ClockReport
    checks: tuple[CheckResult, ...]
    disposition: Disposition
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    failure_modes: tuple[str, ...] = ()
    outcome_labeled_by: Literal["human", "classifier", "unset"] = "unset"
    data_origin: Literal["real", "simulation", "generated_video"] = "real"
    parent_episode_ids: tuple[str, ...] = ()
    action_source: Literal["measured", "simulator_ground_truth", "inverse_dynamics"] = (
        "measured"
    )
    ingest_outcome: IngestOutcome | None = None
    """Never reordered ahead of `disposition` even though it would read
    better there. `ProvenanceRecorder.amend_checks`/`.mark_superseded`
    construct `EpisodeRecord(...)` POSITIONALLY (a deliberate grep-guard
    dodge, `data/provenance/recorder.py`) to avoid ever writing the literal
    text `outcome=` as a keyword argument; inserting a field anywhere before
    the end of that positional argument list would silently shift every
    argument after it into the wrong field. `None` means this record predates
    `RECORD_SCHEMA_VERSION` 2 (see that constant's own docstring for the v1
    back-compat derivation) — a v1 record's absence of this key is never
    treated as `REJECTED_BEFORE_WRITE`."""
    superseded_by_episode_index: int | None = None
    """The LAST field, deliberately — same positional-construction hazard as
    `ingest_outcome` above, so this is appended after it rather than beside
    `disposition`, which it conceptually pairs with. `None` means this
    record is either live (never superseded) or predates this field's
    introduction (04-11) — both read identically as "not superseded", which
    is exactly the safe interpretation for a record that has never been
    rewritten by `ProvenanceRecorder.mark_superseded`. Set only by that one
    method, when a `--retry-rejected` re-conversion of the same input
    succeeds at a NEW episode index — it never means "this episode's own
    write was rejected"; that fact lives entirely in `disposition`/
    `ingest_outcome`, which `mark_superseded` carries through unchanged."""
