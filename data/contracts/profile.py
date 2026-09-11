"""Embodiment profile schema: a declarative YAML mapping of robot topics onto
canonical fields, validated with pydantic v2 so an invalid profile fails loud,
naming the exact dotted field path an author must edit — see `load_profile`.

This module contains the data engine's single YAML-parsing call site, and it is
always the safe loader, `yaml.safe_load`. The unsafe full-constructor loader
(spelling it here would trip this module's own anti-pattern grep) can deserialize
arbitrary Python objects from the YAML — a code-execution vector once a profile
file might arrive from any teammate or CI job, not just its original author — so
it is never used, in this module or anywhere else under `data/`.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ResampleMode = Literal["interp", "zoh", "nearest"]
"""Per-field resampling policy: linear interpolation, zero-order hold, or
nearest-timestamp match."""


class StateField(BaseModel):
    """One state topic's mapping onto a `CanonicalFrame.state` entry.

    `joint_names` is author-declared, never derived from a message field — every
    real sample topic's `msg.name` is a single group label (length 1) or empty,
    while `position` carries the real per-joint values, so there is no per-joint
    name array in the recordings to zip against.

    `max_velocity`/`max_acceleration` are optional hard physics limits (D-07):
    per-field maximum joint velocity and maximum joint acceleration, both
    unset (`None`) by default. They belong here, not in run config, because a
    hard physics limit genuinely is part of the robot's derivation — churning
    `profile_hash` when one is corrected is the *right* behaviour (D-07). Soft
    curation knobs (idle-velocity threshold, discontinuity sensitivity) are
    deliberately NOT declared here; per D-07/D-08 those are check-function
    parameters so tuning them leaves the profile hash stable. Leaving a limit
    unset is meaningful: it is what lets 02-02's physics-limit check mark that
    field's evaluation as `no_limit_declared` rather than silently passing.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str
    field: str
    joint_names: list[str] = Field(min_length=1)
    required: bool
    resample: ResampleMode
    state_key: str
    include_in_observation_state: bool
    max_velocity: float | None = None
    max_acceleration: float | None = None


class ActionSource(BaseModel):
    """One candidate topic/attribute pair in an action field's ordered fallback
    chain. `field` is per-entry, precisely because the measured sources
    disagree: `position` on `sensor_msgs/msg/JointState` versus `p_des` on
    `hdas_msg/msg/MotorControl`.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str
    field: str
    message_type: str


class ActionField(BaseModel):
    """One action field's ordered list of candidate sources.

    There is no separate preferred-topic key and no single named-fallback key:
    the ordering of `sources` is the whole mechanism, and the first entry with
    messages wins at resolution time in Phase 2. A single-source field is still
    expressed as a one-element list, never a bare topic key.
    """

    model_config = ConfigDict(extra="forbid")

    sources: list[ActionSource] = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    required: bool
    resample: ResampleMode
    action_key: str


class CameraSpec(BaseModel):
    """One camera's role, requiredness, and emission policy."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    role: Literal["master_clock", "secondary"]
    required: bool
    emit: bool
    video_key: str | None = None
    sync_policy: Literal["nearest"] | None = None
    max_sync_gap_ms: int | None = None


class MasterClock(BaseModel):
    """The timeline every other stream resamples onto."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    target_fps: float | None
    max_camera_coverage_deficit_ticks: int = Field(default=0, ge=0)
    """G-01/VERIFY-01 (04-04-PLAN.md): the maximum number of master-clock
    grid ticks that may be trimmed from an episode because a constraining
    camera's own coverage ran out before the master camera's did. A deficit
    beyond this value rejects the episode before any write; a deficit at or
    below it truncates the grid to the common coverage and keeps the
    episode, flagged `PASS_WITH_FLAGS`. Lives here, not on the run config,
    because it is measured in this clock's own unit (grid ticks) and the
    synchronizer receives only the profile — threading a run knob into
    `BaseSynchronizer`'s frozen signature to carry a per-robot camera
    property would be the wrong seam. The default of `0` is the strict,
    fail-loud value: a profile that declares nothing rejects any coverage
    hole rather than silently trimming, and every existing fixture (equal
    per-camera counts, zero deficit) is unaffected. Hashed into
    `profile_hash` like every other field on this model — declaring or
    changing it supersedes the hash and every prefix built under the old
    one, which is intended."""


class EmbodimentProfile(BaseModel):
    """The complete declarative description of one robot embodiment.

    Every field an adapter, synchronizer, or writer needs to describe a new robot
    lives here — zero robot-specific Python is required anywhere under `data/`.
    """

    model_config = ConfigDict(extra="forbid")

    embodiment: str
    schema_version: int
    robot_type: str
    master_clock: MasterClock
    cameras: dict[str, CameraSpec]
    state_topics: dict[str, StateField]
    extra_state_topics: dict[str, StateField] = {}
    action_topics: dict[str, ActionField]
    state_vector_order: list[str]
    action_primary: list[str]
    expected_action_state_lag: int | None = None
    """VALID-03 (D-GAP-01 Defect B, `03-11-PLAN.md`'s Option B decision):
    the declared tracking-error-minimizing lag, in persisted-grid
    timesteps, between `action_primary`'s target stream and its paired
    `observation.state` feedback — negative means the action target LEADS
    the achieved feedback by that many ticks. `None` (the default) means no
    expectation is declared for this embodiment, and
    `consistency.action_state_lag` abstains (`UNEVALUATED_KEY`) rather than
    silently passing — mirroring `data/adapters/checks.py`'s
    `no_limit_declared` precedent (Phase 2 D-08): a check with no declared
    expectation has nothing to compare a measurement against. Measured on
    `galaxea_r1_pro`: see the field's own comment in
    `data/profiles/galaxea_r1_pro.yaml`, and
    `data/tools/measure_action_state_pairing.py`, the tool that reproduces
    it."""

    @model_validator(mode="after")
    def _validate_master_clock_camera(self) -> "EmbodimentProfile":
        master_clock_cameras = [
            name for name, cam in self.cameras.items() if cam.role == "master_clock"
        ]
        if len(master_clock_cameras) != 1:
            raise ValueError(
                "exactly one entry in cameras must declare role: master_clock, "
                f"found {len(master_clock_cameras)}: {master_clock_cameras}"
            )
        master_clock_camera = self.cameras[master_clock_cameras[0]]
        if master_clock_camera.topic != self.master_clock.topic:
            raise ValueError(
                f"master_clock.topic ({self.master_clock.topic!r}) must equal "
                f"the role: master_clock camera's topic "
                f"({master_clock_camera.topic!r})"
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_output_keys(self) -> "EmbodimentProfile":
        def _dupes_in(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
            """`pairs` is (key, owning entry name); returns `{key: [name, ...]}`
            for every key claimed by more than one entry."""
            seen: dict[str, list[str]] = {}
            for key, name in pairs:
                seen.setdefault(key, []).append(name)
            return {key: names for key, names in seen.items() if len(names) > 1}

        state_pairs = [(f.state_key, name) for name, f in self.state_topics.items()]
        state_pairs += [
            (f.state_key, name) for name, f in self.extra_state_topics.items()
        ]
        state_dupes = _dupes_in(state_pairs)
        if state_dupes:
            detail = ", ".join(
                f"state_key {key!r} used by {sorted(names)}"
                for key, names in sorted(state_dupes.items())
            )
            raise ValueError(
                "duplicate state_key value(s) across state_topics/"
                f"extra_state_topics: {detail}"
            )

        action_pairs = [(f.action_key, name) for name, f in self.action_topics.items()]
        action_dupes = _dupes_in(action_pairs)
        if action_dupes:
            detail = ", ".join(
                f"action_key {key!r} used by {sorted(names)}"
                for key, names in sorted(action_dupes.items())
            )
            raise ValueError(
                f"duplicate action_key value(s) across action_topics: {detail}"
            )

        video_pairs = [
            (c.video_key, name)
            for name, c in self.cameras.items()
            if c.video_key is not None
        ]
        video_dupes = _dupes_in(video_pairs)
        if video_dupes:
            detail = ", ".join(
                f"video_key {key!r} used by {sorted(names)}"
                for key, names in sorted(video_dupes.items())
            )
            raise ValueError(
                f"duplicate video_key value(s) across cameras: {detail}"
            )
        return self

    @model_validator(mode="after")
    def _validate_camera_emit_requires_video_key(self) -> "EmbodimentProfile":
        for name, cam in self.cameras.items():
            if cam.emit and cam.video_key is None:
                raise ValueError(
                    f"cameras.{name}.emit is true but cameras.{name}.video_key "
                    "is not set"
                )
        return self

    @model_validator(mode="after")
    def _validate_state_vector_order(self) -> "EmbodimentProfile":
        observable_state_keys = {
            f.state_key
            for f in self.state_topics.values()
            if f.include_in_observation_state
        }
        for name in self.state_vector_order:
            if name not in observable_state_keys:
                raise ValueError(
                    f"state_vector_order entry {name!r} is not the state_key of "
                    "any state_topics entry with include_in_observation_state: "
                    f"true (available: {sorted(observable_state_keys)})"
                )

        # Reverse direction: every state_topics entry eligible for the emitted
        # observation.state vector must actually be named in
        # state_vector_order, or it is silently dropped from the output.
        order_set = set(self.state_vector_order)
        missing = [
            (field_name, f.state_key)
            for field_name, f in self.state_topics.items()
            if f.include_in_observation_state and f.state_key not in order_set
        ]
        if missing:
            detail = ", ".join(
                f"state_topics.{field_name} (state_key={state_key!r})"
                for field_name, state_key in sorted(missing)
            )
            raise ValueError(
                "state_topics entries with include_in_observation_state: true "
                f"but missing from state_vector_order: {detail}"
            )
        return self

    @model_validator(mode="after")
    def _validate_secondary_camera_sync_fields(self) -> "EmbodimentProfile":
        for name, cam in self.cameras.items():
            if cam.role != "secondary":
                continue
            if cam.sync_policy is None:
                raise ValueError(
                    f"cameras.{name}.role is secondary but "
                    f"cameras.{name}.sync_policy is not set"
                )
            if cam.sync_policy == "nearest" and cam.max_sync_gap_ms is None:
                raise ValueError(
                    f"cameras.{name}.sync_policy is 'nearest' but "
                    f"cameras.{name}.max_sync_gap_ms is not set"
                )
        return self

    @model_validator(mode="after")
    def _validate_action_primary(self) -> "EmbodimentProfile":
        action_keys = {f.action_key for f in self.action_topics.values()}
        for name in self.action_primary:
            if name not in action_keys:
                raise ValueError(
                    f"action_primary entry {name!r} is not the action_key of "
                    f"any action_topics entry (available: {sorted(action_keys)})"
                )
        return self


def load_profile(path: str | Path) -> EmbodimentProfile:
    """Load and validate an embodiment profile from a YAML file.

    Args:
        path: Path to the profile YAML file.

    Returns:
        The validated `EmbodimentProfile`.

    Raises:
        FileNotFoundError: If no file exists at `path`. The message names the
            path.
        ValueError: If the file parses to an empty document or to a YAML
            document that is not a mapping (e.g. a bare list or scalar). The
            message names the path — a partially-populated profile is never
            returned.
        pydantic.ValidationError: If the parsed document does not satisfy the
            schema. The error names the offending dotted field path and the
            violated constraint; no custom formatting is applied on top of it.

    Examples:
        >>> profile = load_profile("data/profiles/galaxea_r1_pro.yaml")
        >>> profile.robot_type
        'r1pro'
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"profile file not found: {resolved}")

    with open(resolved) as f:
        raw = yaml.safe_load(f)

    if raw is None or not isinstance(raw, dict):
        raise ValueError(
            f"profile file {resolved} parsed to {raw!r}, expected a YAML mapping "
            "— the file may be empty or malformed"
        )

    return EmbodimentProfile.model_validate(raw)


def profile_hash(profile: EmbodimentProfile) -> str:
    """Compute a stable hex sha256 identifier for `profile`.

    The hash covers the profile only — run knobs (fps override, task string,
    output dir, codec) live in a separate batch config file, so a changed run
    knob never churns the profile hash. PROV-01 and CLI-08 use this identifier
    in later phases.

    Args:
        profile: The profile to hash.

    Returns:
        A hex-encoded sha256 digest of the profile's canonical JSON
        serialization (sorted keys, so field order never affects the hash).
    """
    canonical = json.dumps(profile.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
