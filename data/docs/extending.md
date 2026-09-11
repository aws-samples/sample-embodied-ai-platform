# Extending: adding a new robot embodiment

## Profile-only, or new adapter?

This is the decision that determines how much work adding a new robot actually is.

**Profile-only** is enough when your recordings are ROS2 MCAP files whose messages the existing decode path already handles. The shipped profile's own header states there is zero robot-specific Python anywhere under `data/` — that is the design intent, and it holds for any robot whose data looks like Galaxea's (ROS2/MCAP container, `ros2msg`-encoded schemas).

**You need a new `BaseInputAdapter`** when the source container or encoding differs — a non-MCAP container, or an MCAP whose schemas are IDL-encoded rather than `ros2msg` (the embedded-schema decoder handles `ros2msg` only). See [protocols.md](protocols.md) for that case. The rest of this page covers the profile-only path.

## Walking the profile

Use [`../profiles/galaxea_r1_pro.yaml`](../profiles/galaxea_r1_pro.yaml) as the worked example throughout — it is the only real embodiment in this repo, so nothing below is invented. Its schema lives in [`../contracts/profile.py`](../contracts/profile.py).

**Identity.** `embodiment` (a name), `schema_version` (the profile schema's own version), `robot_type` — plain identifying strings.

**`master_clock`.** `topic` names the camera that IS the timeline (it must match the `role: master_clock` camera's own topic, below). `target_fps: null` means the native rate — Galaxea's is measured, not declared. `max_camera_coverage_deficit_ticks` is the tick tolerance for a camera whose coverage runs out before the master's; the default is `0`, the strict, fail-loud value — a profile declaring nothing rejects any coverage hole rather than silently trimming.

**`cameras`.** Each entry declares `role` (`master_clock` or `secondary`), `required`, `emit` (whether this camera's video is written to the output at all), `video_key` (the emitted feature name, required when `emit: true`), `sync_policy` (a secondary camera's match strategy), and `max_sync_gap_ms` (the gap bound `sync_policy: nearest` requires). The emitted stereo pair is a choice, not a given — the real profile captures a second head camera with `emit: false`: it's recorded and available, just not written by default.

**`state_topics`.** Each entry maps one feedback topic to one canonical state field: `topic`, `field` (which message attribute carries the values), `joint_names` (the per-joint name array, author-declared — see the callout below), `resample` (`interp`, `zoh`, or `nearest`), `state_key` (the canonical field name), `include_in_observation_state` (whether this field appears in the emitted `observation.state` vector at all), plus optional `max_velocity`/`max_acceleration` physics limits.

**`extra_state_topics`.** Same shape as `state_topics`, for fields declared and captured but not necessarily part of the primary state contract.

**`action_topics`.** Each entry's `sources` is an **ordered fallback chain** — a list of `{topic, field, message_type}` — and the first entry that actually carries messages wins at resolution time. `field` lives per-entry precisely because real sources disagree on the attribute name: Galaxea's own profile has one arm's target-joint topic use `position` while its motor-control fallback uses `p_des`.

**`state_vector_order`** and **`action_primary`** are the emitted column orders — the sequence your robot's `observation.state` and `action` columns will actually appear in on disk.

**`expected_action_state_lag`** — see its own callout below.

### Callout: `joint_names` is declared, not derived

`joint_names` is author-declared, never derived from any MCAP message field. On the real Galaxea recordings, every feedback topic's `msg.name` carries a single group label of length 1 (e.g. `['left_arm']`) — only the `position` field carries the real per-joint values, at whatever length that field happens to be. There is no per-joint name array anywhere in the recordings to zip against. A new robot's author is therefore declaring authoritative ordering, not transcribing it from the data.

### Callout: `expected_action_state_lag` is measured, not guessed

Measure it with `python -m data.tools.measure_action_state_pairing` against real recordings of your new robot rather than guessing a value. Leaving it unset does not mean "assume zero lag" — it makes the paired-lag consistency check abstain rather than silently pass. The same "unset means abstain, not pass" rule holds for the optional `max_velocity`/`max_acceleration` physics limits: an unset limit means that check has nothing to compare a measurement against, and it says so rather than reporting a false pass.

## The validators that will reject your profile

[`../contracts/profile.py`](../contracts/profile.py) validates the whole document with pydantic v2. Every real check, and the mistake each one catches:

- **Unknown keys are always an error.** Every model forbids extra fields, so a typo'd key surfaces as a validation error instead of a silently ignored no-op.
- **Exactly one master-clock camera.** Exactly one entry in `cameras` must declare `role: master_clock`, and its `topic` must equal `master_clock.topic` — a mismatch or a count other than 1 is rejected.
- **Unique output keys.** `state_key`, `action_key`, and `video_key` values must each be unique across their owners — two fields silently colliding on the same output column is caught here, not at write time.
- **`emit: true` requires `video_key`.** A camera you intend to write must name the feature it will be written as.
- **Secondary cameras must declare a sync policy.** A `role: secondary` camera must set `sync_policy`, and a `nearest` policy must also set `max_sync_gap_ms`.
- **`state_vector_order` is checked in both directions.** Every entry must name a real, observation-eligible `state_key`, and every `state_topics` entry with `include_in_observation_state: true` must actually appear in `state_vector_order` — so an eligible field can never be silently dropped from the emitted vector.
- **`action_primary` entries must be real action keys.** Every entry must name a real `action_topics` entry's `action_key`.

Every one of these errors names the offending dotted field path, so the message tells you exactly which line to edit — it is not a generic "validation failed."

## Workflow

1. Copy the shipped profile: `cp data/profiles/galaxea_r1_pro.yaml data/profiles/<your_robot>.yaml`.
2. Edit it, section by section, per the walk-through above.
3. Validate it in isolation — no dataset needed:
   ```bash
   python -c "from data.contracts import load_profile; load_profile('data/profiles/<your_robot>.yaml')"
   ```
4. Write a separate run config modeled on [`../run_configs/galaxea_r1_pro_batch.yaml`](../run_configs/galaxea_r1_pro_batch.yaml). Make sure `input_glob` actually matches your real on-disk nesting — the shipped config uses a one-level-nested pattern (`*/*.mcap`) because each Galaxea recording sits in its own directory; your layout may differ.
5. Dry-run the conversion to see resolved action sources and predicted frame counts before spending any real decode time: `python -m data.cli <output-prefix> --profile data/profiles/<your_robot>.yaml --run-config <your-run-config> --dry-run`.
6. Convert (drop `--dry-run`).
7. Validate: `python -m data.validate <output-prefix> --profile data/profiles/<your_robot>.yaml`.

**Before your second run, understand this:** the profile is hashed. Any edit to it produces a new `profile_hash` and therefore a new dataset revision — it never silently amends an existing output prefix. See [../README.md](../README.md) for the full command reference and [architecture.md](architecture.md) for the pipeline your profile drives.
