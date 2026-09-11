"""Generates `meta/modality.json` and the `MODALITY_CONFIG_PATH` Python
module from the same `EmbodimentProfile` that drives the LeRobot v2.1
parquet schema (`data/writers/lerobot_v21.py`) — one source of truth for the
state/action slice map, per WRITE-07/WRITE-11.

Hand-authoring a modality-config slice map separately from the writer's own
feature schema is a named anti-pattern
(`.planning/phases/02-adapter-synchronizer-writer/02-RESEARCH.md` § Don't
Hand-Roll): NVIDIA's own shipped SO-101 example
(`training/gr00t/N17/so101_modality_config.py`) has shipped with wrong
camera keys in the past precisely because the two artifacts drifted
independently. Generating both from `profile.state_vector_order` /
`profile.action_primary` / `cameras[*].video_key` — accumulating every
boundary from declared widths, never a literal — makes that drift
structurally unreachable.

`meta/modality.json`'s absence is the sharpest silent failure in this whole
pipeline: N1.7's finetune script (`training/gr00t/N17/finetune_gr00t.py`,
`validate_dataset()`) writes an **ungated** hardcoded 6-DoF SO-101 slice map
when the file is missing, so a 20-DoF Galaxea dataset would be silently
mis-sliced instead of failing loudly
(`.planning/research/DERISK-FINDINGS.md` § DERISK-04). `write_modality_artifacts`
writes both this module's artifacts atomically;
`LeRobotV21Writer.finalize()` calls `assert_modality_artifacts_present`
immediately afterward and raises, naming that exact consequence, if either
is absent — that pairing is what makes the absent-file state unreachable
for any dataset this writer finalizes (WRITE-07).

Also per DERISK-04/`GR00T-CONTRACT.md`: `modality.json` alone is
insufficient for a non-SO101 embodiment — N1.7 additionally loads a Python
module (path given by the `MODALITY_CONFIG_PATH` env var) that registers
`ModalityConfig` objects, via `importlib.util.spec_from_file_location` +
`exec_module`, whose side effect is a `register_modality_config(...)` call
against `EmbodimentTag.NEW_EMBODIMENT`. This module generates that file's
*source text* — it never imports the `gr00t` package itself (not installed
in this development environment; the generated file is executed only
inside the GR00T training container, per this plan's own flagged
assumption for WRITE-11).
"""

import json
import os
from pathlib import Path
from typing import Any

from data.contracts.profile import EmbodimentProfile

MODALITY_JSON_FILENAME = "modality.json"
MODALITY_CONFIG_MODULE_FILENAME = "modality_config.py"

DEFAULT_ACTION_HORIZON = 16
"""The action-chunking horizon N1.7's own shipped SO-101 example declares
(`training/gr00t/N17/so101_modality_config.py`:
`delta_indices=list(range(0, 16))`). This is a training-config policy
choice — how many future action steps one training sample's action target
covers — orthogonal to the embodiment's DoF count, so it is never derived
from `profile` and is safe to share across embodiments unless a future run
explicitly overrides it."""

_ANNOTATION_SUBKEY = "human.task_description"
"""Matches `training/gr00t/N17/finetune_gr00t.py`'s own `annotation_key` —
the language-modality key the loader resolves through `modality.json`'s
`annotation` map (`GR00T-CONTRACT.md` Correction 5 / `DERISK-FINDINGS.md`
§ DERISK-04)."""

_LANGUAGE_MODALITY_KEY = f"annotation.{_ANNOTATION_SUBKEY}"
"""The `language` `ModalityConfig`'s `modality_keys` entry — prefixed with
`annotation.`, unlike `modality.json`'s own `annotation` map key, which
carries the bare `_ANNOTATION_SUBKEY`. Both name text pulled verbatim from
the shipped SO-101 template."""


class MissingModalityArtifactError(RuntimeError):
    """Raised by `assert_modality_artifacts_present` when either
    `meta/modality.json` or `meta/modality_config.py` is absent after a
    write attempt."""


def _state_slice_map(profile: EmbodimentProfile) -> dict[str, dict[str, int]]:
    """One contiguous, non-overlapping `{"start", "end"}` range per
    `state_vector_order` entry, in that declared order, packed from 0 with
    no gaps. Each range's width is the *declared* `joint_names` length of
    the `state_topics`/`extra_state_topics` entry owning that `state_key` —
    never a literal, so widening one field shifts every subsequent boundary
    with no code change."""
    by_state_key = {
        f.state_key: f
        for f in (
            list(profile.state_topics.values()) + list(profile.extra_state_topics.values())
        )
    }
    ranges: dict[str, dict[str, int]] = {}
    cursor = 0
    for state_key in profile.state_vector_order:
        width = len(by_state_key[state_key].joint_names)
        ranges[state_key] = {"start": cursor, "end": cursor + width}
        cursor += width
    return ranges


def _action_slice_map(profile: EmbodimentProfile) -> dict[str, dict[str, int]]:
    """Same construction as `_state_slice_map`, over `action_primary` and
    `action_topics`."""
    by_action_key = {f.action_key: f for f in profile.action_topics.values()}
    ranges: dict[str, dict[str, int]] = {}
    cursor = 0
    for action_key in profile.action_primary:
        width = len(by_action_key[action_key].joint_names)
        ranges[action_key] = {"start": cursor, "end": cursor + width}
        cursor += width
    return ranges


def _video_map(profile: EmbodimentProfile) -> dict[str, dict[str, str]]:
    """One entry per `emit: true` camera, keyed by that camera's own
    profile dict key, carrying an `original_key` mapping to its declared
    `video_key`. A camera with `emit: false` (e.g. the shipped profile's
    `head_right`) never gets an entry — `EmbodimentProfile` validation
    already guarantees every `emit: true` camera has a `video_key`."""
    return {
        name: {"original_key": cam.video_key}
        for name, cam in profile.cameras.items()
        if cam.emit and cam.video_key is not None
    }


def build_modality_json(profile: EmbodimentProfile) -> dict[str, Any]:
    """The full `meta/modality.json` document for `profile` — every
    boundary, key name, and total computed from the profile's declared
    widths, never a literal."""
    return {
        "state": _state_slice_map(profile),
        "action": _action_slice_map(profile),
        "video": _video_map(profile),
        "annotation": {_ANNOTATION_SUBKEY: {"original_key": "task_index"}},
    }


def _rep_for_action_key(action_key: str) -> str:
    """Per-field action representation, matching the shipped SO-101
    template's own rule: a gripper field is `ABSOLUTE` (a gripper position
    is not naturally expressed as a delta); every other action field (an
    arm) is `RELATIVE` — GR00T's processor performs the
    relative<->absolute conversion at load time
    (`.planning/research/DERISK-FINDINGS.md` Correction 8); the stored
    parquet values themselves stay absolute regardless of this
    declaration."""
    return "ABSOLUTE" if "gripper" in action_key else "RELATIVE"


def _sanitize_identifier(name: str) -> str:
    """A Python-identifier-safe stem for the generated module's config
    variable name, derived from `profile.embodiment` — never a literal
    embodiment name baked into this generator itself."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _generate_modality_config_source(
    profile: EmbodimentProfile,
    modality_json: dict[str, Any],
    *,
    action_horizon: int,
) -> str:
    """Generates the `MODALITY_CONFIG_PATH` Python module's full source
    text, modelled on `training/gr00t/N17/so101_modality_config.py`: the
    same four `ModalityConfig` entries (`video`, `state`, `action`,
    `language`), with `state`/`action`/`video` modality keys taken directly
    from `profile.state_vector_order` / `profile.action_primary` /
    `modality_json["video"]` — the exact same key sets `modality.json`
    itself carries, so the two artifacts cannot drift. Never imports the
    `gr00t` package — this only writes source text for a module that is
    executed inside the GR00T container, not here.
    """
    video_keys = list(modality_json["video"].keys())
    state_keys = list(profile.state_vector_order)
    action_keys = list(profile.action_primary)
    config_var = f"{_sanitize_identifier(profile.embodiment)}_config"

    action_config_entries = "\n".join(
        "        ActionConfig(\n"
        f"            rep=ActionRepresentation.{_rep_for_action_key(action_key)},\n"
        "            type=ActionType.NON_EEF,\n"
        "            format=ActionFormat.DEFAULT,\n"
        "        ),"
        for action_key in action_keys
    )

    return f'''"""Modality configuration for {profile.embodiment!r} ({profile.robot_type!r}),
generated by data/writers/modality.py from the {profile.embodiment!r} embodiment
profile — DO NOT HAND-EDIT. Regenerate by re-running the data engine's writer
against the same profile; hand-editing this file separately from
meta/modality.json is exactly the drift this generator exists to prevent.

Loaded by GR00T's finetune script via the MODALITY_CONFIG_PATH environment
variable (training/gr00t/N17/finetune_gr00t.py's own default points at the
SO-101 example; this file replaces that default for this embodiment) —
importing this module triggers register_modality_config(...) as a side
effect.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

{config_var} = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys={video_keys!r},
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys={state_keys!r},
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, {action_horizon})),
        modality_keys={action_keys!r},
        action_configs=[
{action_config_entries}
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[{_LANGUAGE_MODALITY_KEY!r}],
    ),
}}

register_modality_config({config_var}, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''


def write_modality_artifacts(
    profile: EmbodimentProfile,
    out_dir: str | Path,
    *,
    fps: float,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> None:
    """Generates and atomically writes both `meta/modality.json` and the
    `MODALITY_CONFIG_PATH` Python module (`meta/modality_config.py`) from
    `profile` — the same profile that drives `LeRobotV21Writer`'s parquet
    schema.

    `fps` is accepted for interface symmetry with the writer's own
    `finalize()` call (which always has an `fps` value on hand) and is
    reserved for a future modality field; none of `video`/`state`/
    `language`'s `delta_indices` depend on frame rate (all are `[0]`), and
    `action`'s `delta_indices` is entirely determined by `action_horizon` —
    so `fps` contributes nothing to either artifact's content today.

    Writes `meta/modality.json` first, then the Python module, each by its
    own atomic temp-then-rename — never partially, and never leaving a
    file under its final name incomplete. If the second write fails after
    the first succeeded, `modality.json` is left correctly written and
    complete on disk; it is the caller's own post-call existence guard
    (`assert_modality_artifacts_present`, which `LeRobotV21Writer.finalize()`
    calls immediately after this function returns) that turns "one of the
    two artifacts is missing" into a raised error naming the
    silent-mis-slice consequence. This function itself never leaves a
    partial file under either final name.
    """
    del fps  # reserved — see docstring; not used by either artifact yet.
    out_path = Path(out_dir)
    meta_dir = out_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modality_json = build_modality_json(profile)
    json_path = meta_dir / MODALITY_JSON_FILENAME
    tmp_json_path = meta_dir / f".{MODALITY_JSON_FILENAME}.tmp"
    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(modality_json, f, indent=2, sort_keys=True)
    os.replace(tmp_json_path, json_path)

    module_source = _generate_modality_config_source(
        profile, modality_json, action_horizon=action_horizon
    )
    module_path = meta_dir / MODALITY_CONFIG_MODULE_FILENAME
    tmp_module_path = meta_dir / f".{MODALITY_CONFIG_MODULE_FILENAME}.tmp"
    with open(tmp_module_path, "w", encoding="utf-8") as f:
        f.write(module_source)
    os.replace(tmp_module_path, module_path)


def assert_modality_artifacts_present(out_dir: str | Path) -> None:
    """Raises `MissingModalityArtifactError`, naming the silent-mis-slice
    consequence, if either modality artifact is absent under
    `out_dir/meta/`. `LeRobotV21Writer.finalize()` calls this immediately
    after `write_modality_artifacts` returns — this pairing is what makes
    the absent-file state unreachable for any dataset this writer
    finalizes (WRITE-07)."""
    meta_dir = Path(out_dir) / "meta"
    missing = [
        name
        for name in (MODALITY_JSON_FILENAME, MODALITY_CONFIG_MODULE_FILENAME)
        if not (meta_dir / name).exists()
    ]
    if missing:
        raise MissingModalityArtifactError(
            f"required modality artifact(s) missing after write attempt: {missing} — "
            "GR00T's N1.7 finetune script (training/gr00t/N17/finetune_gr00t.py) "
            "writes an UNGATED hardcoded 6-DoF SO-101 slice map when "
            "meta/modality.json is absent, silently mis-slicing this dataset's real "
            "DoF layout instead of failing; the MODALITY_CONFIG_PATH Python module "
            "registration is likewise required and has no safe default for a "
            "non-SO101 embodiment."
        )
