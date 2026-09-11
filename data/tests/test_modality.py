"""Tests for `data/writers/modality.py`: the `meta/modality.json` +
`MODALITY_CONFIG_PATH` Python-module generator, and the `finalize()` guard
that makes their absence unreachable (WRITE-07/WRITE-11).
"""

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from data.contracts.profile import load_profile
from data.writers.lerobot_v21 import LeRobotV21Writer
from data.writers.modality import (
    MissingModalityArtifactError,
    _generate_modality_config_source,
    assert_modality_artifacts_present,
    build_modality_json,
    write_modality_artifacts,
)
from data.tests.frame_fixtures import build_synthetic_episode

PROFILE_PATH = "data/profiles/galaxea_r1_pro.yaml"


@pytest.fixture()
def profile():
    return load_profile(PROFILE_PATH)


# ---------------------------------------------------------------------------
# Test 1: state/action slice maps — contiguous, non-overlapping, sum to the
# emitted widths.
# ---------------------------------------------------------------------------


def test_state_slice_map_contiguous_and_sums_to_20(profile) -> None:
    modality_json = build_modality_json(profile)
    state = modality_json["state"]
    assert list(state.keys()) == list(profile.state_vector_order)

    cursor = 0
    for key in profile.state_vector_order:
        rng = state[key]
        assert rng["start"] == cursor
        assert rng["end"] > rng["start"]
        cursor = rng["end"]
    assert cursor == 20


def test_action_slice_map_contiguous_and_sums_to_16(profile) -> None:
    modality_json = build_modality_json(profile)
    action = modality_json["action"]
    assert list(action.keys()) == list(profile.action_primary)

    cursor = 0
    for key in profile.action_primary:
        rng = action[key]
        assert rng["start"] == cursor
        assert rng["end"] > rng["start"]
        cursor = rng["end"]
    assert cursor == 16


# ---------------------------------------------------------------------------
# Test 2: widening one field's joint_names shifts every subsequent boundary.
# ---------------------------------------------------------------------------


def test_widening_one_field_shifts_every_subsequent_boundary(profile) -> None:
    baseline = build_modality_json(profile)
    baseline_total = max(rng["end"] for rng in baseline["state"].values())

    torso = profile.state_topics["torso"]
    widened_torso = torso.model_copy(
        update={"joint_names": [*torso.joint_names, "torso_j5"]}
    )
    new_state_topics = dict(profile.state_topics)
    new_state_topics["torso"] = widened_torso
    widened_profile = profile.model_copy(update={"state_topics": new_state_topics})

    widened = build_modality_json(widened_profile)
    widened_total = max(rng["end"] for rng in widened["state"].values())

    assert widened_total == baseline_total + 1

    order = list(profile.state_vector_order)
    torso_idx = order.index("torso")
    for key in order[torso_idx + 1 :]:
        assert widened["state"][key]["start"] == baseline["state"][key]["start"] + 1
        assert widened["state"][key]["end"] == baseline["state"][key]["end"] + 1
    for key in order[: torso_idx + 1]:
        assert widened["state"][key]["start"] == baseline["state"][key]["start"]
    assert widened["state"]["torso"]["end"] == baseline["state"]["torso"]["end"] + 1


# ---------------------------------------------------------------------------
# Test 3: video map — one entry per emit:true camera, none for emit:false.
# ---------------------------------------------------------------------------


def test_video_map_has_three_entries_none_for_emit_false_camera(profile) -> None:
    modality_json = build_modality_json(profile)
    video = modality_json["video"]
    assert len(video) == 3
    assert "head_right" not in video  # emit: false in the shipped profile

    emitted_cameras = {
        name: cam for name, cam in profile.cameras.items() if cam.emit
    }
    assert set(video.keys()) == set(emitted_cameras.keys())
    for name, cam in emitted_cameras.items():
        assert video[name] == {"original_key": cam.video_key}


# ---------------------------------------------------------------------------
# Test 4: annotation entry maps the human task description to task_index.
# ---------------------------------------------------------------------------


def test_annotation_entry_maps_human_task_description_to_task_index(profile) -> None:
    modality_json = build_modality_json(profile)
    assert modality_json["annotation"] == {
        "human.task_description": {"original_key": "task_index"}
    }


# ---------------------------------------------------------------------------
# Test 5/6: the generated Python module, executed with GR00T stubbed.
# ---------------------------------------------------------------------------


def _stub_gr00t_modules(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
    """Installs minimal fake `gr00t.*` modules into `sys.modules` (removed
    automatically by `monkeypatch` at test teardown) sufficient to `exec`
    the generated modality-config module's source, and returns the list
    `register_modality_config` calls append `(config, embodiment_tag)`
    pairs to."""
    captured: list[tuple[Any, Any]] = []

    class _FakeModalityConfig:
        def __init__(self, *, delta_indices, modality_keys, action_configs=None):
            self.delta_indices = delta_indices
            self.modality_keys = modality_keys
            self.action_configs = action_configs

    class _FakeActionConfig:
        def __init__(self, *, rep, type, format):
            self.rep = rep
            self.type = type
            self.format = format

    class _FakeActionRepresentation:
        RELATIVE = "relative"
        ABSOLUTE = "absolute"

    class _FakeActionType:
        NON_EEF = "non_eef"

    class _FakeActionFormat:
        DEFAULT = "default"

    class _FakeEmbodimentTag:
        NEW_EMBODIMENT = "new_embodiment"

    def _fake_register_modality_config(config, *, embodiment_tag):
        captured.append((config, embodiment_tag))

    mod_gr00t = types.ModuleType("gr00t")
    mod_configs = types.ModuleType("gr00t.configs")
    mod_configs_data = types.ModuleType("gr00t.configs.data")
    mod_embodiment_configs = types.ModuleType("gr00t.configs.data.embodiment_configs")
    setattr(mod_embodiment_configs, "register_modality_config", _fake_register_modality_config)
    mod_data = types.ModuleType("gr00t.data")
    mod_embodiment_tags = types.ModuleType("gr00t.data.embodiment_tags")
    setattr(mod_embodiment_tags, "EmbodimentTag", _FakeEmbodimentTag)
    mod_types = types.ModuleType("gr00t.data.types")
    setattr(mod_types, "ActionConfig", _FakeActionConfig)
    setattr(mod_types, "ActionFormat", _FakeActionFormat)
    setattr(mod_types, "ActionRepresentation", _FakeActionRepresentation)
    setattr(mod_types, "ActionType", _FakeActionType)
    setattr(mod_types, "ModalityConfig", _FakeModalityConfig)

    for name, mod in (
        ("gr00t", mod_gr00t),
        ("gr00t.configs", mod_configs),
        ("gr00t.configs.data", mod_configs_data),
        ("gr00t.configs.data.embodiment_configs", mod_embodiment_configs),
        ("gr00t.data", mod_data),
        ("gr00t.data.embodiment_tags", mod_embodiment_tags),
        ("gr00t.data.types", mod_types),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    return captured


def test_generated_module_matches_profile_and_modality_json_key_sets(
    profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_gr00t_modules(monkeypatch)
    modality_json = build_modality_json(profile)
    source = _generate_modality_config_source(profile, modality_json, action_horizon=16)

    exec(compile(source, "<generated modality config>", "exec"), {})

    assert len(captured) == 1
    config, embodiment_tag = captured[0]
    assert embodiment_tag == "new_embodiment"
    assert config["state"].modality_keys == list(profile.state_vector_order)
    assert config["action"].modality_keys == list(profile.action_primary)
    assert set(config["video"].modality_keys) == set(modality_json["video"].keys())
    assert config["action"].delta_indices == list(range(0, 16))
    assert len(config["action"].action_configs) == len(profile.action_primary)


def test_generated_module_ends_in_register_modality_config_call(profile) -> None:
    modality_json = build_modality_json(profile)
    source = _generate_modality_config_source(profile, modality_json, action_horizon=16)
    stripped = source.rstrip()
    last_line = stripped.splitlines()[-1]
    assert last_line.startswith("register_modality_config(")
    assert "EmbodimentTag.NEW_EMBODIMENT" in last_line


# ---------------------------------------------------------------------------
# Test 7: cross-artifact key-set agreement.
# ---------------------------------------------------------------------------


def test_modality_json_and_module_key_sets_agree(
    profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_gr00t_modules(monkeypatch)
    modality_json = build_modality_json(profile)
    source = _generate_modality_config_source(profile, modality_json, action_horizon=16)
    exec(compile(source, "<generated modality config>", "exec"), {})

    config, _ = captured[0]
    assert set(modality_json["state"].keys()) == set(config["state"].modality_keys)
    assert set(modality_json["action"].keys()) == set(config["action"].modality_keys)
    assert set(modality_json["video"].keys()) == set(config["video"].modality_keys)


# ---------------------------------------------------------------------------
# Test 8: slice ranges applied to a real written parquet row select exactly
# the joint values belonging to that named field.
# ---------------------------------------------------------------------------


def test_slice_ranges_select_correct_joint_values_from_written_parquet(
    profile, tmp_path: Path
) -> None:
    known = {
        "left_arm": [np.full(7, 1.0, dtype=np.float32)],
        "right_arm": [np.full(7, 2.0, dtype=np.float32)],
        "torso": [np.full(4, 3.0, dtype=np.float32)],
        "left_gripper": [np.full(1, 4.0, dtype=np.float32)],
        "right_gripper": [np.full(1, 5.0, dtype=np.float32)],
    }
    episode = build_synthetic_episode(
        frame_count=1,
        fps=10.0,
        profile=profile,
        state_values=known,
        task_prompt="t",
    )

    writer = LeRobotV21Writer(tmp_path, profile, stdin_isatty=lambda: False)
    result = writer.write_episode(episode)
    writer.finalize()

    table = pq.read_table(result.output_paths["parquet"])
    row = np.asarray(table.column("observation.state")[0].as_py(), dtype=np.float32)

    modality_json = build_modality_json(profile)
    for key, values in known.items():
        rng = modality_json["state"][key]
        sliced = row[rng["start"] : rng["end"]]
        np.testing.assert_array_equal(sliced, values[0])


# ---------------------------------------------------------------------------
# Test 9: finalize() raises when either artifact is missing after its write
# attempt.
# ---------------------------------------------------------------------------


def test_finalize_guard_raises_when_modality_artifact_missing(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = build_synthetic_episode(
        frame_count=1, fps=10.0, profile=profile, task_prompt="t"
    )
    writer = LeRobotV21Writer(tmp_path, profile, stdin_isatty=lambda: False)
    writer.write_episode(episode)

    def _write_only_json(profile_, out_dir, *, fps, action_horizon=16):
        modality_json = build_modality_json(profile_)
        meta_dir = Path(out_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "modality.json", "w") as f:
            json.dump(modality_json, f)

    monkeypatch.setattr(
        "data.writers.lerobot_v21.modality.write_modality_artifacts", _write_only_json
    )

    with pytest.raises(Exception) as exc_info:
        writer.finalize()
    message = str(exc_info.value)
    assert "modality_config.py" in message or "modality.json" in message
    assert "6-DoF" in message or "hardcoded" in message


def test_assert_modality_artifacts_present_guard_names_consequence(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    with pytest.raises(MissingModalityArtifactError, match="hardcoded"):
        assert_modality_artifacts_present(tmp_path)


# ---------------------------------------------------------------------------
# Test 10: both artifacts are written atomically.
# ---------------------------------------------------------------------------


def test_write_modality_artifacts_leaves_no_partial_file_on_interruption(
    profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import data.writers.modality as modality_module

    real_replace = modality_module.os.replace
    call_count = {"n": 0}

    def _flaky_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated interruption before second rename")
        return real_replace(src, dst)

    monkeypatch.setattr(modality_module.os, "replace", _flaky_replace)

    with pytest.raises(OSError):
        write_modality_artifacts(profile, tmp_path, fps=10.0)

    meta_dir = tmp_path / "meta"
    # The first artifact's rename succeeded (call 1); it must be complete,
    # never partial, under its final name.
    json_path = meta_dir / "modality.json"
    if json_path.exists():
        json.loads(json_path.read_text())  # must parse cleanly, not truncated
    # The second artifact's rename was interrupted — its final name must be
    # absent, never a truncated file under that name.
    assert not (meta_dir / "modality_config.py").exists()
    # No leftover hidden temp file for the interrupted artifact should
    # remain under its OWN final-named path.
    with pytest.raises(MissingModalityArtifactError):
        assert_modality_artifacts_present(tmp_path)


# ---------------------------------------------------------------------------
# Test 11: emitted state slice total equals info.json's runtime-read width.
# ---------------------------------------------------------------------------


def test_state_slice_total_equals_info_json_state_width(
    profile, tmp_path: Path
) -> None:
    episode = build_synthetic_episode(
        frame_count=1, fps=10.0, profile=profile, task_prompt="t"
    )
    writer = LeRobotV21Writer(tmp_path, profile, stdin_isatty=lambda: False)
    writer.write_episode(episode)
    writer.finalize()

    with open(tmp_path / "meta" / "info.json") as f:
        info = json.load(f)
    info_width = info["features"]["observation.state"]["shape"][0]

    modality_json = build_modality_json(profile)
    modality_total = max(rng["end"] for rng in modality_json["state"].values())

    assert modality_total == info_width
