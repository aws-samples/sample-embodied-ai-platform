"""CONT-08 fail-loud validation tests and the CONT-07 ordered-source invariant
test.

Every deliberately-broken fixture below is built as an inline YAML string parsed
with `yaml.safe_load`, matching 01-RESEARCH.md Code Examples §5's shape — no
fixture files. Real-profile assertions load the actual
`data/profiles/galaxea_r1_pro.yaml`.
"""

from pathlib import Path

import pydantic
import pytest
import yaml

from data.contracts import EmbodimentProfile, load_profile, profile_hash

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


def _load_broken(yaml_text: str) -> None:
    """Parse `yaml_text` with the safe loader and validate it, the same way
    `load_profile` does, so fixtures never touch the filesystem."""
    raw = yaml.safe_load(yaml_text)
    EmbodimentProfile.model_validate(raw)


BASE_PROFILE = """
embodiment: galaxea_r1_pro
schema_version: 1
robot_type: r1pro
master_clock:
  topic: /hdas/camera_head/left_raw/image_raw_color/compressed
  target_fps: null
cameras:
  head_left:
    topic: /hdas/camera_head/left_raw/image_raw_color/compressed
    role: master_clock
    required: true
    emit: true
    video_key: observation.images.head
state_topics:
  arm_left:
    topic: /hdas/feedback_arm_left
    field: position
    joint_names: [arm_l_j1, arm_l_j2]
    required: true
    resample: interp
    state_key: left_arm
    include_in_observation_state: true
action_topics:
  left_arm:
    sources:
      - topic: /motion_target/target_joint_state_arm_left
        field: position
        message_type: sensor_msgs/msg/JointState
    joint_names: [arm_l_j1, arm_l_j2]
    required: true
    resample: interp
    action_key: left_arm
state_vector_order: [left_arm]
action_primary: [left_arm]
"""


def test_real_profile_validates() -> None:
    profile = load_profile(str(PROFILE_PATH))
    assert isinstance(profile, EmbodimentProfile)
    assert len(profile.state_topics) == 6
    assert len(profile.action_topics) == 4


def test_real_profile_emitted_width_is_20() -> None:
    profile = load_profile(str(PROFILE_PATH))
    keyed = {f.state_key: f for f in profile.state_topics.values()}
    total = sum(
        len(keyed[name].joint_names) for name in profile.state_vector_order
    )
    assert total == 20


def test_left_arm_ordered_sources_preserved() -> None:
    profile = load_profile(str(PROFILE_PATH))
    sources = profile.action_topics["left_arm"].sources
    assert len(sources) == 2
    assert sources[0].topic == "/motion_target/target_joint_state_arm_left"
    assert sources[-1].topic == "/motion_control/control_arm_left"
    assert sources[-1].field == "p_des"


def test_single_source_action_field_is_one_element_list() -> None:
    profile = load_profile(str(PROFILE_PATH))
    assert len(profile.action_topics["left_gripper"].sources) == 1


def test_last_entry_only_populated_still_resolves() -> None:
    """The CONT-07 invariant: a profile whose only populated action source is
    the LAST entry in its ordered `sources` list still validates and still
    exposes a resolvable action mapping for that field."""
    raw = yaml.safe_load(BASE_PROFILE)
    raw["action_topics"]["left_arm"]["sources"] = [
        {
            "topic": "/motion_control/control_arm_left",
            "field": "p_des",
            "message_type": "hdas_msg/msg/MotorControl",
        }
    ]
    profile = EmbodimentProfile.model_validate(raw)
    field = profile.action_topics["left_arm"]
    assert len(field.sources) == 1
    assert field.sources[0].field == "p_des"
    assert field.action_key == "left_arm"


def test_empty_action_sources_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["action_topics"]["left_arm"]["sources"] = []
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "action_topics.left_arm.sources" in str(exc_info.value)


def test_bogus_resample_value_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["resample"] = "bogus_mode"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    message = str(exc_info.value)
    assert "state_topics.arm_left.resample" in message
    assert "interp" in message


def test_missing_required_flag_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    del raw["state_topics"]["arm_left"]["required"]
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "state_topics.arm_left.required" in str(exc_info.value)


def test_unrecognized_extra_key_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["bogus_extra_key"] = "x"
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "bogus_extra_key" in str(exc_info.value)


def test_unknown_state_vector_order_key_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_vector_order"] = ["no_such_key"]
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "no_such_key" in str(exc_info.value)


def test_empty_joint_names_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["joint_names"] = []
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "joint_names" in str(exc_info.value)


def test_duplicate_state_key_across_state_topics_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    dup = dict(raw["state_topics"]["arm_left"])
    dup["topic"] = "/hdas/feedback_arm_left_dup"
    raw["state_topics"]["arm_left_dup"] = dup
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _load_broken(yaml.dump(raw))
    message = str(exc_info.value)
    assert "left_arm" in message
    assert "state_key" in message


def test_duplicate_action_key_across_action_topics_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    dup = dict(raw["action_topics"]["left_arm"])
    dup["sources"] = [dict(s) for s in dup["sources"]]
    raw["action_topics"]["left_arm_dup"] = dup
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _load_broken(yaml.dump(raw))
    message = str(exc_info.value)
    assert "left_arm" in message
    assert "action_key" in message


def test_duplicate_video_key_across_cameras_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["cameras"]["head_left_dup"] = {
        "topic": "/hdas/camera_head/left_raw/image_raw_color/compressed_dup",
        "role": "secondary",
        "required": True,
        "emit": True,
        "video_key": "observation.images.head",
        "sync_policy": "nearest",
        "max_sync_gap_ms": 40,
    }
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _load_broken(yaml.dump(raw))
    message = str(exc_info.value)
    assert "observation.images.head" in message
    assert "video_key" in message


def test_state_field_missing_from_state_vector_order_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_vector_order"] = []
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _load_broken(yaml.dump(raw))
    message = str(exc_info.value)
    assert "arm_left" in message
    assert "left_arm" in message
    assert "state_vector_order" in message


def test_secondary_camera_missing_sync_policy_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["cameras"]["head_right"] = {
        "topic": "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "role": "secondary",
        "required": True,
        "emit": False,
    }
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _load_broken(yaml.dump(raw))
    message = str(exc_info.value)
    assert "head_right" in message
    assert "sync_policy" in message


def test_two_master_clock_cameras_rejected() -> None:
    raw = yaml.safe_load(BASE_PROFILE)
    raw["cameras"]["head_right"] = {
        "topic": "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "role": "master_clock",
        "required": True,
        "emit": False,
    }
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "master_clock" in str(exc_info.value)


def test_load_profile_on_empty_file_names_path(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty_profile.yaml"
    empty_file.write_text("   \n")
    with pytest.raises(ValueError) as exc_info:
        load_profile(empty_file)
    assert str(empty_file) in str(exc_info.value)


def test_load_profile_on_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_profile(missing)
    assert str(missing) in str(exc_info.value)


def test_profile_hash_is_stable_across_reloads() -> None:
    first = load_profile(str(PROFILE_PATH))
    second = load_profile(str(PROFILE_PATH))
    assert profile_hash(first) == profile_hash(second)


def test_profile_hash_changes_after_a_field_changes() -> None:
    original = load_profile(str(PROFILE_PATH))
    original_hash = profile_hash(original)

    raw = yaml.safe_load(BASE_PROFILE)
    changed = EmbodimentProfile.model_validate(raw)
    changed_hash = profile_hash(changed)

    assert original_hash != changed_hash


def test_real_profile_state_fields_have_unset_physics_limits() -> None:
    """D-07: the two optional hard-physics-limit fields default unset, and the
    shipped profile (which declares neither) still validates with both read as
    unset on every state field."""
    profile = load_profile(str(PROFILE_PATH))
    for field in profile.state_topics.values():
        assert field.max_velocity is None
        assert field.max_acceleration is None


def test_state_field_declaring_physics_limits_validates_and_is_readable() -> None:
    """D-07 Test 8: an inline profile declaring a max velocity and max
    acceleration on one state field validates, and both values are readable
    off that field."""
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["max_velocity"] = 3.5
    raw["state_topics"]["arm_left"]["max_acceleration"] = 12.0
    profile = EmbodimentProfile.model_validate(raw)
    field = profile.state_topics["arm_left"]
    assert field.max_velocity == 3.5
    assert field.max_acceleration == 12.0


def test_state_field_unknown_key_still_rejected_after_additive_change() -> None:
    """D-07 Test 9: extra="forbid" rejects unknown YAML keys, unweakened by the
    additive max_velocity/max_acceleration fields."""
    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["bogus_physics_key"] = 1.0
    with pytest.raises(pydantic.ValidationError) as exc_info:
        EmbodimentProfile.model_validate(raw)
    assert "bogus_physics_key" in str(exc_info.value)


def test_profile_hash_stable_after_physics_limit_fields_added() -> None:
    """D-07 Test 10: profile_hash is stable across two reloads of the shipped
    profile after the additive StateField change (it declares neither limit),
    and differs for a profile that declares a limit — proven by comparing
    hashes to each other, never against a literal."""
    first = load_profile(str(PROFILE_PATH))
    second = load_profile(str(PROFILE_PATH))
    assert profile_hash(first) == profile_hash(second)

    raw = yaml.safe_load(BASE_PROFILE)
    raw["state_topics"]["arm_left"]["max_velocity"] = 3.5
    with_limit = EmbodimentProfile.model_validate(raw)
    without_limit = EmbodimentProfile.model_validate(yaml.safe_load(BASE_PROFILE))
    assert profile_hash(with_limit) != profile_hash(without_limit)


def test_profile_py_uses_only_the_safe_yaml_loader() -> None:
    import re

    source = Path("data/contracts/profile.py").read_text()
    assert "yaml.safe_load" in source
    unsafe_pattern = re.compile(r"yaml\.load\(")
    for line in source.splitlines():
        if re.match(r"^[ \t]*#", line):
            continue
        assert not unsafe_pattern.search(line), (
            "unsafe yaml.load( call spelled outside a pure-comment line: " + line
        )


def test_no_single_primary_plus_named_fallback_key() -> None:
    profile_source = Path("data/contracts/profile.py").read_text()
    yaml_source = Path("data/profiles/galaxea_r1_pro.yaml").read_text()
    assert "fallback_topic" not in profile_source
    assert "fallback_topic" not in yaml_source
