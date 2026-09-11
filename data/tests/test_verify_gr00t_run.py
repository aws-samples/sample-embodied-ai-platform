"""Tests for `data.tools.verify_gr00t_run` -- one test per behavior bullet in
04-03-PLAN.md's Task 2. No GPU, no container, no network: every branch is
exercisable from hand-built `trainer_state.json` / `meta/info.json` fixtures.

The width fixtures deliberately use widths that are NEITHER 20 NOR 16 (the
real Galaxea profile's widths) -- a state width of 7 and an action width of 5
-- so a tool that quietly hardcoded the Galaxea profile's real widths would
fail these tests rather than pass them by coincidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from data.tools.verify_gr00t_run import (
    BATCH_WIDTHS_UNEVALUATED,
    DEGENERACY_ASSERTIONS,
    GR00T_VERIFY_REPORT_FILENAME,
    GRAD_NORM_UNEVALUATED,
    BatchWidthReport,
    DegeneracyReport,
    assert_batch_widths,
    assert_non_degenerate,
    main,
    read_declared_widths,
    read_log_history,
)

_FIXTURE_STATE_WIDTH = 7
_FIXTURE_ACTION_WIDTH = 5


def _write_trainer_state(tmp_path: Path, log_history: list[dict]) -> Path:
    checkpoint_dir = tmp_path / "output" / "checkpoint-20"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state_path.write_text(json.dumps({"log_history": log_history}), encoding="utf-8")
    return trainer_state_path


def _write_dataset_info(
    tmp_path: Path,
    *,
    state_width: int | None = _FIXTURE_STATE_WIDTH,
    action_width: int | None = _FIXTURE_ACTION_WIDTH,
    omit_state: bool = False,
    omit_action: bool = False,
    empty_state_shape: bool = False,
) -> Path:
    dataset_dir = tmp_path / "dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    features: dict = {}
    if not omit_state:
        features["observation.state"] = {
            "shape": [] if empty_state_shape else [state_width]
        }
    if not omit_action:
        features["action"] = {"shape": [action_width]}
    (meta_dir / "info.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    return dataset_dir


def _healthy_log_history() -> list[dict]:
    return [
        {"step": 1, "loss": 0.42, "grad_norm": 1.3},
        {"step": 2, "loss": 0.39, "grad_norm": 1.1},
        {"step": 3, "loss": 0.35, "grad_norm": 0.9},
    ]


# ---------------------------------------------------------------------------
# Behavior bullet 1: a healthy log_history passes all four assertions; with
# matching widths also supplied it exits 0.
# ---------------------------------------------------------------------------


def test_healthy_history_passes_all_four_assertions():
    report = assert_non_degenerate(_healthy_log_history())
    assert report.passed is True
    assert report.unevaluated == ()
    assert report.failures == ()


def test_healthy_history_with_matching_widths_exits_zero(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    dataset_dir = _write_dataset_info(tmp_path)
    exit_code = main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-abc12345",
            "--dataset-dir",
            str(dataset_dir),
            "--observed-state-width",
            str(_FIXTURE_STATE_WIDTH),
            "--observed-action-width",
            str(_FIXTURE_ACTION_WIDTH),
        ]
    )
    assert exit_code == 0
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["degeneracy"]["passed"] is True
    assert record["batch_widths"]["match"] is True


# ---------------------------------------------------------------------------
# Behavior bullet 2: loss: 0.0 at the first entry fails the loss-must-exceed-
# zero assertion and the process exits non-zero.
# ---------------------------------------------------------------------------


def test_zero_loss_first_entry_fails_and_names_assertion():
    log_history = [{"step": 1, "loss": 0.0, "grad_norm": 1.0}]
    report = assert_non_degenerate(log_history)
    assert report.passed is False
    assert any("loss_positive_at_first_step" in f for f in report.failures)


def test_zero_loss_first_entry_drives_nonzero_exit(tmp_path):
    trainer_state_path = _write_trainer_state(
        tmp_path, [{"step": 1, "loss": 0.0, "grad_norm": 1.0}]
    )
    exit_code = main(
        ["--trainer-state", str(trainer_state_path), "--revision", "rev-abc12345"]
    )
    assert exit_code != 0


# ---------------------------------------------------------------------------
# Behavior bullet 3: grad_norm NaN fails, naming that assertion, and the
# report identifies the entry's step.
# ---------------------------------------------------------------------------


def test_nan_grad_norm_fails_and_identifies_step():
    log_history = [
        {"step": 1, "loss": 0.5, "grad_norm": 1.0},
        {"step": 2, "loss": 0.4, "grad_norm": math.nan},
    ]
    report = assert_non_degenerate(log_history)
    assert report.passed is False
    grad_norm_failures = [f for f in report.failures if "grad_norm_positive" in f]
    assert grad_norm_failures, report.failures
    assert "step=2" in grad_norm_failures[0]


# ---------------------------------------------------------------------------
# Behavior bullet 4: Inf in any numeric field at any step fails, naming the
# step and the field.
# ---------------------------------------------------------------------------


def test_inf_in_any_field_fails_naming_step_and_field():
    log_history = [
        {"step": 1, "loss": 0.5, "grad_norm": 1.0},
        {"step": 2, "loss": 0.4, "grad_norm": 1.0, "learning_rate": math.inf},
    ]
    report = assert_non_degenerate(log_history)
    assert report.passed is False
    inf_failures = [f for f in report.failures if "no_nan_or_inf" in f]
    assert inf_failures, report.failures
    assert "step=2" in inf_failures[0]
    assert "learning_rate" in inf_failures[0]


# ---------------------------------------------------------------------------
# Behavior bullet 5: loss bit-identical across every logged step fails the
# optimizer-moves assertion.
# ---------------------------------------------------------------------------


def test_bit_identical_loss_fails_optimizer_moves_assertion():
    log_history = [
        {"step": 1, "loss": 0.5, "grad_norm": 1.0},
        {"step": 2, "loss": 0.5, "grad_norm": 1.0},
        {"step": 3, "loss": 0.5, "grad_norm": 1.0},
    ]
    report = assert_non_degenerate(log_history)
    assert report.passed is False
    assert any("loss_not_bit_identical" in f for f in report.failures)


# ---------------------------------------------------------------------------
# Behavior bullet 6: no grad_norm key anywhere reports
# GRAD_NORM_UNEVALUATED, does not raise, does not report a pass, and exits
# non-zero.
# ---------------------------------------------------------------------------


def test_missing_grad_norm_key_reports_unevaluated_not_pass():
    log_history = [
        {"step": 1, "loss": 0.5},
        {"step": 2, "loss": 0.4},
        {"step": 3, "loss": 0.3},
    ]
    report = assert_non_degenerate(log_history)
    assert report.passed is False
    assert GRAD_NORM_UNEVALUATED in report.unevaluated


def test_missing_grad_norm_key_drives_nonzero_exit_no_raise(tmp_path):
    trainer_state_path = _write_trainer_state(
        tmp_path,
        [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}, {"step": 3, "loss": 0.3}],
    )
    exit_code = main(
        ["--trainer-state", str(trainer_state_path), "--revision", "rev-abc12345"]
    )
    assert exit_code != 0


# ---------------------------------------------------------------------------
# Behavior bullet 7: a trainer_state.json that does not exist exits non-zero
# naming the path; a file with no log_history key exits non-zero naming the
# missing key.
# ---------------------------------------------------------------------------


def test_absent_trainer_state_raises_naming_path(tmp_path):
    missing = tmp_path / "does" / "not" / "exist" / "trainer_state.json"
    with pytest.raises(ValueError, match=str(missing).replace("\\", "\\\\")):
        read_log_history(missing)


def test_absent_trainer_state_via_cli_exits_nonzero(tmp_path):
    missing = tmp_path / "output" / "checkpoint-20" / "trainer_state.json"
    exit_code = main(
        ["--trainer-state", str(missing), "--revision", "rev-abc12345"]
    )
    assert exit_code != 0


def test_trainer_state_missing_log_history_key_raises_naming_key(tmp_path):
    checkpoint_dir = tmp_path / "output" / "checkpoint-20"
    checkpoint_dir.mkdir(parents=True)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state_path.write_text(json.dumps({"not_log_history": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="log_history"):
        read_log_history(trainer_state_path)


# ---------------------------------------------------------------------------
# Behavior bullet 8: an empty log_history list exits non-zero as
# unevaluated rather than trivially passing zero assertions.
# ---------------------------------------------------------------------------


def test_empty_log_history_is_unevaluated_not_a_trivial_pass():
    report = assert_non_degenerate([])
    assert report.passed is False
    assert report.unevaluated != ()


def test_empty_log_history_every_assertion_listed_unevaluated():
    report = assert_non_degenerate([])
    # Every one of DEGENERACY_ASSERTIONS-adjacent checks must be named as
    # unevaluated -- loss, grad_norm (via its marker) and the
    # bit-identical check all have nothing to evaluate.
    assert "loss_positive_at_first_step" in report.unevaluated
    assert GRAD_NORM_UNEVALUATED in report.unevaluated
    assert "loss_not_bit_identical" in report.unevaluated


# ---------------------------------------------------------------------------
# Behavior bullet 9: the emitted report carries the dataset revision passed
# in, the four assertion outcomes, every logged step's raw numbers, and the
# D-15 boundary statement verbatim.
# ---------------------------------------------------------------------------


def test_emitted_record_carries_revision_steps_and_d15_boundary(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-deadbeef",
        ]
    )
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["dataset_revision"] == "rev-deadbeef"
    assert record["degeneracy"]["assertions"] == [
        "loss_positive_at_first_step",
        "grad_norm_positive",
        "no_nan_or_inf",
        "loss_not_bit_identical",
    ]
    assert record["steps"] == _healthy_log_history()
    assert "VERIFY-01" in record["documentation"]
    assert "GR00T#314" in record["documentation"]


# ---------------------------------------------------------------------------
# Behavior bullet 10 (-k widths): read_declared_widths over a meta/info.json
# whose widths are neither 20 nor 16 returns those widths, taken from each
# feature's trailing shape entry.
# ---------------------------------------------------------------------------


def test_widths_read_declared_widths_from_info_json(tmp_path):
    dataset_dir = _write_dataset_info(tmp_path, state_width=7, action_width=5)
    declared = read_declared_widths(dataset_dir)
    assert declared == {"observation.state": 7, "action": 5}


# ---------------------------------------------------------------------------
# Behavior bullet 11 (-k widths): read_declared_widths over an info.json
# missing observation.state or action, or carrying an empty shape list,
# raises an error naming the file and the feature.
# ---------------------------------------------------------------------------


def test_widths_missing_observation_state_feature_raises(tmp_path):
    dataset_dir = _write_dataset_info(tmp_path, omit_state=True)
    with pytest.raises(ValueError, match="observation.state"):
        read_declared_widths(dataset_dir)


def test_widths_missing_action_feature_raises(tmp_path):
    dataset_dir = _write_dataset_info(tmp_path, omit_action=True)
    with pytest.raises(ValueError, match="action"):
        read_declared_widths(dataset_dir)


def test_widths_empty_shape_list_raises(tmp_path):
    dataset_dir = _write_dataset_info(tmp_path, empty_state_shape=True)
    with pytest.raises(ValueError, match="observation.state"):
        read_declared_widths(dataset_dir)


# ---------------------------------------------------------------------------
# Behavior bullet 12 (-k widths): observed widths equal to declared widths
# yield a BatchWidthReport whose match is true, and the emitted record's
# width block carries both sides.
# ---------------------------------------------------------------------------


def test_widths_matching_observed_and_declared_yields_match_true():
    declared = {"observation.state": 7, "action": 5}
    observed = {"observation.state": 7, "action": 5}
    report = assert_batch_widths(declared, observed)
    assert report.match is True
    assert report.declared == declared
    assert report.observed == observed
    assert report.mismatches == ()


def test_widths_emitted_record_carries_both_sides(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    dataset_dir = _write_dataset_info(tmp_path, state_width=7, action_width=5)
    main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-abc12345",
            "--dataset-dir",
            str(dataset_dir),
            "--observed-state-width",
            "7",
            "--observed-action-width",
            "5",
        ]
    )
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["batch_widths"]["declared"] == {"observation.state": 7, "action": 5}
    assert record["batch_widths"]["observed"] == {"observation.state": 7, "action": 5}


# ---------------------------------------------------------------------------
# Behavior bullet 13 (-k widths): an observed observation.state width one
# greater than declared yields match false, names observation.state in
# mismatches with both numbers, and drives a non-zero exit even when all
# four degeneracy assertions pass.
# ---------------------------------------------------------------------------


def test_widths_mismatch_yields_match_false_and_names_both_numbers():
    declared = {"observation.state": 7, "action": 5}
    observed = {"observation.state": 8, "action": 5}
    report = assert_batch_widths(declared, observed)
    assert report.match is False
    assert any(
        "observation.state" in m and "7" in m and "8" in m for m in report.mismatches
    ), report.mismatches


def test_widths_mismatch_drives_nonzero_exit_despite_healthy_degeneracy(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    dataset_dir = _write_dataset_info(tmp_path, state_width=7, action_width=5)
    exit_code = main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-abc12345",
            "--dataset-dir",
            str(dataset_dir),
            "--observed-state-width",
            "8",
            "--observed-action-width",
            "5",
        ]
    )
    assert exit_code != 0
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["degeneracy"]["passed"] is True
    assert record["batch_widths"]["match"] is False


# ---------------------------------------------------------------------------
# Behavior bullet 14 (-k widths): omitting the dataset directory or either
# observed width reports the comparison as BATCH_WIDTHS_UNEVALUATED, exits
# non-zero, and never records match as true.
# ---------------------------------------------------------------------------


def test_widths_omitting_dataset_dir_reports_unevaluated():
    report = assert_batch_widths(None, {"observation.state": 7, "action": 5})
    assert report.match is None


def test_widths_omitting_observed_width_reports_unevaluated():
    report = assert_batch_widths({"observation.state": 7, "action": 5}, None)
    assert report.match is None


def test_widths_omitting_dataset_dir_via_cli_exits_nonzero_and_match_not_true(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    exit_code = main(
        ["--trainer-state", str(trainer_state_path), "--revision", "rev-abc12345"]
    )
    assert exit_code != 0
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["batch_widths"]["match"] is not True
    assert BATCH_WIDTHS_UNEVALUATED in record["unevaluated"]


def test_widths_omitting_one_observed_width_via_cli_is_unevaluated(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    dataset_dir = _write_dataset_info(tmp_path, state_width=7, action_width=5)
    exit_code = main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-abc12345",
            "--dataset-dir",
            str(dataset_dir),
            "--observed-state-width",
            "7",
            # --observed-action-width intentionally omitted
        ]
    )
    assert exit_code != 0
    report_path = trainer_state_path.parent.parent / GR00T_VERIFY_REPORT_FILENAME
    record = json.loads(report_path.read_text(encoding="utf-8"))
    assert record["batch_widths"]["match"] is not True


# ---------------------------------------------------------------------------
# Behavior bullet 15: a run whose degeneracy assertions all pass AND whose
# widths match exits 0; that is the only combination that does.
# ---------------------------------------------------------------------------


def test_only_all_pass_and_widths_match_exits_zero(tmp_path):
    trainer_state_path = _write_trainer_state(tmp_path, _healthy_log_history())
    dataset_dir = _write_dataset_info(tmp_path, state_width=7, action_width=5)

    # All pass + widths match -> 0.
    exit_code = main(
        [
            "--trainer-state",
            str(trainer_state_path),
            "--revision",
            "rev-abc12345",
            "--dataset-dir",
            str(dataset_dir),
            "--observed-state-width",
            "7",
            "--observed-action-width",
            "5",
        ]
    )
    assert exit_code == 0

    # All pass but widths NOT supplied (unevaluated) -> nonzero.
    exit_code_no_widths = main(
        ["--trainer-state", str(trainer_state_path), "--revision", "rev-abc12345"]
    )
    assert exit_code_no_widths != 0


# ---------------------------------------------------------------------------
# Additional CLI-shape checks (not a behavior bullet, but an acceptance
# criterion): --help exits 0 and names the three required-argument flags.
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_names_required_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--trainer-state" in captured.out
    assert "--revision" in captured.out
    assert "--dataset-dir" in captured.out


def test_dataclasses_are_frozen():
    report = DegeneracyReport(passed=True, unevaluated=(), failures=(), steps=())
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]

    width_report = BatchWidthReport(match=True, declared={}, observed={}, mismatches=())
    with pytest.raises(Exception):
        width_report.match = False  # type: ignore[misc]
