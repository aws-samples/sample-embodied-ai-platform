"""`python -m data.tools.verify_gr00t_run --trainer-state <path> --revision <rev> ...`

VERIFY-02's mechanical grading of a real GR00T fine-tune run's own logged numbers
against D-14's four named degeneracy assertions, plus VERIFY-01's runnable
comparison of the observed batch widths against the widths the converted
dataset's own `meta/info.json` declares.

Stdlib plus `numpy` only. This module imports nothing under `gr00t` — it is
container-free and GPU-free by construction, so it can run from the repo-root
`.venv` where `gr00t` is never installed, both for its own unit tests and for
the real workstation run against a `trainer_state.json` on EFS.

D-15's declared boundary: VERIFY-01 (the converted dataset loads and yields
correctly-shaped batches) is the conversion contract and must pass
unconditionally. A degenerate action head on a high-DoF embodiment
(loss: 0.0 / grad_norm: NaN) is the known upstream initialization issue
GR00T#314 -- evidence about GR00T's own initialization, not evidence the
converter is wrong. This tool therefore records a degenerate run as a finding
with its measured numbers rather than treating it as a converter defect;
callers still see it in the non-zero exit code and the emitted record, and it
is named as an open item for a training-side phase, never silently absorbed
into a pass.

An unreadable or unrun signal (an absent `grad_norm` key across the whole
history, or a width comparison nobody supplied both sides for) is reported as
explicitly `*_UNEVALUATED` -- it never raises and it never counts as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GR00T_VERIFY_REPORT_FILENAME = "gr00t_verification.json"

GRAD_NORM_UNEVALUATED = "grad_norm_unevaluated"
BATCH_WIDTHS_UNEVALUATED = "batch_widths_unevaluated"

# D-14's four named degeneracy assertions, inspectable in one place -- the
# same convention as this project's other check registries (e.g.
# data/validation/registry.py's named-check lists).
DEGENERACY_ASSERTIONS: tuple[str, ...] = (
    "loss_positive_at_first_step",
    "grad_norm_positive",
    "no_nan_or_inf",
    "loss_not_bit_identical",
)

# D-15's boundary statement, recorded verbatim in every emitted record so a
# reader of gr00t_verification.json sees the same distinction the plan and
# this module's docstring make.
_D15_BOUNDARY_STATEMENT = (
    "VERIFY-01 (the dataset loads and yields correctly-shaped batches) is the "
    "conversion contract and must pass unconditionally. A degenerate action "
    "head on a high-DoF embodiment (loss: 0.0 / grad_norm: NaN) is the known "
    "upstream initialization issue GR00T#314 -- evidence about GR00T's own "
    "initialization, not evidence the converter is wrong. A degenerate step "
    "is therefore a recorded finding with its measured numbers, named as an "
    "open item for a training-side phase, and this phase still closes on "
    "VERIFY-01 plus a completed non-crashing step."
)

_FLOAT_KEY_DENYLIST = frozenset({"step", "epoch"})


@dataclass(frozen=True)
class DegeneracyReport:
    """The outcome of `assert_non_degenerate` over one `log_history`.

    `passed` is true only when every one of `DEGENERACY_ASSERTIONS` is
    affirmatively satisfied -- an empty `unevaluated` and an empty
    `failures`. `unevaluated` names an assertion this report could not even
    attempt (currently only `grad_norm_positive`, via `GRAD_NORM_UNEVALUATED`,
    when no entry in the whole history carries a `grad_norm` key at all).
    `failures` names an assertion that was attempted and did not hold, each
    string carrying the assertion name, the offending step, and the value.
    `steps` is every logged entry's raw numbers, unmodified.
    """

    passed: bool
    unevaluated: tuple[str, ...]
    failures: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BatchWidthReport:
    """VERIFY-01's declared-versus-observed batch width comparison.

    `match` is `True` only when every declared width equals its observed
    counterpart, `False` when at least one comparison was made and at least
    one disagreed, and `None` when the comparison was never run at all
    (either `--dataset-dir` or an observed width was omitted) -- `None` is
    the `BATCH_WIDTHS_UNEVALUATED` state, and it is never conflated with
    `True`.
    """

    match: bool | None
    declared: dict[str, int]
    observed: dict[str, int]
    mismatches: tuple[str, ...]


def read_log_history(path: str | Path) -> list[dict[str, Any]]:
    """Opens `path` (a `checkpoint-*/trainer_state.json`) and returns its
    `log_history` list verbatim -- no filtering, no reshaping. Raises
    `ValueError` naming the path when the file is absent or unparseable, and
    naming the missing key when `log_history` itself is absent."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"trainer_state.json not found: {resolved}")
    try:
        with open(resolved, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse trainer_state.json at {resolved}: {exc}") from exc

    if not isinstance(payload, dict) or "log_history" not in payload:
        raise ValueError(f"trainer_state.json at {resolved} has no 'log_history' key")

    log_history = payload["log_history"]
    if not isinstance(log_history, list):
        raise ValueError(
            f"trainer_state.json at {resolved} has a 'log_history' that is not a list"
        )
    return log_history


def _is_bad_float(value: Any) -> bool:
    """True when `value` is a float and is NaN or +/-Inf. Non-float values
    (e.g. an int `step`) are never flagged."""
    if not isinstance(value, float):
        return False
    return value != value or value in (float("inf"), float("-inf"))  # NaN != NaN


def assert_non_degenerate(log_history: list[dict[str, Any]]) -> DegeneracyReport:
    """D-14's four named degeneracy assertions over a real training run's
    own logged numbers -- no convergence threshold, no loss-must-decrease
    check. An empty `log_history` reports every assertion as unevaluated
    rather than vacuously passing zero assertions, per this project's own
    fail-loud convention (Phase 2 D-08 / Phase 3 D-12): zero assertions
    quietly passing is exactly the silently-green outcome that convention
    forbids."""
    steps = tuple(log_history)

    if not steps:
        # Zero assertions vacuously passing over zero steps is exactly the
        # quietly-green outcome this project's own convention forbids
        # (Phase 2 D-08 / Phase 3 D-12) -- every assertion is unevaluated,
        # never a trivial pass.
        return DegeneracyReport(
            passed=False,
            unevaluated=(
                "loss_positive_at_first_step",
                GRAD_NORM_UNEVALUATED,
                "no_nan_or_inf",
                "loss_not_bit_identical",
            ),
            failures=(),
            steps=steps,
        )

    failures: list[str] = []
    unevaluated: list[str] = []

    # Assertion 1: loss finite and strictly > 0 at the first logged entry
    # that carries a loss.
    first_loss_entry: dict[str, Any] | None = None
    first_loss_step: Any = None
    for entry in steps:
        if "loss" in entry:
            first_loss_entry = entry
            first_loss_step = entry.get("step")
            break
    if first_loss_entry is None:
        unevaluated.append("loss_positive_at_first_step")
    else:
        loss_value = first_loss_entry["loss"]
        if _is_bad_float(loss_value) or not isinstance(loss_value, (int, float)) or loss_value <= 0:
            failures.append(
                f"loss_positive_at_first_step: step={first_loss_step!r} loss={loss_value!r} "
                "is not finite and strictly > 0"
            )

    # Assertion 2: grad_norm finite and strictly > 0. When no entry anywhere
    # in the history carries a grad_norm key, this key's presence depends on
    # the transformers version pinned inside the container image, which no
    # static trace could confirm -- report GRAD_NORM_UNEVALUATED rather than
    # reading a key that is not there.
    grad_norm_seen = any("grad_norm" in entry for entry in steps)
    if not grad_norm_seen:
        unevaluated.append(GRAD_NORM_UNEVALUATED)
    else:
        for entry in steps:
            if "grad_norm" not in entry:
                continue
            grad_norm_value = entry["grad_norm"]
            if (
                _is_bad_float(grad_norm_value)
                or not isinstance(grad_norm_value, (int, float))
                or grad_norm_value <= 0
            ):
                failures.append(
                    f"grad_norm_positive: step={entry.get('step')!r} "
                    f"grad_norm={grad_norm_value!r} is not finite and strictly > 0"
                )
                break

    # Assertion 3: no NaN or Inf in any numeric field at any step.
    for entry in steps:
        for field_name, value in entry.items():
            if field_name in _FLOAT_KEY_DENYLIST:
                continue
            if _is_bad_float(value):
                failures.append(
                    f"no_nan_or_inf: step={entry.get('step')!r} field={field_name!r} "
                    f"value={value!r}"
                )

    # Assertion 4: loss is not bit-identical across every logged step --
    # compare the raw float values, never a rounded rendering.
    loss_values = [entry["loss"] for entry in steps if "loss" in entry]
    if len(loss_values) < 2:
        unevaluated.append("loss_not_bit_identical")
    elif all(v == loss_values[0] for v in loss_values):
        failures.append(
            f"loss_not_bit_identical: every logged loss equals {loss_values[0]!r} -- "
            "the optimizer did not move"
        )

    passed = not failures and not unevaluated
    return DegeneracyReport(
        passed=passed,
        unevaluated=tuple(unevaluated),
        failures=tuple(failures),
        steps=steps,
    )


def read_declared_widths(dataset_dir: str | Path) -> dict[str, int]:
    """Opens `<dataset_dir>/meta/info.json` and returns the trailing shape
    entry (`shape[-1]`) of `observation.state` and `action` -- derived,
    never restated as a literal, which is the exact mistake DERISK-FINDINGS
    pitfall (c) names. Raises `ValueError` naming the file and the feature
    when the feature is absent or its shape list is empty."""
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"info.json not found: {info_path}")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    declared: dict[str, int] = {}
    for feature_name in ("observation.state", "action"):
        feature = features.get(feature_name)
        if feature is None:
            raise ValueError(f"{info_path} has no {feature_name!r} feature")
        shape = feature.get("shape")
        if not shape:
            raise ValueError(f"{info_path}'s {feature_name!r} feature has an empty shape")
        declared[feature_name] = int(shape[-1])
    return declared


def assert_batch_widths(
    declared: dict[str, int] | None, observed: dict[str, int] | None
) -> BatchWidthReport:
    """Compares `declared` and `observed` by integer equality per feature
    name. When either side is `None` (the CLI's `--dataset-dir` or an
    observed width was omitted), does not compare and does not guess: `match`
    is `None`, never `True`, and never `False` either -- a comparison
    nobody ran must be visibly unrun, exactly as an absent `grad_norm` is."""
    if declared is None or observed is None:
        return BatchWidthReport(match=None, declared={}, observed={}, mismatches=())

    mismatches: list[str] = []
    for feature_name in sorted(set(declared) | set(observed)):
        declared_value = declared.get(feature_name)
        observed_value = observed.get(feature_name)
        if declared_value != observed_value:
            mismatches.append(
                f"{feature_name}: declared={declared_value!r} observed={observed_value!r}"
            )

    return BatchWidthReport(
        match=not mismatches,
        declared=dict(declared),
        observed=dict(observed),
        mismatches=tuple(mismatches),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data.tools.verify_gr00t_run",
        description=(
            "Grades a real GR00T fine-tune run's own logged numbers against "
            "D-14's four named degeneracy assertions, and compares the "
            "observed batch widths against the widths the converted "
            "dataset's own meta/info.json declares (VERIFY-01). Container-free, "
            "GPU-free -- runs from the repo-root .venv against a "
            "checkpoint-*/trainer_state.json on disk."
        ),
    )
    parser.add_argument(
        "--trainer-state",
        required=True,
        metavar="checkpoint-path",
        help="Path to a checkpoint-*/trainer_state.json written by the training run.",
    )
    parser.add_argument(
        "--revision",
        required=True,
        metavar="dataset-revision",
        help="The dataset revision identifier (CLI-09) the run consumed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="output-dir",
        help=(
            "Where to write gr00t_verification.json. Defaults to the "
            "trainer-state path's grandparent directory (the training run's "
            "OUTPUT_DIR)."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        metavar="dataset-prefix",
        help="The converted dataset's output prefix, whose meta/info.json declares the widths.",
    )
    parser.add_argument(
        "--observed-state-width",
        type=int,
        default=None,
        metavar="width",
        help="observation.state's trailing tensor dimension, read off the real job log.",
    )
    parser.add_argument(
        "--observed-action-width",
        type=int,
        default=None,
        metavar="width",
        help="action's trailing tensor dimension, read off the real job log.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        log_history = read_log_history(args.trainer_state)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    degeneracy = assert_non_degenerate(log_history)

    declared_widths: dict[str, int] | None = None
    if args.dataset_dir is not None:
        try:
            declared_widths = read_declared_widths(args.dataset_dir)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    observed_widths: dict[str, int] | None = None
    if args.observed_state_width is not None and args.observed_action_width is not None:
        observed_widths = {
            "observation.state": args.observed_state_width,
            "action": args.observed_action_width,
        }

    width_report = assert_batch_widths(declared_widths, observed_widths)

    trainer_state_path = Path(args.trainer_state)
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else trainer_state_path.parent.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # A run-level unevaluated list combines both unreadable-signal sources
    # (an absent grad_norm key, or a width comparison nobody ran) -- neither
    # is ever silently absorbed into a pass.
    run_unevaluated = list(degeneracy.unevaluated)
    if width_report.match is None:
        run_unevaluated.append(BATCH_WIDTHS_UNEVALUATED)

    record: dict[str, Any] = {
        "dataset_revision": args.revision,
        "trainer_state_path": str(trainer_state_path),
        "unevaluated": run_unevaluated,
        "degeneracy": {
            "passed": degeneracy.passed,
            "unevaluated": list(degeneracy.unevaluated),
            "failures": list(degeneracy.failures),
            "assertions": list(DEGENERACY_ASSERTIONS),
        },
        "steps": [dict(step) for step in degeneracy.steps],
        "batch_widths": {
            "match": width_report.match,
            "declared": width_report.declared,
            "observed": width_report.observed,
            "mismatches": list(width_report.mismatches),
        },
        "documentation": _D15_BOUNDARY_STATEMENT,
    }

    final_path = output_dir / GR00T_VERIFY_REPORT_FILENAME
    tmp_path = output_dir / f".{GR00T_VERIFY_REPORT_FILENAME}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True, indent=2)
    os.replace(tmp_path, final_path)

    print(str(final_path))

    success = degeneracy.passed and not run_unevaluated and width_report.match is True
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
