"""The single inspectable post-write validation check registry (D-01),
structurally mirroring `data/adapters/checks.py`'s `Check`/`CheckRegistry` —
same "declared order in, same order out" run methods — but independently
defined under new names. This module imports `CheckResult`/`Disposition`
from `data/contracts/containers.py` and nothing at all from the ingest check
package (`data/adapters/checks.py`): D-01 rejects a shared `stage` axis, so
the two registries are parallel, never one importing the other.

Severity semantics for the post-write lifecycle, documented once, here:
a failing `fatal` check drives the owning episode's (or the dataset's)
disposition to `REJECT` and makes the run exit non-zero; a failing
`advisory` check downgrades a `PASS` to `PASS_WITH_FLAGS` and never on its
own changes the exit code. A result carrying `UNEVALUATED_KEY` true is
neither: it never counts toward a disposition downgrade and never counts as
a pass — it sets the report's `fully_validated` flag false instead.
"""

import dataclasses
from typing import Callable, Literal

from data.contracts.containers import CheckResult

UNEVALUATED_KEY = "unevaluated"
"""Set `True` in a `CheckResult.evidence` mapping when a check had no basis
for a verdict (e.g. a zero-episode dataset, a zero-row feature) — never
silently counted as a pass, never as a fatal failure."""

SKIP_REASON_KEY = "skip_reason"
"""Paired with `UNEVALUATED_KEY` — names, in evidence, why a check could not
be evaluated."""


@dataclasses.dataclass(frozen=True)
class ValidationCheck:
    """One registered post-write check: its dotted name, severity, scope,
    and callable. `fn`'s call signature depends on `scope`: an
    `"episode"`-scoped check is called `fn(ctx, episode_index)`; a
    `"dataset"`-scoped check is called `fn(ctx)`."""

    name: str
    severity: Literal["fatal", "advisory"]
    scope: Literal["episode", "dataset"]
    fn: Callable[..., tuple[CheckResult, ...]]


@dataclasses.dataclass(frozen=True)
class ValidationRegistry:
    """Checks run in the declared, stable order below — `episode_scoped`/
    `dataset_scoped` preserve that order, so two runs over the same fixture
    produce identical check sequences."""

    checks: tuple[ValidationCheck, ...] = dataclasses.field(default_factory=tuple)

    @property
    def episode_scoped(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.scope == "episode")

    @property
    def dataset_scoped(self) -> tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.scope == "dataset")

    @classmethod
    def default(cls) -> "ValidationRegistry":
        """Registers `videos.frame_count_agreement` (VALID-01, Task 1) plus
        the three dataset-scoped read-back checks (VALID-06/07/10, Task 3),
        followed by plan 03-03's two decode-based content checks —
        `videos.spot_decode_content` (VALID-02) and
        `videos.decoded_pixel_uniqueness` (D-10) — then plan 03-04's three
        numeric checks — `consistency.action_state_lag` (VALID-03),
        `statistics.constant_channel` (VALID-04) and
        `statistics.stats_recomputation` (VALID-09) — then this plan's two
        cross-artifact modality checks — `consistency.modality_slice_range`
        (VALID-05) and `consistency.modality_agreement` (VALID-08) — and
        `regression.golden_fingerprint` (REGR-01), and finally
        `03-09-PLAN.md`'s `statistics.image_histogram_integrity` (WINDOWS.md
        id 2) — appended after them so the declared run order stays stable.
        `diff_datasets` (REGR-02) is exported but deliberately NOT
        registered here — it compares two prefixes rather than validating
        one, so it has no `ValidationCheck` shape to fit. Imports are
        deferred into this method body (rather than module-level) so
        `frame_integrity.py`/`inventory.py`/`correlation.py`/
        `constant_channels.py`/`stats_oracle.py`/`modality_agreement.py`/
        `fingerprint.py`/`image_histogram.py` — which import this module's
        `UNEVALUATED_KEY`/`SKIP_REASON_KEY` at their own module level —
        never form an import cycle with this module."""
        from data.validation.constant_channels import check_constant_channel
        from data.validation.correlation import check_action_state_lag
        from data.validation.fingerprint import check_golden_fingerprint
        from data.validation.frame_integrity import (
            check_decoded_pixel_uniqueness,
            check_frame_count_agreement,
            check_spot_decode_content,
        )
        from data.validation.image_histogram import check_image_histogram_integrity
        from data.validation.inventory import (
            check_episode_length_agreement,
            check_nan_inf,
            check_task_index_join,
        )
        from data.validation.modality_agreement import (
            check_modality_agreement,
            check_modality_slice_range,
        )
        from data.validation.stats_oracle import check_stats_recomputation

        return cls(
            (
                ValidationCheck(
                    "videos.frame_count_agreement",
                    "fatal",
                    "episode",
                    check_frame_count_agreement,
                ),
                ValidationCheck(
                    "metadata.episode_length_agreement",
                    "fatal",
                    "dataset",
                    check_episode_length_agreement,
                ),
                ValidationCheck("anomalies.nan_inf", "fatal", "dataset", check_nan_inf),
                ValidationCheck(
                    "metadata.task_index_join", "fatal", "dataset", check_task_index_join
                ),
                ValidationCheck(
                    "videos.spot_decode_content",
                    "fatal",
                    "episode",
                    check_spot_decode_content,
                ),
                ValidationCheck(
                    "videos.decoded_pixel_uniqueness",
                    "fatal",
                    "episode",
                    check_decoded_pixel_uniqueness,
                ),
                ValidationCheck(
                    "consistency.action_state_lag",
                    "fatal",
                    "episode",
                    check_action_state_lag,
                ),
                ValidationCheck(
                    "statistics.constant_channel",
                    "advisory",
                    "dataset",
                    check_constant_channel,
                ),
                ValidationCheck(
                    "statistics.stats_recomputation",
                    "fatal",
                    "dataset",
                    check_stats_recomputation,
                ),
                ValidationCheck(
                    "consistency.modality_slice_range",
                    "fatal",
                    "dataset",
                    check_modality_slice_range,
                ),
                ValidationCheck(
                    "consistency.modality_agreement",
                    "fatal",
                    "dataset",
                    check_modality_agreement,
                ),
                ValidationCheck(
                    "regression.golden_fingerprint",
                    "fatal",
                    "dataset",
                    check_golden_fingerprint,
                ),
                ValidationCheck(
                    "statistics.image_histogram_integrity",
                    "fatal",
                    "dataset",
                    check_image_histogram_integrity,
                ),
            )
        )
