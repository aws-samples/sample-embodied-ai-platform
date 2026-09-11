"""Exact per-feature statistics: no approximation anywhere (D-04).

For the `float32` features (`observation.state`, `action`, `timestamp`),
`WelfordPartial` retains every sample AND maintains a running mean and
second-moment term via Welford's online algorithm — never a naive
`sum`/`sum_of_squares` accumulator, whose `Var = E[X^2] - E[X]^2` identity
catastrophically cancels for large sums of similar-magnitude floats. Retained
values are what make `q01`/`q99` *exact* quantiles rather than an estimate:
roughly the emitted width in float32 times the frame count is under 100 KB
per episode for the shipped Galaxea profile (D-04).

**Memory ceiling, stated explicitly (D-05):**
`retained_value_bytes_per_episode(width, frame_count) * episode_count`
bytes — no silent mode switch, no size-dependent accuracy. Production
retrieves episodes dynamically and statistics get adjusted before training
regardless, so the at-scale retention strategy is a later-phase decision,
not a Phase 2 problem to pre-solve; Phase 2 converts a handful of sample
episodes.

For uint8 image channels, `Uint8Histogram` maintains a 256-bin-per-channel
histogram — *exact*, not approximate, because uint8 has exactly 256
possible values. Image statistics are computed and retained (in the
per-episode partial file) for future use but are deliberately **not**
emitted into `meta/stats.json`: GR00T's own `generate_stats`/
`check_stats_validity` iterate features by a substring test for `"float"`
against the `info.json` dtype string, so a `video`-dtype camera feature is
excluded from its shape check by construction and would be silent garbage
in that file.

`merge_partials` implements Chan's parallel-variance combination so a
dataset-wide merge over per-episode partials never re-scans a parquet file,
re-decodes a video, or re-reads an MCAP (D-06). Naively averaging
per-episode standard deviations is a documented real-world failure
(near-zero normalized actions) — this merge is what avoids it.
"""

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99", "count")
"""The seven statistics `finalize_stats` emits per feature."""


def retained_value_bytes_per_episode(width: int, frame_count: int, dtype_bytes: int = 4) -> int:
    """D-05's explicit memory-ceiling expression for one float feature's
    retained-value buffer over one episode: `width * dtype_bytes *
    frame_count` bytes. The dataset-wide ceiling is this times the number
    of episodes converted."""
    return width * dtype_bytes * frame_count


@dataclasses.dataclass
class WelfordPartial:
    """A mergeable per-feature accumulator: `count`, the running `mean`,
    the second-moment term (`m2`, Welford's `M2`), `min`, `max`, and every
    retained raw sample (`values`, shape `(count, width)`). `sum` is
    derived (`mean * count`) rather than tracked separately — Welford's
    update is defined in terms of the running mean, and re-deriving `sum`
    from it costs nothing and cannot drift from it.
    """

    count: int
    mean: np.ndarray
    m2: np.ndarray
    min: np.ndarray
    max: np.ndarray
    values: np.ndarray

    @classmethod
    def empty(cls, width: int) -> "WelfordPartial":
        return cls(
            count=0,
            mean=np.zeros(width, dtype=np.float64),
            m2=np.zeros(width, dtype=np.float64),
            min=np.full(width, np.inf, dtype=np.float64),
            max=np.full(width, -np.inf, dtype=np.float64),
            values=np.zeros((0, width), dtype=np.float32),
        )

    def update(self, sample: np.ndarray) -> None:
        """Welford's online update with one new `(width,)` sample."""
        width = self.mean.shape[0]
        sample64 = np.asarray(sample, dtype=np.float64).reshape(width)
        self.count += 1
        delta = sample64 - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = sample64 - self.mean
        self.m2 = self.m2 + delta * delta2
        self.min = np.minimum(self.min, sample64)
        self.max = np.maximum(self.max, sample64)
        self.values = np.concatenate(
            [self.values, np.asarray(sample, dtype=np.float32).reshape(1, width)], axis=0
        )

    def to_json_dict(self) -> dict[str, Any]:
        sum_ = (self.mean * self.count).tolist()
        return {
            "count": self.count,
            "sum": sum_,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def from_json_dict(
        cls, d: Mapping[str, Any], values: np.ndarray | None = None
    ) -> "WelfordPartial":
        mean = np.array(d["mean"], dtype=np.float64)
        width = mean.shape[0]
        return cls(
            count=int(d["count"]),
            mean=mean,
            m2=np.array(d["m2"], dtype=np.float64),
            min=np.array(d["min"], dtype=np.float64),
            max=np.array(d["max"], dtype=np.float64),
            values=values if values is not None else np.zeros((0, width), dtype=np.float32),
        )


def merge_partials(a: WelfordPartial, b: WelfordPartial) -> WelfordPartial:
    """Chan's parallel-variance combination of two per-feature partials.
    Returns the other partial UNCHANGED when either count is zero — never
    divides by a zero combined count and never mutates the zero-count
    input."""
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n_a, n_b = a.count, b.count
    n_ab = n_a + n_b
    delta = b.mean - a.mean
    mean_ab = a.mean + delta * (n_b / n_ab)
    m2_ab = a.m2 + b.m2 + (delta**2) * (n_a * n_b / n_ab)
    return WelfordPartial(
        count=n_ab,
        mean=mean_ab,
        m2=m2_ab,
        min=np.minimum(a.min, b.min),
        max=np.maximum(a.max, b.max),
        values=np.concatenate([a.values, b.values], axis=0),
    )


def _rank_quantile(sorted_values: np.ndarray, q: float, method: str) -> np.ndarray:
    """The `n`-th smallest-element index used by both the retained-value
    exact quantile (`finalize_stats`) and the uint8 histogram exact
    quantile (`Uint8Histogram.quantile`) — the same rank definition in both
    places, so a channel's quantile means the same thing regardless of
    whether it came from retained float samples or a uint8 histogram.

    For `method="lower"`: floor(q * (n - 1)); for `method="higher"`:
    ceil(q * (n - 1)). On exactly two samples this yields precisely the
    lower sample for `q=0.01` and precisely the upper sample for `q=0.99`.
    """
    n = sorted_values.shape[0]
    if n == 0:
        return np.zeros(sorted_values.shape[1:], dtype=sorted_values.dtype)
    h = q * (n - 1)
    k = math.floor(h) if method == "lower" else math.ceil(h)
    k = min(max(k, 0), n - 1)
    return sorted_values[k]


def finalize_stats(partial: WelfordPartial) -> dict[str, list[float]]:
    """Computes the seven `STATS_KEYS` statistics from a (merged or
    single-episode) partial. Raises on a zero-count partial — there is no
    meaningful mean/std/quantile for zero samples, and this must never be
    silently reported as a passing/valid entry."""
    if partial.count == 0:
        raise ValueError("cannot finalize statistics for a zero-count partial")
    variance = partial.m2 / partial.count
    std = np.sqrt(variance)
    sorted_values = np.sort(partial.values, axis=0)
    q01 = _rank_quantile(sorted_values, 0.01, "lower")
    q99 = _rank_quantile(sorted_values, 0.99, "higher")
    return {
        "mean": partial.mean.tolist(),
        "std": std.tolist(),
        "min": partial.min.tolist(),
        "max": partial.max.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "count": [partial.count],
    }


@dataclasses.dataclass
class Uint8Histogram:
    """An exact 256-bin-per-channel histogram over uint8 image samples.
    `counts` has shape `(channels, 256)`; exact because uint8 has exactly
    256 possible values, unlike a streaming quantile estimator over an
    unbounded value range."""

    counts: np.ndarray

    @classmethod
    def empty(cls, channels: int) -> "Uint8Histogram":
        return cls(counts=np.zeros((channels, 256), dtype=np.int64))

    def update(self, image: np.ndarray) -> None:
        """`image` is `(H, W, C)` uint8; every pixel of every channel is
        counted."""
        channels = image.shape[-1]
        for c in range(channels):
            values, counts = np.unique(image[..., c], return_counts=True)
            self.counts[c, values] += counts

    def merge(self, other: "Uint8Histogram") -> "Uint8Histogram":
        return Uint8Histogram(counts=self.counts + other.counts)

    def quantile(self, q: float, method: str) -> np.ndarray:
        result = np.zeros(self.counts.shape[0], dtype=np.float64)
        for c in range(self.counts.shape[0]):
            n = int(self.counts[c].sum())
            if n == 0:
                continue
            h = q * (n - 1)
            k = math.floor(h) if method == "lower" else math.ceil(h)
            k = min(max(k, 0), n - 1)
            cumsum = np.cumsum(self.counts[c])
            result[c] = float(np.searchsorted(cumsum, k, side="right"))
        return result


def disposition_included_in_merge(disposition: Any) -> bool:
    """REJECT is excluded from the dataset-wide merge; PASS_WITH_FLAGS and
    PASS are included. `None` (no disposition recorded — e.g. a synthetic
    fixture with no adapter present) defaults to included."""
    if disposition is None:
        return True
    value = disposition.value if hasattr(disposition, "value") else disposition
    return value != "REJECT"


def write_partial_file(
    path: Path,
    *,
    disposition: Any,
    included_in_merge: bool,
    feature_partials: Mapping[str, WelfordPartial],
    histograms: Mapping[str, Uint8Histogram] | None = None,
) -> None:
    """Persists one episode's mergeable partial alongside the episode —
    count, sum, mean, the second-moment term, min, max, the retained
    values, and the image histograms — so a later merge recomputation
    needs no parquet scan, no video re-decode, and no MCAP re-read. Also
    records which disposition policy was actually applied, so two runs
    over the same data cannot silently disagree about which episodes
    contributed to the dataset-wide merge."""
    disposition_value = disposition.value if hasattr(disposition, "value") else disposition
    payload = {
        "disposition": disposition_value,
        "included_in_merge": included_in_merge,
        "features": {
            name: {**p.to_json_dict(), "values": p.values.tolist()}
            for name, p in feature_partials.items()
        },
        "histograms": (
            {name: h.counts.tolist() for name, h in histograms.items()} if histograms else {}
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
