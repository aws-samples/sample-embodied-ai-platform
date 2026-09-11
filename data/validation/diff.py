"""REGR-02/D-19: schema and inventory diff between two output prefixes.

`diff_datasets(prefix_a, prefix_b)` is an exported function, deliberately
**not** a registered check: REGR-02 compares two prefixes rather than
validating one, and Phase 4's promotion rule is what will call it against a
previous release. A previous release *IS* a directory, which is why no
stored baseline artifact is needed here, unlike REGR-01's golden-fingerprint
machinery (`fingerprint.py`).

**Equality is the absence of a key.** Never emit a key carrying two equal
values: a diff whose output is non-empty must mean something changed, or a
reader learns to ignore it. A missing `meta/info.json` or
`meta/episodes.jsonl` on either side is reported as a named-reason entry —
never an empty mapping, which would read as "no changes" and is the single
most dangerous output shape this function could produce.

Every string key (feature names, camera keys) is compared as exact UTF-8 —
never a case-folded comparison — and every set difference is reported with
BOTH sides sorted, so the diff is insensitive to either document's own key
order. `fps` is compared by exact float equality on the value
`meta/info.json` carries — no tolerance absorbs a real clock-rate change.

Deliberately no generic deep-diff library: a full-JSON-tree diff reports
ordering artifacts and keys this requirement does not ask about, drowning
the fields that matter. This reports exactly the fields REGR-02 names.
"""

import json
from pathlib import Path
from typing import Any

DIFFED_INFO_KEYS: tuple[str, ...] = (
    "fps",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "total_videos",
    "codebase_version",
    "robot_type",
)
"""The scalar `meta/info.json` keys compared once, declared here so the set
is never typed twice."""

_VIDEO_INFO_FIELDS: tuple[str, ...] = (
    "video.codec",
    "video.pix_fmt",
    "video.height",
    "video.width",
    "video.fps",
)
"""The `video_info` sub-fields compared for a camera feature, in addition
to the generic `dtype`/`shape` fields every feature carries."""


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_episode_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            lengths.append(int(json.loads(stripped)["length"]))
    return lengths


def _length_distribution(lengths: list[int]) -> dict[str, Any]:
    sorted_lengths = sorted(lengths)
    return {
        "lengths": sorted_lengths,
        "count": len(sorted_lengths),
        "min": sorted_lengths[0] if sorted_lengths else None,
        "max": sorted_lengths[-1] if sorted_lengths else None,
        "sum": sum(sorted_lengths),
    }


def _camera_keys(features: dict[str, Any]) -> set[str]:
    return {name for name, feat in features.items() if feat.get("dtype") == "video"}


def _first_provenance_field(prefix: Path, field: str) -> Any:
    """Reads `field` from the first (lowest-indexed)
    `meta/data_engine/episode_*.json` provenance record found under
    `prefix` — `pipeline_version`/`profile_hash` are constant across every
    episode one conversion run wrote, so any single record's value
    represents the whole prefix. Returns `None` if no provenance record
    exists (never raises — a missing provenance directory is reported as
    an absent value, not a crash)."""
    data_engine_dir = prefix / "meta" / "data_engine"
    if not data_engine_dir.is_dir():
        return None
    for path in sorted(data_engine_dir.glob("episode_*.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        return payload.get(field)
    return None


def diff_datasets(prefix_a: Path, prefix_b: Path) -> dict[str, Any]:
    """Diffs two already-written LeRobot v2.1 dataset prefixes on schema
    and inventory. Returns `{}` when nothing differs; every other returned
    shape means something genuinely changed. Reads `meta/info.json` and
    `meta/episodes.jsonl` from BOTH prefixes and the provenance records
    under `meta/data_engine/` for the pipeline version and profile hash
    each prefix was written under.

    A missing/unreadable `meta/info.json` or `meta/episodes.jsonl` on
    either side short-circuits with a single `{"error": ...}` entry naming
    which file, on which side — never an empty mapping.
    """
    prefix_a, prefix_b = Path(prefix_a), Path(prefix_b)

    loaded: dict[str, Any] = {}
    for label, prefix in (("a", prefix_a), ("b", prefix_b)):
        info_path = prefix / "meta" / "info.json"
        episodes_path = prefix / "meta" / "episodes.jsonl"
        try:
            loaded[f"info_{label}"] = _read_json(info_path)
        except (OSError, json.JSONDecodeError) as e:
            return {
                "error": (
                    f"meta/info.json missing or unreadable on side {label!r} "
                    f"({info_path}): {type(e).__name__}: {e}"
                )
            }
        try:
            loaded[f"lengths_{label}"] = _read_episode_lengths(episodes_path)
        except (OSError, json.JSONDecodeError) as e:
            return {
                "error": (
                    f"meta/episodes.jsonl missing or unreadable on side {label!r} "
                    f"({episodes_path}): {type(e).__name__}: {e}"
                )
            }

    info_a, info_b = loaded["info_a"], loaded["info_b"]
    lengths_a, lengths_b = loaded["lengths_a"], loaded["lengths_b"]

    diff: dict[str, Any] = {}

    for key in DIFFED_INFO_KEYS:
        value_a, value_b = info_a.get(key), info_b.get(key)
        if value_a != value_b:
            diff[key] = {"a": value_a, "b": value_b}

    pipeline_version_a = _first_provenance_field(prefix_a, "pipeline_version")
    pipeline_version_b = _first_provenance_field(prefix_b, "pipeline_version")
    if pipeline_version_a != pipeline_version_b:
        diff["pipeline_version"] = {"a": pipeline_version_a, "b": pipeline_version_b}

    profile_hash_a = _first_provenance_field(prefix_a, "profile_hash")
    profile_hash_b = _first_provenance_field(prefix_b, "profile_hash")
    if profile_hash_a != profile_hash_b:
        diff["profile_hash"] = {"a": profile_hash_a, "b": profile_hash_b}

    features_a = info_a.get("features", {}) or {}
    features_b = info_b.get("features", {}) or {}

    camera_a, camera_b = _camera_keys(features_a), _camera_keys(features_b)
    if camera_a != camera_b:
        diff["camera_set"] = {"a": sorted(camera_a), "b": sorted(camera_b)}

    names_a, names_b = set(features_a), set(features_b)
    if names_a != names_b:
        diff["feature_names"] = {"a": sorted(names_a), "b": sorted(names_b)}

    feature_changes: dict[str, Any] = {}
    for name in sorted(names_a & names_b):
        feat_a, feat_b = features_a[name], features_b[name]
        field_diffs: dict[str, Any] = {}

        for field in ("dtype", "shape"):
            if feat_a.get(field) != feat_b.get(field):
                field_diffs[field] = {"a": feat_a.get(field), "b": feat_b.get(field)}

        if name in camera_a or name in camera_b:
            video_info_a = feat_a.get("video_info", {}) or {}
            video_info_b = feat_b.get("video_info", {}) or {}
            video_field_diffs: dict[str, Any] = {}
            for vf in _VIDEO_INFO_FIELDS:
                va, vb = video_info_a.get(vf), video_info_b.get(vf)
                if va != vb:
                    video_field_diffs[vf] = {"a": va, "b": vb}
            if video_field_diffs:
                field_diffs["video_info"] = video_field_diffs

        if field_diffs:
            feature_changes[name] = field_diffs

    if feature_changes:
        diff["feature_changes"] = feature_changes

    distribution_a = _length_distribution(lengths_a)
    distribution_b = _length_distribution(lengths_b)
    if distribution_a != distribution_b:
        diff["length_distribution"] = {"a": distribution_a, "b": distribution_b}

    return diff
