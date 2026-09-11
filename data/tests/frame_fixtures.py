"""Synthetic `CanonicalEpisode` builder for Track B's writer tests.

Not a `test_*.py` module — pytest does not collect it; importable as
`data.tests.frame_fixtures`. Every array width here is derived from the
`EmbodimentProfile` handed to `build_synthetic_episode`, never a numeric
literal, so a fixture is automatically correct for any profile and this
module needs no edit if the shipped Galaxea profile's widths ever change.

Deliberately has ZERO dependency on `data.adapters` — Track B's writer must
be provable with no working adapter present. Do not import
`data.adapters.galaxea` or reuse `data/tests/mcap_fixture.py` from this
module.
"""

import dataclasses
from typing import Iterator, Mapping, Sequence

import numpy as np
import simplejpeg

from data.contracts.containers import CanonicalEpisode, CanonicalFrame, EpisodeHeader, ImageRef
from data.contracts.profile import EmbodimentProfile
from data.decoders import get_decoder_backend


@dataclasses.dataclass(frozen=True)
class FixtureImageRef:
    """Satisfies `ImageRef` (data/contracts/containers.py) structurally —
    no import from, no inheritance of, that module. Real JPEG bytes are
    encoded once by the caller and shared across every frame/camera that
    reuses this ref, so `decode()` always returns real pixels via the
    configured decoder backend (`data/decoders.py`), never a stub array.
    """

    data: bytes
    timestamp_ns: int

    def decode(self) -> np.ndarray:
        return get_decoder_backend().decode(self.data)


def _build_gradient_jpeg(width: int, height: int) -> bytes:
    """Encodes one small deterministic RGB gradient image to real JPEG
    bytes, once, for the caller to reuse across every frame and camera in
    a synthetic episode."""
    x_ramp = np.linspace(0, 255, width, dtype=np.uint8)
    y_ramp = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.tile(x_ramp, (height, 1))
    green = np.tile(y_ramp.reshape(-1, 1), (1, width))
    blue = np.full((height, width), 128, dtype=np.uint8)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return simplejpeg.encode_jpeg(rgb, colorspace="rgb")


def _state_layout(profile: EmbodimentProfile) -> list[tuple[str, int]]:
    """`(state_key, width)` pairs in `profile.state_vector_order`'s order —
    the sole width authority; never a literal in this module."""
    by_state_key = {f.state_key: f for f in profile.state_topics.values()}
    return [
        (state_key, len(by_state_key[state_key].joint_names))
        for state_key in profile.state_vector_order
    ]


def _action_layout(profile: EmbodimentProfile) -> list[tuple[str, int]]:
    """`(action_key, width)` pairs in `profile.action_primary`'s order."""
    by_action_key = {f.action_key: f for f in profile.action_topics.values()}
    return [
        (action_key, len(by_action_key[action_key].joint_names))
        for action_key in profile.action_primary
    ]


def build_synthetic_episode(
    *,
    frame_count: int,
    fps: float,
    profile: EmbodimentProfile,
    state_values: Mapping[str, Sequence[np.ndarray]] | None = None,
    action_values: Mapping[str, Sequence[np.ndarray]] | None = None,
    image_size: tuple[int, int] = (64, 48),
    task_prompt: str | None = None,
    disposition: object | None = None,
    emit_cameras: Sequence[str] | None = None,
) -> CanonicalEpisode:
    """Builds a `CanonicalEpisode` of `frame_count` frames, with state/action
    dicts keyed by the profile's own `state_vector_order`/`action_primary`
    entries and one `ImageRef` per requested camera per frame.

    `state_values`/`action_values`, when given, override the default
    deterministic per-frame values so a test can assert exact numbers: each
    is a mapping from the profile's key (state_key / action_key) to a
    per-frame sequence of `(width,)` arrays, one entry per frame.

    `emit_cameras`, when given, is the subset of `video_key`s to populate
    `CanonicalFrame.images` with (each must be an `emit: true` camera's
    `video_key`) — lets a test wire exactly one camera end to end (the
    tracer's scenario) without needing a second, minimal profile. Defaults
    to every `emit: true` camera in `profile`.

    `frames` is a generator — iterating the returned episode's `frames` a
    second time yields nothing, matching `CanonicalEpisode`'s single-pass
    contract.
    """
    width_px, height_px = image_size
    jpeg_bytes = _build_gradient_jpeg(width_px, height_px)

    all_emitted_video_keys = [
        cam.video_key
        for cam in profile.cameras.values()
        if cam.emit and cam.video_key is not None
    ]
    emitted_video_keys = (
        list(emit_cameras) if emit_cameras is not None else all_emitted_video_keys
    )

    state_layout = _state_layout(profile)
    action_layout = _action_layout(profile)

    def _frames() -> Iterator[CanonicalFrame]:
        for i in range(frame_count):
            ts_ns = int(round(i * (1e9 / fps))) if fps > 0 else i

            state: dict[str, np.ndarray] = {}
            for key, width in state_layout:
                if state_values is not None and key in state_values:
                    arr = np.asarray(state_values[key][i], dtype=np.float32).reshape(width)
                else:
                    arr = np.full(width, float(i), dtype=np.float32)
                state[key] = arr

            action: dict[str, np.ndarray] = {}
            for key, width in action_layout:
                if action_values is not None and key in action_values:
                    arr = np.asarray(action_values[key][i], dtype=np.float32).reshape(width)
                else:
                    arr = np.full(width, float(i) + 0.5, dtype=np.float32)
                action[key] = arr

            images: dict[str, ImageRef] = {
                video_key: FixtureImageRef(data=jpeg_bytes, timestamp_ns=ts_ns)
                for video_key in emitted_video_keys
            }

            yield CanonicalFrame(
                timestamp_ns=ts_ns,
                frame_index=i,
                images=images,
                state=state,
                action=action,
                task_prompt=task_prompt,
            )

    global_meta: dict[str, object] = {}
    if disposition is not None:
        global_meta["disposition"] = disposition

    header = EpisodeHeader(
        episode_id="synthetic-episode",
        fps=fps,
        robot_type=profile.robot_type,
        task_prompt=task_prompt,
        frame_count=frame_count,
        global_meta=global_meta,
    )
    return CanonicalEpisode(header=header, frames=_frames())
