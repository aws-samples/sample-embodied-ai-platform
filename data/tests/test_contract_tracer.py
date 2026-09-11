"""End-to-end tracer test.

One path through every layer this phase touches: a YAML profile validates into an
`EmbodimentProfile`, a class inheriting from nothing satisfies `BaseInputAdapter`
under `mypy`, and consuming its `CanonicalEpisode` once yields exactly one frozen
`CanonicalFrame` whose state width comes from the profile's declared joint names
and whose image stays undecoded until a consumer asks for it.
"""

import dataclasses
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from data.contracts import (
    BaseInputAdapter,
    CanonicalFrame,
    EmbodimentProfile,
    EpisodeHeader,
    ImageRef,
    CanonicalEpisode,
    load_profile,
)

PROFILE_PATH = Path(__file__).parent.parent / "profiles" / "galaxea_r1_pro.yaml"


class TracerImageRef:
    """No-inheritance `ImageRef` stub: `decode()` returns a tiny fixed image.

    Neither this class nor `TracerAdapter` below inherits from anything — that
    absence is the CONT-02 evidence that structural typing needs no base class.
    """

    def decode(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    @property
    def timestamp_ns(self) -> int:
        return 0


class TracerAdapter:
    """No-inheritance `BaseInputAdapter` stub: yields exactly one frame."""

    def parse(self, path: str, profile: EmbodimentProfile) -> CanonicalEpisode:
        joint_names = profile.state_topics["arm_left"].joint_names
        header = EpisodeHeader(
            episode_id="tracer-episode",
            fps=25.0,
            robot_type=profile.robot_type,
            task_prompt="tracer",
        )

        def _frames() -> Iterator[CanonicalFrame]:
            yield CanonicalFrame(
                timestamp_ns=0,
                frame_index=0,
                images={"head_left": TracerImageRef()},
                state={"left_arm": np.zeros(len(joint_names), dtype=np.float32)},
                action={},
            )

        return CanonicalEpisode(header=header, frames=_frames())


def test_tracer_end_to_end() -> None:
    profile = load_profile(str(PROFILE_PATH))

    # These annotated assignments are the CONT-02 conformance proof: mypy checks
    # them, not a runtime isinstance() call.
    adapter: BaseInputAdapter = TracerAdapter()
    ref: ImageRef = TracerImageRef()

    assert TracerAdapter.__mro__ == (TracerAdapter, object)
    assert TracerImageRef.__mro__ == (TracerImageRef, object)

    episode = adapter.parse("<unused>", profile)
    frames = list(episode.frames)

    # (a) exactly one frame
    assert len(frames) == 1

    # (b) the single-pass hazard, pinned: a second iteration over the same
    # CanonicalEpisode.frames yields nothing.
    assert list(episode.frames) == []

    # (c) emitted state width is derived from the profile's declared joint names,
    # never from a literal.
    assert frames[0].state["left_arm"].shape[0] == len(
        profile.state_topics["arm_left"].joint_names
    )

    # (d) decode is deferred until a consumer asks.
    assert frames[0].images["head_left"].decode().shape == (2, 2, 3)
    assert ref.decode().shape == (2, 2, 3)

    # (e) frozen — mutation raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        frames[0].timestamp_ns = 1  # type: ignore[misc]
