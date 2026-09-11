"""D-11's run-config schema and loader: the YAML file operators pass via
`--run-config`, carrying batch knobs (input discovery, output prefix, format
key, codec, task string, fps override, `emit_episodes_stats`) — deliberately
separate from the hashed embodiment profile (`data/contracts/profile.py`), so
a run-config edit never changes `profile_hash` and therefore never
invalidates a resumed prefix's CLI-08 promotion guard.

Mirrors `data/contracts/profile.py::load_profile`'s load shape exactly: same
`FileNotFoundError`/`ValueError`-on-non-mapping behavior, same
`yaml.safe_load` — never the unsafe full-constructor loader, which is
forbidden project-wide (see that module's own docstring).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

RUN_CONFIG_VERSION = 1
"""Freezes the on-disk shape of `RunConfig` — independent of `profile_hash`
(the embodiment side) and of `PIPELINE_VERSION`/`RECORD_SCHEMA_VERSION` (the
provenance side): a fourth, deliberately separate version axis."""


class RunConfig(BaseModel):
    """One batch run's knobs.

    `input_dir`/`input_glob` and `inputs` are two alternative ways to name
    the conversion inputs: an explicit `inputs` list, when non-empty,
    overrides `input_dir`/`input_glob` entirely and is used verbatim in list
    order (Task 2's `discover_inputs` implements that precedence). At least
    one of `input_dir` or `inputs` must be set — the model validator below
    rejects a config that supplies neither, naming both keys.
    """

    model_config = ConfigDict(extra="forbid")

    run_config_version: int
    input_dir: str | None = None
    input_glob: str = "*.mcap"
    inputs: list[str] | None = None
    output_prefix: str | None = None
    format_key: str = "v2.1"
    codec: str = "h264"
    task_string: str | None = None
    fps_override: float | None = None
    emit_episodes_stats: bool = False

    @model_validator(mode="after")
    def _validate_input_source(self) -> "RunConfig":
        if self.input_dir is None and not self.inputs:
            raise ValueError(
                "run config supplies neither 'input_dir' nor 'inputs' — at "
                "least one of these two keys is required to name the "
                "conversion inputs"
            )
        return self


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from a YAML file.

    Mirrors `data.contracts.profile.load_profile`'s own load shape exactly.

    Args:
        path: Path to the run-config YAML file.

    Returns:
        The validated `RunConfig`.

    Raises:
        FileNotFoundError: If no file exists at `path`. The message names
            the path.
        ValueError: If the file parses to an empty document or to a YAML
            document that is not a mapping (e.g. a bare list or scalar). The
            message names the path — a partially-populated run config is
            never returned.
        pydantic.ValidationError: If the parsed document does not satisfy
            the schema (including the input-source model validator above).
            The error names the offending dotted field path and the
            violated constraint; no custom formatting is applied on top of
            it.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"run config file not found: {resolved}")

    with open(resolved) as f:
        raw = yaml.safe_load(f)

    if raw is None or not isinstance(raw, dict):
        raise ValueError(
            f"run config file {resolved} parsed to {raw!r}, expected a YAML "
            "mapping — the file may be empty or malformed"
        )

    return RunConfig.model_validate(raw)
