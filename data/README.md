# Physical AI Data Engine

A raw Galaxea R1 Pro ROS2/MCAP teleoperation recording becomes a LeRobot v2.1 dataset that this repo's GR00T training modules load directly, with no manual step in between.

Two constraints shape everything else here. First, the output targets LeRobot **v2.1** specifically — the current `lerobot` PyPI package targets v3 and actively rejects v2.1 datasets, so this module writes directly to the v2.1 file spec rather than depending on that package. Second, the custom ROS2 message types (`hdas_msg`, `teleoperation_msg_ros2`, `system_manager_msg`) decode from schema text embedded in the MCAP file itself — no ROS2 install is required or supported anywhere in this pipeline.

## Install

Activate the repo-root `.venv` (uv, Python 3.14; see the repo's top-level `CLAUDE.md`), then:

```bash
uv pip install -r data/requirements.txt
```

`data/requirements.txt` is the pinned source of truth for every dependency this module needs — see that file rather than trusting version numbers restated elsewhere. `ffprobe` (from the `ffmpeg` package) is a separate CLI install used only by inspection/verification tooling outside the hot path; the pipeline itself never imports it.

## Quickstart

Every command below runs from the repo root with the venv active. Each command's own `--help` is authoritative — this README does not enumerate every flag.

**1. Preflight a recording batch.** `data.tools.inspect_mcap` is standing tooling meant to be re-run against every new recording batch, not a one-off script. It writes nothing to disk.

```bash
python -m data.tools.inspect_mcap schemas <mcap-path>...
python -m data.tools.inspect_mcap topics <mcap-path>...
python -m data.tools.inspect_mcap compare <mcap-path>...
```

**2. Convert.**

```bash
python -m data.cli <output-prefix> \
  --profile data/profiles/galaxea_r1_pro.yaml \
  --run-config data/run_configs/galaxea_r1_pro_batch.yaml \
  --dry-run
```

Run `--dry-run` first. It reports resolved action sources, predicted frame counts, which inputs would abort and why, the resume skip list, the output-prefix verdict, and the predicted dataset revision — without decoding, encoding, or creating anything under the output prefix. Drop `--dry-run` to actually convert.

**3. Validate.**

```bash
python -m data.validate <output-prefix> --profile data/profiles/galaxea_r1_pro.yaml
```

Runs the post-write check registry against an already-written dataset prefix. It reports and amends provenance only — it never repairs, trims, or re-encodes anything on disk — and its process exit code is the verdict.

**4. Look at the result.**

```bash
data/tools/visualizer/visualize.sh --root <dataset-dir>
```

Optional and non-gating. See `tools/visualizer/README.md` for flags and detail rather than restating them here.

## The two-config split

The single most confusable thing in this interface: `--profile` and `--run-config` are two different files with two different jobs.

- `--profile` is the hashed embodiment description — *what the robot is*: topic mapping, joint layout, camera roles, resample policy. Editing it changes `profile_hash` and therefore produces a new dataset revision.
- `--run-config` carries run knobs — *which inputs, where to write, codec, task string, fps override*. Editing a run knob never changes the profile hash, so it never invalidates a resumed output prefix.

See [docs/extending.md](docs/extending.md) for the profile schema in full.

## Training handoff

The emitted LeRobot v2.1 dataset is what GR00T's own loader reads. The training workflow itself — bootstrap, fine-tune, evaluate, teardown — lives in [`../training/gr00t/SKILL.md`](../training/gr00t/SKILL.md) (the authoritative playbook) and [`../training/gr00t/README.md`](../training/gr00t/README.md); this module restates none of it. `python -m data.tools.verify_gr00t_run` mechanically grades a finished fine-tune's own logged numbers for non-degeneracy — the last mile between "a job ran" and "the job actually learned something." [`../workstation/README.md`](../workstation/README.md) covers the GPU/DCV instance that production ingestion and GR00T verification run on. The blow-by-blow procedure for converting a real recording session and running a real GR00T fine-tune verification step on that workstation is maintained as a local working document outside this repo — it exists, it is deliberately not tracked here.

## Where to go next

| Doc | Read this when |
|---|---|
| [docs/architecture.md](docs/architecture.md) | You want the pipeline map — modules, data flow, and the six concepts that will otherwise bite you. |
| [docs/protocols.md](docs/protocols.md) | You're implementing (or deciding whether to implement) one of the five pluggable layers. |
| [docs/extending.md](docs/extending.md) | You're adding a new robot embodiment. |
| [tools/visualizer/README.md](tools/visualizer/README.md) | You want to look at a converted dataset in a browser. |

## Where things run

Develop locally on macOS. Run production ingestion and all GR00T verification on the GPU workstation / EC2 — see `../workstation/` and [`../training/gr00t/SKILL.md`](../training/gr00t/SKILL.md) for that environment.
