# lerobot-dataset-visualizer (optional, non-gating)

**This tool is optional and explicitly non-gating.** It gates no automated check in
`data/validation/`, blocks no verification step, and adds no Python runtime dependency to this
repository. It exists purely so a developer can look at a converted dataset — video, per-episode
charts, and (best-effort) stats — in a browser. Nothing this tool renders is ever recorded,
reported, or cited as a validation verdict; a human looking at charts is an inspection aid, not
evidence. The container is not part of the validation control flow at all.

## Reversal, recorded knowingly

`REQUIREMENTS.md`'s fourth-pass note previously excluded human sign-off on visual diffs as
out of scope. This is a **recorded reversal**, made after confirming (below) that this specific
upstream tool genuinely supports the LeRobot **v2.1** format this project emits — not an
oversight or a scope-creep addition.

## Pinned version

- Repository: <https://github.com/huggingface/lerobot-dataset-visualizer> (Apache-2.0)
- Pinned commit SHA: `dc59887796fd41f37040c0df6b10e6f6a30a1854`
- Resolution date: 2026-09-06, via `git ls-remote https://github.com/huggingface/lerobot-dataset-visualizer refs/heads/main`

**Why pin by exact commit SHA, not a branch or the prebuilt HF Space image:** a quality judgement
made with an unversioned tool is not reproducible — if the tool itself can change silently between
two visual inspections, a "looks fine" from last week means nothing today. `checkout.sh` refuses to
build from a SHA that does not resolve, naming the SHA, rather than silently falling back to
whatever `main` currently points at.

## Why this tool works where the `lerobot` PyPI package does not

`src/utils/versionUtils.ts` in the pinned commit declares:

```ts
const SUPPORTED_VERSIONS = ["v3.0", "v2.1", "v2.0"];
```

Unlike the current `lerobot` PyPI package (which targets v3 and rejects v2.1 with a
`BackwardCompatibilityError`), this visualizer explicitly supports the **v2.1** format this
project's writer emits.

## Known upstream limits (documented, not discovered at use time)

- **No 3D pose view for Galaxea R1 Pro.** The 3D URDF viewer supports SO-100 / SO-101 / OpenArm
  only. Video playback and the state/action charts work; there is no 3D robot pose rendering for
  this embodiment.
- **The Annotations panel does not apply.** It targets the v3.1 language-annotation schema, which
  a v2.1 dataset does not carry.

## Manual mirror of an automated check

The app's "Action Insights" panel computes autocorrelation and state/action alignment — this is a
**manual, visual mirror** of the automated `consistency.action_state_lag` check that plan 03-04
owns. The panel's output is useful for a human to eyeball, but — as with everything else this tool
renders — it is never recorded as a verdict. The automated check is the actual gate; this panel is
a second pair of eyes, nothing more.

## Usage

The supported path is one command:

```bash
data/tools/visualizer/visualize.sh --root <dataset-dir>
```

`--root` is required and has no default — `data/output` is not gitignored, so defaulting there
would risk pulling an untracked ~60MB dataset into the working tree. The script:

1. Builds the pinned base image via `checkout.sh` if it is not already present locally.
2. Idempotently derives a **patched** image, tagged `<short-sha>-localpatch-p<serve-port>` (see
   "Why the base image alone is not enough" below for what gets patched and why).
3. Starts `data.serve_dataset` in the background and the visualizer container, wired together.
4. Prints the one URL to open: `http://localhost:<viz-port>/local/dataset/0`.

There is nothing to type into the visualizer's landing page — the dataset id is fixed and
pre-wired. Re-running the same command with the same `--serve-port` reuses the cached patched
image instead of re-deriving it. Ctrl-C stops the container and the backgrounded server.

Every patched image carries a `data-engine.serve-origin` label recording the exact origin baked
into its bundle. `visualize.sh` reuses a cached patched image only when that label matches the
origin it is about to use; any mismatch — including an image with no label at all, i.e. anything
built before the IPv4-origin fix — is re-derived automatically, with an explicit message naming
both the stale label and the required one. Practically: if you already ran an older version of
this script, there is nothing to clean up. The first run after updating re-derives on its own,
and the superseded image is simply left dangling for a normal `docker image prune`.

Flags:

- `--root PATH` — required, no default. Must contain `meta/info.json`.
- `--serve-port N` — port for the local `data.serve_dataset` backend. Default: `8000`.
- `--viz-port N` — port for the visualizer container (published on the host; the container's own
  internal port is always `7860`). Default: `7860`.
- `--help`

Environment overrides: `PYTHON`, `VISUALIZER_COMMIT_SHA`, `CHECKOUT_DIR` (see the
`## Environment overrides` section below — `visualize.sh` reads the pin out of `checkout.sh`
rather than re-typing it, and passes these through to it untouched).

### Manual three-step path (for seeing the pieces individually)

This is the path this README previously — and incorrectly — presented as sufficient on its own.
It still works for building the base image and running the backend, but a bare `docker run` of
the **base** image (skipping `visualize.sh`'s patch step) leaves browser-side fetches pointed at
`huggingface.co`, not at your local dataset. Two independent reasons make that so:

**(a) The app's own route always sends a two-segment id.** The route shape is
`[org]/[dataset]/[episode]`, and the landing page's own code does a bare `router.push(value)` — so
whatever a user types is pushed as-is, and the id the server side receives is always at least two
path segments (`org/dataset`), never one. `data/serve_dataset.py`'s route matcher now accepts a
multi-segment `repo_id` for exactly this reason (see its module docstring).

**(b) The client bundle has the upstream origin inlined at build time.** The environment variable
that builds dataset URLs (`DATASET_URL`) is **not** `NEXT_PUBLIC_`-prefixed, but the module that
reads it is pulled into the client bundle anyway. Next.js inlines whatever value was present *at
build time* into that bundle — so passing `-e DATASET_URL=...` at `docker run` only reaches the
**server-side** fetch path; every parquet/mp4 fetch the **browser** makes still goes to
`huggingface.co`. This is why `visualize.sh` derives a patched image from the **built artifact**
via `docker commit` (rewriting the literal origin string under `/app/.next/static`) rather than
relying on the environment variable alone — and why that still satisfies D-06: the upstream
`Dockerfile` is never edited and no upstream source is vendored into this repository.

**The origin baked into the client bundle must be an IPv4 address literal, never a resolvable
name.** `data/serve_dataset.py` binds the IPv4 loopback only, by design; a resolvable name can
resolve to `::1` first — the macOS default — and a browser's `fetch()` does not retry across
address families the way `curl` transparently does. The observable symptom of getting this
wrong is a page shell that loads fine while every dataset/video/chart fetch silently fails, even
though a `curl`-based check of the exact same URL still returns 200 (it fell back to IPv4 behind
your back). `::1` is the searchable term for this failure mode.

This applies only to the *dataset* origin baked into the client bundle above. The `Open:` URL
this script prints, and the manual-path `docker run` URL below, deliberately keep the hostname
form (`localhost`) — do not "fix" either of them to an address literal. Docker publishes the viz
port (`7860`) **dual-stack**, so a hostname resolves to whichever loopback address is reachable
either way there; an address literal there would be a usability regression for zero gain. A
tempting alternative — making `data/serve_dataset.py` itself dual-stack (binding `::` instead of
`127.0.0.1`) — was considered and rejected: it would also fix the symptom, but it widens the
listening surface of a security-reasoned, stdlib-only module whose docstring reasons deliberately
about loopback-only exposure, purely to work around a defect that lives entirely in this one
consumer script. Fixing the one place that bakes the origin is sufficient and strictly safer.

Steps:

1. Build the image locally (clones the pinned commit, builds from the upstream repo's own
   `Dockerfile` — never a rewritten one, never the prebuilt Space image):

   ```bash
   data/tools/visualizer/checkout.sh
   ```

2. Serve a locally converted dataset over plain HTTP (from the repo root, with `.venv` active):

   ```bash
   python -m data.serve_dataset --root data/output
   ```

3. Run the container, pointing `DATASET_URL` at the host's shim. This reaches only the
   server-side path (see (b) above) — browser-side fetches will still go upstream unless you use
   the patched image `visualize.sh` derives instead:

   ```bash
   docker run --rm -p 7860:7860 \
     -e DATASET_URL=http://host.docker.internal:8000 \
     lerobot-dataset-visualizer:<short-sha>
   ```

4. Open <http://localhost:7860> in a browser.

## Environment overrides

`checkout.sh`:

- `VISUALIZER_COMMIT_SHA` — override the pinned commit (must be a full 40-hex-character SHA).
  Lets a later phase move the pin without editing `checkout.sh`'s body.
- `CHECKOUT_DIR` — override the clone working directory. Defaults to a path under `${TMPDIR:-/tmp}`,
  outside this repository's own tree. The script refuses to check out inside this repo (D-06: no
  vendored TypeScript source may enter this repository).

`visualize.sh` (reads `VISUALIZER_COMMIT_SHA` / `CHECKOUT_DIR` out of `checkout.sh` and passes them
through untouched; adds one of its own):

- `PYTHON` — interpreter to run `data.serve_dataset` with. Default:
  `<repo-root>/.venv/bin/python`, falling back to `python3` if that path is absent.

`visualize.sh`'s own flags (`--serve-port`, `--viz-port`) are documented in `## Usage` above; the
patched image's tag shape is `<short-sha>-localpatch-p<serve-port>` — the serve port is baked into
the tag because the patch writes a literal origin, including that port, into the client bundle, so
one patched image is only ever valid for one serve port. The viz port is never baked into any tag;
it is only published at `docker run`.
