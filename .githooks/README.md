# .githooks/ -- credential/secret scanning

Tracked git hooks that run [git-secrets](https://github.com/awslabs/git-secrets) (AWS-specific
patterns) and [gitleaks](https://gitleaks.io/) (generic secret detection) before a commit lands
and before a push leaves this machine.

This directory exists because `.git/hooks/` is **not** tracked by git -- a checked-in hooks
directory activated via `core.hooksPath` is the only way local enforcement travels with the repo
across clones. `CLAUDE.md` documents the same one-time setup in short form (its own Environment
Setup section), but `CLAUDE.md` itself is gitignored, so this file is the copy that actually
reaches a fresh clone.

## One-time setup (per clone)

```bash
# 1. Install both scanners (Homebrew formulas; no binaries vendored into the repo)
brew install git-secrets gitleaks

# 2. Point this repo's hooks at the tracked directory
git config core.hooksPath .githooks

# 3. Arm git-secrets' AWS pattern set for THIS clone
git secrets --register-aws
```

Step 3 is **per-clone and untrackable**: `git secrets --register-aws` writes its patterns into
this repo's local `.git/config` (`secrets.patterns`, `secrets.providers`, `secrets.allowed`), which
lives outside git's own version control. Anyone setting up a new clone must run it again -- there
is no way to make this step travel with the repo the way `.githooks/` itself does.

`git secrets --install` is deliberately **not** used here. It would write hook scripts directly
into the untracked `.git/hooks/` directory, which is exactly the problem `core.hooksPath` +
`.githooks/` exists to avoid.

## Why these hooks chain instead of replace

`core.hooksPath` is already set at **system** scope on Amazon-managed machines, pointing at Amazon
git-defender -- a mandatory, root-owned secret-scanning control. Setting `core.hooksPath` at the
*local* (repo) scope, as step 2 above does, overrides that for this clone only. If these hooks
did not re-invoke git-defender's own same-named hook, enabling `.githooks/` here would be a **net
loss** of scanning, not a gain.

Both `pre-commit` and `pre-push` resolve `git config --system --get core.hooksPath` at runtime and
delegate to whatever they find there (with the original args and, for `pre-push`, the replayed
stdin ref list), then use its exit status. This is unconditional: it runs even when git-secrets or
gitleaks are not installed on this machine, so a missing local tool can never suppress the
mandatory org-level control underneath. If there is no system hook to delegate to (a machine
without git-defender), the hook just exits on its own scan result.

## Missing-tool policy: warn, don't block

If `git-secrets` or `gitleaks` is not on `PATH`, the hook prints a visible warning naming the tool
and this doc, and continues -- it does not hard-fail. This hooks directory is opt-in per clone; a
hard failure on a missing tool would mean a contributor who has not yet run `brew install` cannot
commit or push at all. The git-defender delegation above still always runs, so the mandatory
control is never weakened by this choice -- only the *extra*, local scan is skipped.

## What gets scanned

- `pre-commit` scans **staged** content (`git secrets --scan --cached`, `gitleaks git --staged .`).
- `pre-push` scans the **commits actually being pushed** for each ref (parsed from git's own
  stdin ref list), not full repository history. `git secrets --scan-history` is deliberately not
  used for this: it hardcodes `git log --all`, which sweeps every local and remote-tracking ref,
  not just the one being pushed -- on this repo that would pick up an unrelated, already-published
  finding on a different branch and block every future push regardless of content. A full
  `--scan-history` sweep is still valuable as a periodic whole-repo audit (see
  `260910-v0o-SUMMARY.md` for the one this task ran) -- it just does not belong in a per-push gate.

## Allowlists

Two files at the repo root suppress known-safe matches:

- `.gitallowed` -- read automatically by git-secrets (`load_allowed()` in its own script); each
  non-comment line is an extended regex tested against every match's `file:line:matched-text`
  string.
- `.gitleaks.toml` -- gitleaks config (`extend.useDefault = true` plus a narrow `[allowlist]`).

Both are scoped to exactly the same four known-safe sites (AWS's own `123456789012` documentation
placeholder, Canonical's public AMI-owner account `099720109477`, and the
`training/sample_dataset/meta/episodes_stats.jsonl` dataset-statistics file), each with a comment
explaining why it's safe. Widen these only for a newly classified, genuinely safe finding -- never
to force a red gate green.
