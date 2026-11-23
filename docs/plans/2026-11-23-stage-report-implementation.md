# Stage Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish verified SFT and DAgger results under `stage_report/` in the `robotwin-hil` repository.

**Architecture:** Read finalized manifests and compact indexes from the local DAgger archive. Materialize Markdown, CSV, and one standalone HTML report in a clean worktree. Validate result counts, path portability, and Git scope before pushing.

**Tech Stack:** Python 3 standard library, Markdown, CSV, HTML, Git.

---

### Task 1: Freeze results

**Files:** Create `stage_report/README.md`, `stage_report/sft/README.md`, `stage_report/sft/seed_results.csv`, `stage_report/dagger/README.md`, `stage_report/dagger/seed_results.csv`, `stage_report/dagger/collection_seed_index.csv`.

1. Read the six archived task manifests and the DAgger report from `/hdd/robotwin-hil-worktrees/dagger-audit/data/evaluations` and `docs/hg_dagger/`.
2. Copy every completed paired outcome into the appropriate CSV; include horizon, protocol, checkpoint, seed, instruction, success, and status.
3. Record `click_alarmclock` SFT 34/50 separately as a diagnostic with its infrastructure error.
4. Write stage summaries with exact counts and relative links.

### Task 2: Interactive view and validation

**Files:** Create `stage_report/dagger/index.html`.

1. Embed the DAgger CSV rows as JSON in a standalone HTML page with task, checkpoint, outcome, and seed filters.
2. Check each clean group has exactly 50 unique seeds and no infrastructure errors or skipped scenes.
3. Verify CSV counts match the manifests and HTML contains every DAgger row.
4. Verify no published Markdown link targets `/hdd` and no raw binary data was staged.

### Task 3: Publish

1. Review `git diff --check`, staged paths, and resulting file sizes.
2. Commit only `stage_report/` and the two plan documents.
3. Push the commit to `origin/main`; verify the remote commit and report paths.
