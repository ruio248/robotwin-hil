# Stage report design

The repository root will contain `stage_report/`, with `sft/` for autonomous SFT baselines and `dagger/` for HIL collection and LoRA comparisons. This follows the user's chosen destination. The report covers the six tasks represented in the finalized DAgger evidence archive.

Each stage gets a readable Markdown summary and a compact per-seed CSV. DAgger also gets the collection seed index and a standalone interactive HTML table. The published HTML uses embedded outcomes, so it works without a backend. Source trajectories and videos remain in the local evidence archive; their approximately 7.6 GB of binary files will not be copied into Git.

The SFT summary separates five clean fixed-seed baselines from `click_alarmclock`'s historical diagnostic baseline, which has a scene-screening infrastructure error. DAgger results retain each run's horizon, protocol, checkpoint, exact seed outcomes, and limitations. Markdown links use repository-relative paths. The release must validate counts against source manifests and confirm that no local absolute paths appear in published links.

Publication uses an isolated worktree based on the latest `origin/main`. Only `stage_report/` and this release plan will be committed, then pushed to `main`, leaving the existing dirty worktrees untouched.
