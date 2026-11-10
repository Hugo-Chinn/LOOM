# Explorations

Self-contained studies of new ideas. Each idea gets one sub-folder holding its
plan, code, tests and run commands. Rules:

- Nothing here is imported by `loom/`; the supported package and its defaults
  stay unchanged. Studies subclass or wrap the core instead of editing it.
- Historical code in `research/` is never edited; studies subclass its models
  and inject them at launch time.
- Outputs go to `runs_local/` or `configs/local/` (both git-ignored).
- A study states what was and was not executed. No result is claimed until
  per-seed, per-task numbers exist.

| Folder | Idea | Status |
|---|---|---|
| `relational_supervision/` | Sibling and parent/child relational losses (RelSup) | Code written, **not yet executed** |
