# Frontier Lab Extension

Project-local pi extension for BFS/hot frontier fanout.

## Commands

```text
/frontier-init ml-perf
/frontier-start --dry-run
/frontier-start --max=5
/frontier-start --max=1 --lane=baseline-harness
/frontier-start --max=5 --no-merge
/frontier-start --resource=gpu --max=1
/frontier-status
/frontier-stop
/frontier-curate
```

## How it works

- Reads `.frontier/config.json` and `.frontier/lanes.json`.
- Selects unblocked lanes by heat/filter.
- Creates isolated git worktrees under `.frontier/worktrees/`.
- Starts child `pi --mode json --no-extensions` workers so they get skills and project context but cannot recursively orchestrate.
- Schedules by resource limits from config, e.g. one GPU lane at a time.
- Requires worker receipts and open notes.
- Merges worker branches sequentially unless `--no-merge` is passed.
- Runs a wiki curator after integration unless `--no-curator` is passed.

Runtime artifacts are ignored under `.frontier/runs/` and `.frontier/worktrees/`.
