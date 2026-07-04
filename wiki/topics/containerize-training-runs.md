# Containerize the training run (backlog — "for soon")

> 🧊 **DESIGN-NEVER-BUILT / DORMANT (icebox, seeded 2026-05-24).** Captured during
> the research-lab ↔ training integration design; **not started**. This page
> exists so the idea and its one real open question (Docker-on-macOS has no MPS
> passthrough) survive to the session that picks it up. Resolve reading (1) vs (2)
> below before building.

## The idea

Make a training run a **containerized unit**, and run **one container at a
time**. Refine the `gomoku-train` skill around that unit to **ease startup
friction and lower startup time** (proper layer/dependency caching, warm
model + venv, no cold rebuilds).

## Why it fits the design we just landed

The research lab's GPU-required queue is **serial by construction** — one MPS
tenant at a time, because the device contends with itself ([[project-light-all-engines]]).
"Run one docker at a time" is literally that serial lane, with the run unit
swapped from "a process group" to "a container." A time-capped training task
(the `--max-wall-secs` self-terminate design) becomes: start container → train to cap →
self-terminate → next queue item. The container is just a cleaner, more
reproducible run boundary than `pkill -f sweep_runs/<cell>/`.

Goals:
- **Lower friction**: one `run` verb, reproducible env, no "which venv / which
  flags / which gotcha" each launch (the launch-sequence-runbook's pre-flight
  shrinks).
- **Lower startup time**: cached deps + warm weights so a 10-min training slice
  isn't dominated by a multi-minute cold start. Matters more as slices get
  short (10-min interleave with exploration).
- **Isolation / reproducibility**: pinned environment per run, clean teardown,
  no leaked workers.

## The one open question (resolve before building)

**Docker on macOS has no Metal/MPS passthrough.** Docker Desktop runs Linux
containers in a VM with no access to the Apple GPU — a PyTorch run inside it
falls back to CPU, which defeats the entire point of MPS acceleration
([[user-hardware]]). So the literal "Dockerize the training run on the M5 Max"
does not work for the GPU path today.

Two readings that *do* work, pick one when this thaws:

1. **Off-Mac / at-scale target.** Container = the bridge to a Linux+CUDA host
   (cloud or a box), which is exactly the [[az-at-scale-vs-laptop]] / 15×15 +
   Gomocup era. There, Docker + one-at-a-time + cached layers is the standard,
   correct pattern. This is the most likely intended target.
2. **Non-Docker isolation unit on the Mac.** If the goal is friction/startup/
   reproducibility *on this Mac* (not portability), the win is achievable
   without containers losing MPS: a pinned lockfile + a warm prebuilt venv +
   a single `run` entry point + cached weights. Same friction/startup goals,
   keeps Metal. "Container" then means "reproducible run unit," not Docker.

## Next action when picked up

Decide reading (1) vs (2) first — it changes everything. If (1), this is part
of the at-scale move and should be designed against the cloud host's GPU/driver
stack. If (2), it's a `gomoku-train` skill refactor: lockfile + warm venv +
`run` verb + weight cache, measured by cold-start seconds saved per slice.

Cross-refs: [[az-at-scale-vs-laptop]], [[m5-max-as-mainframe]],
the research-lab/training integration thread ([research-lab-charter.md](research-lab-charter.md)).
