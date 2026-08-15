#!/usr/bin/env python3
"""
Count the growing "LM buffer empty" pad-token stall documented in the
pad-token-sampling-drift investigation — automates what every session in
that investigation so far has done by hand: counting consecutive
occurrences of moshi-rag's own

    [VAD] User stopped speaking but LM buffer empty, remaining in user turn

log line (TurnManager._update_active_speaker(), real moshi-rag source) in
the run-up to each "switching to model" turn transition, from a demo
session's server.log.

Usage:
  uv run scripts/count_pad_stalls.py demo/sessions/2026-07-29T07-19-59Z/
  uv run scripts/count_pad_stalls.py demo/sessions/2026-07-29T07-19-59Z/server.log

Prints one stall count per "switching to model" event, in session order —
the first value is what the still-outstanding controlled experiment (ask a
<ret>-triggering question as literally the first turn of a fresh session)
cares about: if it's already small with no prior conversation depth, that
points at raw step count (state.offsets, never reset except a true is_first
slot reset) rather than conversation depth/content as the real driver.

A stall count is attributed to the switch-to-model event it immediately
precedes; "immediately precedes" means "since the previous switch event (or
session start)", not literal file-line-adjacency — other log lines (step
timing, RAG events) interleave at the same ~80ms cadence and shouldn't break
a count that's otherwise one unbroken run of empty-buffer polls.

Also flags, per run, any `batched step ... took Xms` warning that fell
inside it — the fast path to ruling out "the GPU was just slow right then"
for every run in the session, not just one hand-checked window. A run with
no warnings and a count in the tens/hundreds means the step loop was
ticking on schedule the whole time and the model was genuinely,
repeatedly choosing not to produce a real token — not waiting on compute.

Matching the turn-transition line itself is a substring match ("switching
to model", case-insensitive) — confirmed verbatim against a real
server.log on wb-gpu-a1ultra (2026-07-29T18-08-14Z session); an earlier
version of this script guessed "switch to model" (present tense, unverified
— this project's own source tree doesn't vendor moshi-rag, torch/moshi are
VM-only per the `--all-extras` install) and matched nothing at all
on that real log. If a future moshi-rag version changes this wording again,
a session reporting zero events despite having real turns is the signal to
recheck it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_STALL_LINE = "LM buffer empty, remaining in user turn"
_SWITCH_LINE = re.compile("switching to model", re.IGNORECASE)
_STEP_WARNING = re.compile(r"batched step.*took [\d.]+ms", re.IGNORECASE)
_TIMESTAMP = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{2})")


def resolve_log_path(target: Path) -> Path:
    if target.is_dir():
        return target / "server.log"
    return target


def analyze_runs(lines: list[str]) -> list[dict]:
    """Returns one dict per "switching to model" event, in order:
    {count, start_ts, switch_ts, step_warnings}. `count` is how many stall
    lines appeared since the previous switch event (or session start);
    `step_warnings` is every `batched step ... took Xms` line that fell
    inside the stall itself — from its first stall line to the switch line,
    not from the previous switch event, so a warning from the *prior* turn's
    own speech (which can precede the next run's first stall line by many
    seconds) never gets misattributed to a run it has nothing to do with.
    Present means the step loop was genuinely running slow during the stall
    (a compute explanation), absent means it wasn't (a sampling-behavior
    explanation). Timestamps are the log's own leading `HH:MM:SS.ss`, or
    None if a line didn't start with one."""
    runs: list[dict] = []
    running = 0
    start_ts: str | None = None
    step_warnings: list[str] = []
    for line in lines:
        if _STALL_LINE in line:
            running += 1
            if start_ts is None:
                m = _TIMESTAMP.match(line)
                if m:
                    start_ts = m.group(1)
        elif _STEP_WARNING.search(line):
            if start_ts is not None:  # only once a stall run is actually in progress
                step_warnings.append(line.strip())
        elif _SWITCH_LINE.search(line):
            m = _TIMESTAMP.match(line)
            runs.append(
                {
                    "count": running,
                    "start_ts": start_ts,
                    "switch_ts": m.group(1) if m else None,
                    "step_warnings": step_warnings,
                }
            )
            running = 0
            start_ts = None
            step_warnings = []
    return runs


def count_stalls_per_switch(lines: list[str]) -> list[int]:
    """Returns one count per "switching to model" event, in order: how many
    stall lines appeared since the previous switch event (or session
    start)."""
    return [run["count"] for run in analyze_runs(lines)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=str, help="Path to demo/sessions/<session_id>/ or a server.log file directly")
    args = parser.parse_args()

    log_path = resolve_log_path(Path(args.target))
    if not log_path.exists():
        print(f"No server.log found at {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path, errors="replace") as f:
        lines = f.readlines()

    runs = analyze_runs(lines)

    if not runs:
        print(f"No 'switching to model' events matched in {log_path}.")
        print("Either this session had no turns, or the match string needs updating —")
        print("see this script's own docstring before trusting a 0 result.")
        sys.exit(0)

    counts = [run["count"] for run in runs]
    print(f"Pad-stall counts per switching-to-model event, in session order ({log_path}):")
    print("  " + ", ".join(str(c) for c in counts))
    print()
    print(f"First turn (no prior conversation depth): {counts[0]}")
    if len(counts) > 1:
        print(f"Remaining turns: {counts[1:]}")

    print()
    flagged = [run for run in runs if run["step_warnings"]]
    if not flagged:
        print("Compute check: no 'batched step ... took Xms' warnings inside any run —")
        print("the step loop was on schedule throughout every stall; these are not")
        print("compute/warm-up delays.")
    else:
        print(f"Compute check: {len(flagged)}/{len(runs)} run(s) had step-timing warnings inside them:")
        for i, run in enumerate(runs):
            if run["step_warnings"]:
                window = f"{run['start_ts']}–{run['switch_ts']}" if run["start_ts"] else "?"
                print(f"  run #{i} (count={run['count']}, {window}):")
                for w in run["step_warnings"]:
                    print(f"    {w}")


if __name__ == "__main__":
    main()
