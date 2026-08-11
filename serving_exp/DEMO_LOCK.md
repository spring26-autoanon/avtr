# DEMO-LOCKED CONFIG — frozen 2026-08-11 ~06:35 UTC

**One command on the box: `bash ~/serve_demo.sh`** (box also holds `~/channel.py.v41_DEMO_LOCKED`).

## The composite
- Checkpoint: 4c2-600 (`runs/moshika_rag_stage4c2/.../checkpoint_000600`, mirrored in Mac runs_backup/)
- MOSHI_LORA_SCALING=2.0, MOSHI_LORA_RANK=64, --stt-wait-time 0.5, port 8998, GPU 0
- Backstop: e2_forced_ret **v4.1** (commit 3944ac0) — MOSHI_BACKSTOP_FORCE=1, DELAY=0.8,
  REFRACTORY=3.0, LOOKBACK=8.0 (default), TTL=8.0 (default)
- Encoder on :8001 must be up (NEVER `pkill -f moshi.server` bare — kills it; use `pkill -f "moshi.server --hf"`)

## Why this config (evidence: results.md, logs 2026-08-11_*)
Best voice of sprint (ums/so/likes), identity intact, statement-carry, healthy delivery,
100% grounding (~6% native fires + backstop). Fallback: 4c2-400 same serve config (more
natives, decline-safe, weaker voice).

## Demo protocol
- **2-minute segments, reconnect between** (every long session stalls once ~2.5-3 min in)
- Open with a STATEMENT ("Hi Danielle!"), question is the last thing you say, keep the ball moving
- Run-sheet: identity q -> work (native fires happen here) -> houseplants + follow-up -> travel
- Avoid live-data questions (declines flatten at this dose) unless demonstrating the failure mode
- Attribution in serve.log: `native fire observed` / `forced ret consumed` / `refractory skip`

## Known flaws (accepted)
Wrong-first-then-correct when her memorized guess races the note; declines flatten;
occasional Moshi/Danielle name split; late-session stall (mitigated by short segments).

## v4.2 backlog (tomorrow)
1. Session-start backstop holdoff (~12s) — kills opening hallucinations mechanically
2. E4 decline sentinel (template line + skip-inject on sentinel)
3. Reference-template lines: FIRST PERSON always + "'moshi' speaker IS Danielle" (both files, restart after)
