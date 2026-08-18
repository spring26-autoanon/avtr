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

## v4.2 implementation notes (so tomorrow is turnkey)
- All work extends `serving_exp/patches/e2_forced_ret.py` (v4.1 = commit 3944ac0); same
  patchlib apply/revert flow, tests in `tests/test_e2_patch.py` (28 green at v4.1).
- **Holdoff**: record session-start time where the slot's backstop state initializes; in the
  question handler, skip arming when `now - session_start < MOSHI_BACKSTOP_HOLDOFF` (default
  ~12.0, env-tunable). Log `[Backstop] holdoff skip`. Rationale: both opening hallucinations
  (Ruri woodcarver, Borges) were backstopped FIRST questions on cold context; natives are
  never blocked.
- **Sentinel**: per the design spec's E4 section — template line "if the question asks for
  live or unknowable data, output exactly: Reference: no reference available"; server-side,
  drop the injection when the generated reference contains that sentinel (check in
  `_handle_reference_text` path). Fixes the "six thousand" Bitcoin confabulation.
- **Templates**: two files on the box (find via `find ~ -name "*reference_prompt_template*"`);
  append FIRST-PERSON rule + "'moshi' speaker IS Danielle" to BOTH; server restart required
  (templates load at startup). Mechanism proven tonight: she parrots the note's grammatical
  person, and whichever person enters context first owns the session.
- Already env-tunable, no code needed: REFRACTORY (3.0 in serve_demo.sh), DELAY, TTL, LOOKBACK.
- Regression bar: statement opening stays clean, natives still logged/unsuppressed, 2-min
  segment protocol unchanged. Full suite green before any box rsync.
