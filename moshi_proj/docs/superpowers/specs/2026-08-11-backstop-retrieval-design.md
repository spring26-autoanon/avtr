# Backstop retrieval via streaming_sum — design

Date: 2026-08-11. Status: approved (post-demo-sprint serving-experiments phase).

## Context

The demo sprint proved the injection half of moshi-rag is flawless — every fired
reference steered the model correctly within ~1 s — and that the ⟨ret⟩ trigger is
the sole bottleneck. Findings this design builds on (docs/audition_log.md):

- Fire rate is inversely coupled to persona imprint depth (the "election": memorized
  answers outrace the marker in the softmax at the turn boundary).
- A session-start pre-seeded note was ignored on no-fire questions while fired notes
  steered perfectly. Confound unresolved: is note-reading gated on the ⟨ret⟩ token,
  or on temporal/content adjacency to a question? (The pre-seed was a composite bio
  injected at t=0 — both variables differ from a fired note.)
- Forcing text tokens works at the text level; a 14-token forced greeting broke the
  audio. A single forced token at an in-distribution position is untested.
- stt-wait widens the fire window (dice-roll mechanism, confirmed by late-window
  fire timestamps).

Constraints: exactly ONE training run remains in budget. This phase is serve-time
only; its experiments must also determine what that final run should train.

## Goal

Consistent grounding of retrieval-worthy questions on existing checkpoints, by
adding a server-side backstop that engages only when the model does not fire
natively — preserving the model-initiated (full-duplex) retrieval as the primary
path and instrumenting the gap. Non-goals: retraining (yet), question-type
classifier (deferred until E4 shows declines actually break), scripted greetings.

## Workspace

- Branch: `serving-experiments` (off `dialogue-data-prep`). Demo-locked artifacts
  (audition log, run-sheets, runs_backup/) untouched.
- New directory `serving_exp/`:
  - `patches/` — versioned apply/revert scripts for channel.py surgery (anchored
    string replacement with assert-on-drift, backup file per apply; no more
    unversioned heredocs).
  - `protocol.md` — experiment matrix + verbatim probe scripts.
  - `results.md` — one row per session: date, checkpoint, scaling, stt-wait,
    patch config, native fires, backstops, per-question verdicts, log path.
  - `preseed_texts/` — note variants under test.
- Serving logs continue to `replay/auditions/logs/` (existing scorer works on them).

## Experiments (cheapest, most informative first)

### E1 — Temporal adjacency (resolves the pre-seed confound)
Patch: silent injection — on turn-switch after a user question, the server
generates a question-specific reference (existing Gemini path, existing context
snippet) and injects via the existing `_async_update_reference` path. No ⟨ret⟩
anywhere; the model never "asked".
Checkpoint: 4c2-700 @ 2.0, stt-wait 2.0 (fire-starved → nearly every turn is a
clean no-fire trial). Probes: the trained persona set + 2 factuals, ~10 questions.
Decision rule: if answers track injected notes on ≥ half the no-fire turns →
note-reading is adjacency-gated, silent backstop suffices, skip E2. If ignored →
⟨ret⟩-token gating confirmed → E2.

### E2 — Single forced ⟨ret⟩ at the boundary (only if E1 fails)
Patch: at turn-switch on an unfired question turn, force exactly one ⟨ret⟩ token
via `on_text_hook` (in-distribution position, unlike the greeting experiment),
then let the normal fire machinery react (real context, real Gemini call).
Explicit audio gate: if a single boundary token audibly garbles speech, the
forcing family is dead; record and stop.
Decision rule: grounded answers on backstopped turns without audio damage →
backstop mechanism = forced-fire. Otherwise → serve-time backstop infeasible;
the final training run must carry the whole fix (see Training implications).

### E3 — Cross-checkpoint sweep of the winning mechanism
Checkpoints: 4b-300 @ 2.0 rank 32 (fire-rich control — measure native-fire
interference), 4c2-400 @ 2.0 (mid), 4c2-700 (already done in E1/E2), Stage 3 if
checkpoints are locatable (coin-flip trigger — generalization bonus, sessions
kept < 90 s due to its 99 s ceiling).
Measure per session: native fires, backstop engagements, grounded-answer rate,
stalls, voice sanity.

### E4 — Decline probe
On the winning mechanism + 4c2-400: the trained decline set (live Bitcoin price,
weight of the moon, etc.). Backstop WILL engage on these (they are questions).
Measure whether injected notes flatten the trained refusal. Mitigation to test in
the same session: decline-aware reference prompt line ("if the question asks for
live or unknowable data, output exactly: no reference available") + server skips
injection on that sentinel. Decision: if declines survive (natively or via
sentinel), the type classifier stays deferred permanently for this project.

## Backstop patch architecture (built after E1/E2 picks the mechanism)

Channel-level, single file, env-gated (`MOSHI_BACKSTOP=1`):
- Per-turn state: `fired_this_turn` (set by existing fire detection, reset on
  switch-to-user).
- Trigger point: the existing switch-to-model event. Engage iff: backstop enabled,
  turn unfired, and the STT transcript tail is question-shaped (ends with `?` —
  the STT already punctuates; no NLP).
- Action: winning mechanism from E1/E2 (silent inject, or forced ⟨ret⟩ + normal
  machinery).
- Every engagement logs `[Backstop] engaged (turn N)`; the session backstop rate
  (backstops ÷ question turns) becomes the standing trigger-health metric,
  extractable by scripts/audition_checkpoint.py with a one-pattern addition.

## Success criteria

1. ≥ 90% of retrieval-worthy questions grounded (native + backstop) on 4c2-700 —
   best voice, worst trigger.
2. 4b-300 native fire rate unchanged with backstop armed (non-interference).
3. Declines survive per E4 (or documented failure + sentinel mitigation verdict).
4. Backstopped notes injected ≤ 1.5 s after question end.
5. serving_exp/results.md row for every session; no unlogged experiments.

## Implications for the single remaining training run

- E1 says adjacency-gated → final run targets fire-rate only: synthetic question
  variety (many voices/phrasings per answer) + contradiction clips (reference
  disagrees with the memorized answer, keeping asking valuable). Success metric:
  backstop rate driven toward zero on the E3 protocol.
- E1/E2 say token-gated and forcing works → same as above (backstop covers serving
  meanwhile).
- E2 fails entirely → final run additionally includes ambient-reference clips
  (notes present without markers, answers using them) to train un-asked-for note
  reading, making the silent backstop viable natively.

## Risks

- Backstop fires Gemini on every unfired question: API cost/latency in live loop —
  bounded by one call per question turn, same as a native fire.
- Injected note arrives mid-answer (same timing profile as the known late-fire
  pattern; say-wrong-then-correct may appear on backstopped turns).
- channel.py drift between patches — mitigated by anchored asserts + per-apply
  backups in versioned scripts.
- Contention lesson stands: all verdict sessions on a quiet box, no training
  running during E-sessions.

## Outcomes (2026-08-11)

E1 verdict: TOKEN-GATED and destabilizing — silent injection closed. E2 verdict: PROVEN
(forced single ret at boundary; note must land before the answer forms — delay 0.8s on a
fast checkpoint). Winning mechanism shipped as e2_forced_ret v4.1 (native timestamps with
8s lookback, 6s refractory, exact forced/native attribution logging). Locked composite:
4c2-400 @ scaling 2.0, stt-wait 0.5, delay 0.8. Success criteria: grounding ~85% first-ask
(re-homed to 400; 700-class rungs delivery-limited beyond rescue), non-interference PASS,
declines PARTIAL (confabulate-then-disclaim; sentinel untested), latency PASS. Training-run
branch selected: fire-rate + contradiction + statement-response data; ambient-reference
clips unnecessary. Full evidence: serving_exp/results.md.
