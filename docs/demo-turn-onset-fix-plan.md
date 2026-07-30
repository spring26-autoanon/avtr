# Demo turn-onset regression — fix plan

Companion to [`demo-turn-onset-regression.md`](demo-turn-onset-regression.md), which is the
evidence. This file is the sequenced work, the VM validation points, and the decisions that
need a human call before implementation.

**Recommended shape: one code drop, staged validation.** Every behaviour change below is
already (or becomes) env-gated, so all of Phase A and Phase B can land in a single
implementation pass and then be A/B'd on **one** VM trip by toggling environment variables
between sessions. That is more efficient than strict phase-by-phase implementation and loses
no attributability, because each session isolates exactly one variable. Phase B3 is the one
item that must wait for real data.

## Status (2026-07-30)

Decisions taken: `ttfat_s` redefined in place; one code drop; Option E deprecated now and
deleted later; Phase D kept separate.

### VM session 1 — HYPOTHESIS CONFIRMED (`demo/sessions/2026-07-30T19-12-46Z/`)

Run with `DEMO_QUEUE_DIAG=1 DEMO_STEP_PACING=1 --batch-size 16` (the known-bad config), 4
turns, 3 `<ret>` triggers, 96 sampled steps over 94.5 s. §8's backlog prediction holds on
every axis:

| Measurement | Value | Reading |
|---|---|---|
| `work_ms` | mean **49.9**, p95 50.2, max 52.0 | Real compute is 50 ms against an 80 ms budget — **37% headroom, even at batch 16**. The hardware was never the constraint |
| `mean_period_ms` | mean **82.86**, min **80.50** | Never below the 80 ms floor, so the loop can never drain. Exactly the latch in §4.3 |
| `lag_frames` | **52 → 73**, max 92, monotone non-decreasing | Starts at 4.16 s (the startup backlog), ratchets to 7.2 s |
| `qsize` vs `lag_frames` | `qsize ≈ lag_frames − 1` throughout | The backlog is entirely in `input_queue`, not the WebSocket/Opus layer |

**The lag accounts for the latency, quantitatively.** Turn 1: lag 52 frames = 4.16 s,
`ttfat_s` = 4.4545 → residual **0.29 s**. Turn 4: lag ~90 frames = 7.2 s, `ttfat_s` = 7.1043
→ near-exact. Subtract the backlog and the model's true TTFAT is ~0.3 s against the paper's
0.18 s (Table 2). Turn 2's `ttfat_s` of 0.065 s is consistent rather than anomalous: the
model was still delivering turn 1's tail, so `model_text_buffer` was already non-empty.

**A2 validated, both metrics in the same file:**

| turn | `ttfat_s` (new) | `pad_stall_steps` | `turn_switch_to_first_token_s` (old anchor) |
|---|---|---|---|
| 1 | 4.4545 | 61 | 0.1603 |
| 2 | 0.0652 | 3 | 0.1444 |
| 3 | 3.3688 | 49 | 0.0785 |
| 4 | 7.1043 | 86 | 0.4026 |

The old anchor's 0.08–0.40 s band is exactly what §6.1 predicted it would report regardless
of the real wait.

**Content-level proof of stale input** — stronger than the timing, and the evidence the
"barge-in overhang" attempt above went looking for and couldn't find in old logs. Every
turn's `model_response_text` opens with the *previous* turn's tail, and turn 4 (*"can you
tell me about Lionel Messi's career?"*) answers with *"Brazil has won five times, a number no
other country comes close to"* — **turn 2's answer**. Timing alone cannot separate "heard it
late" from "waited"; a stale answer can.

**Two bonus findings from the token stream:**

- **D1 is resolved, no further VM time needed.** Token 3 = PAD
  (`loaders.py: existing_text_padding_id: 3`). Token **0 = EPAD**, established empirically
  with 100% consistency: all 146 occurrences are immediately followed by a real word token,
  PAD is *never* followed directly by a word, and 65% of word tokens are preceded by token 0
  (the remaining 35% are mid-word continuations). That is precisely the Moshi paper's Eq. 5
  construction (*"EPAD is inserted before the next word to indicate the end of the
  padding"*), visible raw in the stream: `.........EWWWEWEW..EWW`. If a turn-onset nudge is
  ever wanted, the lever is a logit bonus on **token 0** — now known, not inferred.
- **Independent refutation of the sampling-drift theory.** PAD is 67.7% of all sampled
  tokens; the Moshi paper states padding is "about 65% of the tokens" in English
  conversational speech. The model's padding rate is *normal*.

One anomaly worth noting for B3: `mean_period_ms` max 156.38, with a 134.21 sample around
`t=80 s` where `lag_frames` jumps 80 → 90. Occasional hitches (GC, client hiccup) each add
permanently to the backlog under pacing — the ratchet caught in the act.

### VM session 2 — fix confirmed, and the real trade-off found (`2026-07-30T19-33-05Z/`)

`DEMO_QUEUE_DIAG=1`, pacing off, batch 16 — one variable changed from session 1. 3 turns.

`ttfat_s` **4.45 / 0.07 / 3.37 / 7.10 → 1.98 / 0.05 / 0.16**. `lag_frames` **47 → 1**,
`qsize` 0. Content bleed gone: session 1's turn 4 answered *"Brazil has won five times"* to a
question about Messi; session 2's turns are each on-topic, with one token of boundary fuzz
(turn 2 opening `"a good question!"`). Turn 1's residual 1.98 s is the startup backlog still
draining, as predicted.

**The important finding is in the drain phase**, and it came from the operator's subjective
report ("responded much faster, basically conversational, but speech degraded a little —
felt slightly fast with metallic artifacts more noticeable") being taken seriously and
measured rather than waved off:

| t_sess | `period_ms` | `lag_frames` |
|---|---|---|
| 0.0 s | — | 47 (3.76 s) |
| 3.9–20.7 s | **65.8 → 67.5** | 36 → 6 |
| 25.0 s | **80.69** | 1 |
| 30–54 s | 66.5 / 80.1 / 80.9 / 66.4 / 80.0 | 4 / 1 / 1 / 6 / 1 |

While draining, the loop consumes an 80 ms frame every ~66 ms — **it ships audio ~21% faster
than real time.** The instant `lag_frames` reaches 1, `period_ms` snaps to 80.7 and tracks
real time; each subsequent hitch triggers a brief 66 ms catch-up burst before settling back.
That transient is what the operator heard, concentrated in the first ~25 s.

**This partially rehabilitates `_patch_server_state_step_pacing()`, and §4.3 overstated the
case against it.** Its premise is still false *as stated* — the step loop cannot outrun the
client, and steady-state output is real-time. But faster-than-real-time delivery **is real
during backlog drain**, which is what its author observed. They read a transient as
continuous and inherent. The patch therefore traded a ~25-second audio artifact at session
start for a permanent multi-second latency, by guaranteeing the drain never completes. That
is a materially fairer characterisation than "the premise was wrong", and it is why B3 is now
required rather than conditional.

### B3 — reverted to OPTIONAL / low priority (see the correction below)

This section briefly read "DECIDED: bounded queue with drop-oldest", on the theory that the
drain phase's faster-than-real-time delivery overran the client's playback buffer and caused
the audible artifact. **Session 3's client-side data disproved that theory** — see "Session 3"
below. Recording the reversal rather than quietly rewriting it, since the intermediate
reasoning was wrong for an instructive reason: it was inferred from `delay`, the field
CLAUDE.md already flags as superseded, instead of `liveBufferS`, the corrected one. Exactly
the mistake this whole investigation has been about.

What survives: the drain-phase fast delivery is real *server-side* (measured `mean_period_ms`
of 45–66 ms while draining, vs. 80 ms settled). What does not: any evidence it is audible or
that it stresses the client buffer.

Current standing: a bounded `Channel.input_queue` (cap ~2 frames, drop-oldest) remains a
reasonable robustness measure — it would eliminate the drain phase entirely and absorb
post-hitch mini-backlogs — but it fixes **no observed problem** at `--batch-size 1`, because
the backlog drains in ~5 s, before the first user turn even completes (session 3: `lag_frames`
reached 1 at t=5.4 s; the first model turn began at t=4.5 s). It is insurance against slower
hardware or genuinely concurrent slots, not a fix. Do not implement it on latency grounds.

**A1, A2, B1, B2, C1 and the C2 documentation half are implemented.** Full local suite green
(411 → 432 tests).

Step 0c of the runbook below has been run on `wb-gpu-a1ultra` (2026-07-30): the web client
**builds clean** — `tsc && vite build`, 104 modules transformed, `dist/` written. That clears
the outstanding risk CLAUDE.md recorded against the 2026-07-27 client changes
(`getAudioDiagLog`/`liveBufferS`), which had been written with no Node toolchain available and
were never type-checked or built. They compile. `npm audit`'s 22 reported vulnerabilities from
the same run are parked as D9 below (assessed non-blocking; do **not** run
`npm audit fix --force` before the trip).

Nothing else is VM-validated yet — the A/B below is the next step, and B1/B2
should be treated as unproven until session 1 confirms the mechanism.

D1 was folded into A1 rather than tracked separately: `ServerState._deliver_step_row` already
receives `text_token` as a plain Python int (`BatchRunner.run_step` does the `.item()` sync
itself), so recording the raw token stream costs no extra device synchronisation. That
mattered — any per-step sync added by the diagnostic would inflate the very step timings it
exists to measure. Session 1 therefore produces actionable data on **both** branches of §8's
prediction, instead of needing a second VM trip if the backlog hypothesis fails.

### Attempts to settle the mechanism from existing logs — both failed

Tried before committing a VM session to it, and recorded here so nobody repeats them:

1. **Barge-in overhang.** If the front-end model's input is L seconds stale, then when the
   *live* STT VAD reports the user has started speaking, the model should keep emitting real
   tokens for ~L more seconds (it has not yet "heard" the interruption). Measured the
   contiguous model-token run after each `switching to user` event across eight sessions:
   **~0 in all of them, pacing on or off.** Non-discriminating rather than falsifying — the
   users in these sessions politely wait for audible silence before speaking, so the model
   had already finished its turn every time and there are no genuine barge-ins to measure.
2. **Turn-boundary text fragmentation.** Under a lagged model stream, the live VAD switch
   should chop the model's utterance mid-phrase, which is visible in the retrieval context
   string. Real fragmentation does appear (e.g. `2026-07-28T06-17-49Z`: `"...today? It"` /
   `"really is! Wondering where you'd be going"` / `"Both are"` / `"great choices!"`), but it
   does not correlate cleanly with `DEMO_STEP_PACING` — Moshi free-runs and produces filler
   independently, so the signal is confounded.

**Conclusion: the existing logs genuinely cannot distinguish "heard the question 4 s late"
from "heard it on time and waited 4 s."** A1 is required, not merely preferable.

---

## Phase A — instrumentation (must land and be validated first)

Nothing else should be trusted until the demo can measure the thing users complain about.

### A1 — Log `Channel.input_queue` depth and step-index-vs-wall-clock

**Why:** the single decisive measurement, and the only one that can falsify the backlog
mechanism. See regression doc §8.

**What:** a new opt-in patch in `scripts/instrumented_server.py`, e.g.
`_patch_step_queue_diagnostics()`, gated by `DEMO_QUEUE_DIAG=1` (default off, matching the
existing `DEMO_ASYNCIO_DEBUG` precedent). Per model step, sampled (every Nth step, to keep
the log readable), record into the session directory:

- `input_queue.qsize()` for each active slot
- step index and wall clock, so cumulative drift vs. the 80 ms budget is directly derivable
- the measured duration of `run_one_step()` itself, separated from any pacing sleep

Write to a dedicated `demo/sessions/<id>/step_diag.jsonl`, not `server.log` — at 12.5 Hz this
is ~1500 records for a 2-minute session and would drown the existing log.

**Tests:** unit-test any pure helper (sampling decision, record shape). Per
`tests/test_instrumented_server.py`'s own stated discipline, do **not** mock
`ServerState`/`Channel` to test the patch wiring — that gets VM-verified.

**VM:** yes, folded into the A/B trip below.

### A2 — Fix the demo's `ttfat_s` anchor

**Why:** it currently reads 0.08–0.32 s on sessions where the human waited 4–19 s, because
the clock starts after the stall (regression doc §6.1). Every demo latency number produced so
far measures the display-gating delay, not response latency.

**What:** move the anchor into `_ChannelState` so it stays unit-testable, and set it from the
VAD end-of-utterance transition rather than from the turn switch. Proposed rule:

- On each `_update_active_speaker` evaluation while `active_speaker == "user"`, if `vad_neg`
  is true and the previous evaluation's was false, (re)set `t_user_stopped`. Taking the
  **last** such transition before onset is important: a mid-question pause produces an
  earlier one that would anchor too early.
- `ttfat_s = t_first_nonpad_model_token − t_user_stopped`, clamped at 0 — matching the
  paper's §3.1 definition ("the delay between the end of a user's utterance and the moment
  the model generates the first audio token").
- Keep the current value under a new name (`turn_switch_to_first_token_s`) rather than
  deleting it. It does measure something real (the `stt_wait_steps` + display-gating
  interval) and prior sessions carry it; silently redefining `ttfat_s` would make old and
  new session logs incomparable without any signal that the definition changed.
- Also emit `pad_stall_steps` — the raw count of consecutive `LM buffer empty` steps. It is
  the cheapest and most robust form of this signal, and `scripts/count_pad_stalls.py`
  already derives it from logs; having it in `turns.jsonl` makes it queryable per turn.

**Tests:** yes, and they are cheap — `_ChannelState` is pure logic with no moshi/torch
import. Drive it through a synthetic sequence (user speaks → pause → resumes → stops → 60
pad steps → first token) and assert the anchor lands on the *last* stop, not the first, and
not on the switch.

**Open decision D-1** below covers whether to redefine `ttfat_s` in place or add a new field
and leave `ttfat_s` alone.

**VM:** yes — validated in the same sessions as A1.

---

## Phase B — the fixes

### B1 — Default `DEMO_STEP_PACING` to disabled

**Why:** regression doc §4.3. The patch's premise ("the server ships audio strictly faster
than real-time... with nothing downstream to throttle it") is false —
`ServerState._gather_step_inputs()` is `get_nowait()`-driven, so the client's real-time
uplink *is* the throttle. The patch only engages when a backlog exists, and then freezes it
permanently.

Additionally worth recording: the client-side jitter-buffer overrun that motivated the patch
was measured with the **first, later-corrected version** of the `liveBufferS` field (CLAUDE.md
notes this itself). The evidence that justified the patch has since been invalidated, and the
corrected field reads 0 ms — a starved client, the opposite condition.

**What:** `scripts/instrumented_server.py:541` —
`os.environ.get("DEMO_STEP_PACING", "1") == "0"` becomes a default-off check
(`== "1"` with default `"0"`), mirroring how `_stt_off_thread_enabled()` was flipped. Keep
the variable so the old behaviour is one export away for A/B. Rewrite the docstring to state
the corrected mechanism and point at the regression doc — do not delete the existing
reasoning trail.

**What NOT to do:** do not build a "safer" pacing variant now. If a real overrun ever
reappears, fix it client-side (target buffer depth / playback-rate trim) or by dropping stale
frames server-side. Any future server-side pacing must be non-latching — e.g. sleep only when
the queue is already empty, or pair the floor with a bounded queue and a drop policy.

**Tests:** assert that with `DEMO_STEP_PACING` unset the patch is a no-op. This needs
`moshi.server` faked via `sys.modules` injection (precedent exists in
`tests/test_model_interface.py` for `torch`), or the gate check can be extracted into a
testable helper. Prefer extracting the helper.

**VM:** yes — session 2 of the A/B.

### B2 — Set `--batch-size 1` for the demo

**Why:** regression doc §4.6. `BatchRunner.run_step()` computes the full batch tensor every
step regardless of how many slots are active, so a single-user demo at the upstream default
of 16 pays ~16× the necessary per-step work against an 80 ms budget.

**What:** add a `--batch-size` flag to `scripts/run_demo.sh` (default `1`, overridable),
appended to `SERVER_CMD` alongside the existing `--init-active-speaker model`. It must **not**
go into `_DEFAULT_GENERATION` / `print_demo_env.py` — `specs/moshirag-evals-requirements.md`'s
"Generation parameters" section deliberately keeps `batch_size` and `init_active_speaker`
path-specific and CLI-only, and that boundary is still correct.

Also correct that spec section's *rationale*, in place, without rewriting the decision: it
currently reads "the demo needs headroom for concurrent live sessions (moshi's own default is
16)". Unused slots are not free headroom; they are a per-step compute multiplier. Note the
original intent and why it was wrong, per this repo's convention for spec corrections.

**Tests:** extend `tests/test_print_demo_env.py` with a negative assertion (`--batch-size`
must not appear in `DEMO_GENERATION_FLAGS`). The shell flag itself has no existing test
harness; leave it to VM validation rather than inventing one.

**VM:** yes — session 3 of the A/B.

### B3 — Startup-backlog handling *(conditional — decide after A1 data)*

**Why:** the 3.9–4.9 s backlog created during model warm-up / STT priming exists in every
session regardless of pacing (regression doc §4.4). Before 2026-07-27 it was drained by
headroom alone. With B1+B2 that should be true again, with a much larger margin — but it is a
latent fragility that will resurface on slower hardware or with several slots genuinely
active, and upstream shares it.

**Options, in order of preference:**

1. **Do nothing.** If A1 shows `qsize()` returning to 0–2 within the first ~15 s and staying
   there, headroom is sufficient and no code is warranted.
2. **Bounded queue with drop-oldest** on `Channel.input_queue`. Cheap, self-correcting, and
   protects against any future slowdown. Dropping stale user audio is the right trade for a
   live conversation — arriving late is worse than missing 80 ms.
3. **Client-side gating** — don't start uploading microphone audio until the server signals
   readiness. Cleanest conceptually, but touches the client and the handshake.

**Do not implement until A1 has run.** If option 1 holds, this closes as "measured, not
needed", with the measurement recorded.

---

### VM session 3 — FIX VALIDATED (`2026-07-30T19-49-54Z/`), with two new findings

`DEMO_QUEUE_DIAG=1`, pacing off, `--batch-size 1`. 4 turns, 4 `<ret>` triggers, plus the
client-side `.webm` recording and `client_audio_diag_*.json`.

| Measurement | Session 1 (pacing, b16) | Session 2 (no pacing, b16) | Session 3 (no pacing, b1) |
|---|---|---|---|
| `ttfat_s` per turn | 4.45 / 0.07 / 3.37 / 7.10 | 1.98 / 0.05 / 0.16 | **0.035 / 0.210 / 0.036 / 0.003** |
| `work_ms` | 49.9 | 50.0 | **33.1** |
| `lag_frames` (first → last) | 52 → 73 (max 92) | 47 → 1 | **29 → 1** |
| backlog drained | never | ~25 s, at 1.21× real time | **~5 s, at 1.78×** |
| `mean_period_ms` settled | 82.9 (floored) | 80.1 | **78.8–80.4 for 58 s straight** |

Both pre-registered predictions held: batch 1 cut `work_ms` 50 → 33 ms (**batch 16 was
costing ~34% more compute per step for nothing**), and the backlog drained in ~5 s at ~1.8×
real time rather than ~18 s at 1.21×. `ttfat_s` of 0.003–0.21 s now **matches the paper's own
figures** (TTFAT 0.0 s, Full-Duplex-Bench turn-taking latency 0.18 s). The latency regression
is fixed.

`liveBufferS` (the *corrected* client buffer field) is flat and healthy throughout —
mean 0.074 s during the drain window, 0.081 s and 0.070 s in the two settled windows, max
0.155 s. Notably this also corroborates the session-1 reinterpretation in regression doc §4.5:
a pacing-on session measured ~0 ms (starved), a pacing-off session measures a stable 70–80 ms.

**New finding 1 — the metallic artifact is a client-side per-utterance buffer warm-up, not
anything server-side.** All five `buffer_drop` bursts coincide with a model turn onset to
within ±0.31 s (onsets at client-clock 4.5 / 12.2 / 28.3 / 42.7 / 57.8 s; bursts at 4.79 /
12.34 / 28.02 / 42.48 / 57.81 s), and the events show `newMaxBufferMs` ratcheting 15 → 20 →
25 → 30 → 35 ms — an adaptive playback cap that starts tight and grows, dropping 18–69 ms
chunks until it settles. 56 drops, 5 underruns, 6 resumes across the session, spread
throughout rather than confined to the drain. This matches the operator's description exactly
("confined to the first few seconds **of the model speaking**, then smooths out"), is present
in every session, and is unrelated to pacing or batch size. Tracked as D10.

**New finding 2 — retrieval quality is now the dominant remaining problem**, and it is not
caused by the fix. Session 3's answers show a "Uruguay fixation": turn 2 (*"which country won
the most?"*) and turn 3 (*"what was Argentina's record?"*) both received references that
answer neither question, restating only the 1930 Uruguay fact. Turn 4's reference even
*explicitly flags the model's own error* — "it is erroneously stated that Uruguay has won the
tournament five times... though historically Brazil holds the record" — and the model still
said "five Uruguay titles, leading to some confusion". Grounding timing does not explain it:
turns 2 and 3 had grounding land *before* speech onset (+1.06 s / +1.03 s vs. +1.23 s /
+1.14 s) and were wrong anyway, and session 2 answered the same question correctly ("Brazil
has won the World Cup five times"). This is the already-tracked hallucination-reinforcement /
paraphrase-not-facts failure mode in `project_retrieval_quality_findings`, now the biggest
thing standing between this demo and a good one. Escalate it.

One hypothesis worth testing rather than assuming, since it *would* be caused by the fix:
with sub-100 ms TTFAT the model's ungrounded lead ("it's quite a story", "question.", "you.")
enters `TurnManager.conversation_context` before grounding lands, and that context is what the
next retrieval sees — a tighter feedback loop than the old 4–7 s stall allowed, since back
then grounding always landed before the model spoke. Evidence for: turn 1 spoke 1.38 s before
grounding. Evidence against: turns 2–3 were grounded in time and still wrong. Unproven either
way.

## Phase C — reduce the surface that never needed to exist (no VM required)

### C1 — DONE (2026-07-30): STT-off-thread + Option E chain deleted

Deleted in its own single-purpose commit, after session 3 validated the onset
fix. Removed: `_run_stt_frames_sync`, `_patched_stt_send_audio`,
`_stt_off_thread_enabled`, `_patch_local_stt_off_thread`,
`_patch_compiled_functions_thread_safe`, `_patch_stt_no_cuda_graph`,
`_patch_stt_second_gpu`, `_patch_cuda_graph_thread_local`,
`_COMPILED_FUNCTIONS_LOCK`, `core/gpu.py`'s dual-GPU front-end visibility, the
`STT_OFF_THREAD` env var and its forwarding, and ~850 lines of tests. Three
now-unused imports (`functools`, `threading`, `contextlib.contextmanager`) and
six orphaned test helpers went with them.

**Kept deliberately: `_warm_up_stt_exec_mask()`.** It is called
unconditionally — never gated by `STT_OFF_THREAD` — so it is the only part of
the chain that ran on the default path, and keeping it is what makes this
deletion a genuine no-op on current behaviour rather than a behaviour change
smuggled into a cleanup. It survives on the latency argument alone
(pre-capturing `set_exec_mask`'s CUDA graph so a turn's first timed frame pays
a replay, not a capture); whether that still earns its keep single-threaded is
a separate, measurable question noted in its docstring.

`wb-gpu-a1ultra2g` now has no remaining justification and should stay stopped;
CLAUDE.md's "Remote instances" section says so up front instead of instructing
the reader to start it.

#### Original plan (historical)

### C1 — Formally deprecate the STT-off-thread + Option E chain

`_stt_off_thread_enabled()` already defaults to off (2026-07-29). By this repo's own
re-verification the eval path never needed it, and the entire three-crash CUDA-graph chain,
the second-GPU requirement, and the dropped-user-utterance bug all descend from it.

**Recommendation: mark deprecated now, delete in a separate later commit.** Do not couple a
large deletion to a latency fix — if the demo regresses for an unrelated reason, a
single-purpose commit is what makes it bisectable. In this pass: update the docstrings of
`_patch_local_stt_off_thread`, `_patch_compiled_functions_thread_safe`,
`_patch_cuda_graph_thread_local`, `_patch_stt_no_cuda_graph`, `_patch_stt_second_gpu`, and
`_warm_up_stt_exec_mask` to state that the mechanism they serve is now believed to be a
consequence of B1's patch and that the chain is dormant pending deletion.

Downstream of that deletion, when it happens: `core/gpu.py`'s
`ensure_cuda_visible_devices("frontend")` returning `"0,1"` exists only for Option E, and
`wb-gpu-a1ultra2g` has no remaining justification.

### C2 — Documentation and memory corrections

- `CLAUDE.md`: correction pointers on the three superseded sections (already added alongside
  this plan — verify wording once the VM results are in and upgrade "leading hypothesis" to
  "confirmed" or revise).
- `specs/moshirag-evals-requirements.md`: the `batch_size` rationale (B2), and any
  "Generation parameters" text implying `temp_text`/`top_k_text` are the relevant levers for
  response lag.
- Memories to update: `project_pad_token_sampling_investigation` (root cause found, not
  sampling), `project_demo_audio_quality_investigation` (the `liveBufferS ≈ 0` reading was
  backwards), `project_stt_cuda_graph_thread_safety` (chain now deprecated),
  `project_evals_implementation_status` (demo `ttfat_s` was measuring the wrong interval).

---

## Phase D — separate issues found in passing (each its own ticket, none is the delay)

| # | Issue | Notes |
|---|---|---|
| ~~D1~~ | ~~PAD vs EPAD invisible in logs~~ | **RESOLVED 2026-07-30** from session 1's `text_tokens`: PAD = 3, **EPAD = 0** (146/146 followed by a real word; PAD never is). See the session-1 findings above. The EPAD logit-bonus lever is now concretely actionable if ever needed |
| D2 | `"Reference: "` prefix leaking into the injected reference text | Seen in `07-29T05-50-41Z`, `07-10-18Z`, `19-03-58Z` turn 3 — the prefix is being ARC-encoded as content |
| D3 | Retrieval sometimes returns a restatement of the question, not a reference | `07-28T07-23-04Z`: `reference_text='The user is asking for more information about the World Cup.'` |
| D4 | `max_reference_tokens: 64` vs upstream's 512 | Confirm deliberate; at 4× ARC compression 64 tokens is ~16 injection steps, so this bounds how much grounding can land |
| D5 | `Channel._handle_reference_text` has no exception handling around the conditioner call | Already tracked in CLAUDE.md; still open upstream |
| D6 | Verify `gs://${GCS_BUCKET}/checkpoints/base/moshirag-base-bf16` == `kyutai/moshika-rag-pytorch-bf16` | Cheap; removes a variable from every future model-behaviour claim |
| D7 | Report the `--power-threshold` default gap upstream | `moshi.server` defaults `None`; `run_inference.py` and the paper's training both use `-65` |
| D8 | `power_threshold=-65` zeroes ~14% of 80 ms frames *inside* real user speech | Measure only for now. A higher or hysteretic gate may suit live microphone input better than the value tuned for TTS training audio |
| D9 | `demo/client` reports 22 npm vulnerabilities (5 moderate, 16 high, 1 critical) | Parked 2026-07-30, assessed non-blocking — see below. **Do not run `npm audit fix --force`** before the VM trip |
| D10 | **Metallic artifact at every model-utterance onset** — client playback drop threshold started at its floor and was learned by failing | **FIX IMPLEMENTED 2026-07-30, needs a listen** — see below |

### D10 — metallic artifact at each utterance onset (fix implemented, needs a listen)

**Root cause**, from session 3's `client_audio_diag_*.json` — the first such
capture taken on a session whose server side is fully understood:

`audio-processor.ts` starts *both* of its adaptive buffer parameters at the
floor of an 80 ms range and only ever raises them **when a failure occurs**:

- `maxBufferSamples` (10 ms, → drop threshold `totalMaxBufferSamples()` = 100 ms)
  grows by 5 ms only inside the drop branch — each increment paid for with a
  real discard of 18–69 ms of speech.
- `partialBufferSamples` (10 ms) grows by 5 ms only inside the underrun branch.

So reaching a workable threshold takes 14 audible glitches, and every session
re-learns from scratch. Measured: **56 drops, 5 underruns, 6 resumes**, with
all five drop bursts landing within ±0.31 s of a model turn onset (onsets at
client-clock 4.5 / 12.2 / 28.3 / 42.7 / 57.8 s; bursts at 4.79 / 12.34 /
28.02 / 42.48 / 57.81 s). That is precisely "metallic at the start of each
utterance, then it smooths out", and it is present in every session ever
recorded — unrelated to pacing, batch size, or anything in the latency fix.
It is very likely the real cause of the long-running "metallic/buzzy audio"
thread in `project_demo_audio_quality_investigation`, which was previously
misattributed to step pacing.

**Fix**: start `maxBufferSamples` at its own ceiling
(`maxMaxBufferWithIncrements`, 80 ms) in both the constructor and
`initState()`. One value, in two places.

Why this costs no steady-state latency: `totalMaxBufferSamples()` is the
*drop threshold*, not the operating depth. Depth is set by the arrival/drain
balance — measured `liveBufferS` sat at 70–81 ms while the threshold was
100–170 ms. Raising the threshold only stops the trimming, so burst audio
gets played slightly later rather than discarded, which is strictly better
than a glitch. The increment machinery is left in place (it is upstream's,
and is simply a no-op at the ceiling), so this stays a one-value revert.

`partialBufferSamples` is deliberately **not** raised: unlike the other one it
does cost real latency (`start()` makes playback wait that long before
resuming, on every stream start and underrun recovery), and at 5 underruns vs
56 drops it is a minor contributor. Revisit only if underruns dominate once
the drop fix lands.

**Validation**: cannot be verified locally — there is no Node toolchain here
and no client test suite (`vitest` is configured but `demo/client/src` has no
`*.test.ts`). Needs `npm run build` on the VM plus a listen. Expected: drop
events fall from ~56 to near zero, `liveBufferS` stays in the same 70–81 ms
band, and the per-utterance onset artifact goes away. If artifacts persist
with drops at zero, the remaining source is the underrun path and
`partialBufferSamples` becomes the next lever.

### D9 — npm vulnerabilities in `demo/client` (parked, non-blocking)

Observed on the VM during step 0c on 2026-07-30: `npm install` reports 22 vulnerabilities
(5 moderate, 16 high, 1 critical) against 431 audited packages. The build itself succeeds
(`tsc && vite build`, 104 modules, `dist/` written).

**Why this is being parked rather than fixed now:**

- The dependency split is 8 runtime `dependencies` (`react`, `react-dom`,
  `react-router-dom`, `opus-recorder`, `webm-duration-fix`, `zod`, `eruda`, `ws`) against 24
  `devDependencies` whose transitive tree (`vite`, `esbuild`, `@swc/core`, `eslint@8`,
  `prettier-eslint`, `typescript-eslint@7`, `vitest`, tailwind/postcss) is where a count like
  this normally originates. `npm audit` counts the whole tree regardless of whether anything
  reaches the shipped bundle.
- The known CVE clusters for the pinned versions here are dev-server-only —
  `vite@5.4.6`'s reported issues are in its dev/preview file server and
  `esbuild@0.21.5`'s is dev-server CORS. **This project never runs `vite dev`**:
  `run_demo.sh` runs `npm run build` and `moshi.server` serves the static `dist/` via
  `--static`, so that surface does not exist in the demo path.
- Threat model: single-user, `localhost`-only via SSH/IAP tunnel (the VM has no external
  IP), no untrusted input, no credentials in the client beyond a WebSocket to localhost.

**The one command that turns this from a guess into an answer** (5 seconds, run on the VM):

```bash
cd demo/client && npm audit --omit=dev
```

That reports only advisories reachable from runtime `dependencies`. If it returns zero, every
one of the 22 is build tooling and the shipped bundle is unaffected — which is the expected
result and would justify closing this as "assessed, no action". If it returns anything,
especially the critical, that finding is worth acting on and should be re-triaged.
`npm audit --json > audit.json` is worth capturing at the same time so the specific
advisories are on record rather than inferred.

**Why not `npm audit fix --force` now:** it bumps majors (`vite`, `eslint`) and can break
`npm run build`. Breaking the client build immediately before a validation trip whose step 0c
exists specifically to prove the build works would be the worst possible timing. If a
dependency refresh is wanted, do it as its own change with its own build+lint+`vitest run`
verification, not as a side effect.

**Two cosmetic notes from the same output**, neither blocking: `caniuse-lite` is outdated
(`npx update-browserslist-db@latest`), and npm's `allow-scripts` prompt lists
`@swc/core`/`esbuild` postinstall scripts — both are expected build-tooling postinstalls, and
the packages are already installed and working, so no approval is required to proceed.

---

## VM validation runbook — one trip, three sessions

Single-GPU `wb-gpu-a1ultra` is sufficient; nothing here needs `wb-gpu-a1ultra2g`. Budget
~15 min setup plus ~10 min per session. Read the whole of step 0 before starting — two of
its checks exist because skipping them has already cost real round trips.

### Step 0 — pre-flight (local, then on the VM)

**0a. Sync.** No commit needed; `sync` rsyncs the working tree (`.git` is excluded).

```bash
make sync
```

**0b. Clear stale state on the VM.** A leftover tmux server or orphaned python process
holding GPU memory is the most common cause of a session that loads and then behaves
strangely.

```bash
make ssh
tmux ls || echo "no tmux server — good"
tmux kill-server 2>/dev/null || true      # only if nothing else you care about is running
nvidia-smi                                 # expect ~0 MiB used, no python processes
```

`run_demo.sh` does `tmux kill-session -t demo` itself, but that does not reap a stray
conditioner or eval process from an earlier session, and it does not free their VRAM.

**0c. Confirm the web client still builds.** `run_demo.sh` runs `npm install && npm run
build` and **aborts the whole launch if the build fails**. The client changes from the
2026-07-27 session (`getAudioDiagLog`/`liveBufferS` in `useServerAudio.ts`) were written
without any Node toolchain available and are recorded in CLAUDE.md as *not build-verified*.
Prove it builds before spending a session on it:

```bash
cd /home/jupyter/moshirag-evals/demo/client && npm install --no-audit --no-fund && npm run build
```

**0d. Check `.env`.** `GEMINI_API_KEY`, `GCS_BUCKET`, `GCP_PROJECT` must be set (see
CLAUDE.md's first-time-setup Step 7). `run_demo.sh` fails fast on a missing key.

**0e. Do not run an eval concurrently.** Session 1 uses `--batch-size 16`, which with the
conditioner reaches ~66–69 GB of the A100's 80 GB. A second model load will OOM.

### Step 1 — the three sessions

Same question script every time, so the sessions are comparable. Reuse the sequence already
in `18-37-07Z`/`19-03-58Z`:

1. *"Hey, can you tell me about the history of the World Cup?"*
2. *"Yeah, it is. And then can you tell me more about the history of football?"*
3. *"And then how about the history of basketball?"*
4. *"What about the history of tennis?"* — a fourth turn matters: the growth effect needs
   **≥4 turns and ≥2 minutes** to separate from turn-to-turn noise.

Wait for each answer to finish completely before asking the next, and use **headphones**.
(Acoustic causes are ruled out — §5.1 — but leaving speaker bleed in the recording makes the
acoustic cross-check in step 3 harder to read.)

```bash
# Session 1 — reproduce the regression, instrumented (known-bad config)
DEMO_QUEUE_DIAG=1 DEMO_STEP_PACING=1 bash scripts/run_demo.sh --checkpoint base --batch-size 16

# Session 2 — isolate pacing (only variable changed vs. session 1)
DEMO_QUEUE_DIAG=1 bash scripts/run_demo.sh --checkpoint base --batch-size 16

# Session 3 — final config, both new defaults
DEMO_QUEUE_DIAG=1 bash scripts/run_demo.sh --checkpoint base
```

Leave `STT_OFF_THREAD` **unset** in all three (defaults to 0). Changing it would confound
the comparison, and the Option E chain is deprecated (C1).

**1a. Verify the config actually took effect before talking to it.** `run_demo.sh` now
echoes the session dir, batch size and every forwarded diagnostic var; the server prints its
own banner too. Both must agree:

```bash
tmux attach -t demo    # Ctrl-b 1 for the server window; Ctrl-b d to detach
```

Expected in the server window's first lines, per session:

| Session | `[QueueDiag]` | `[Pacing]` | `[STT]` | `--batch-size` |
|---|---|---|---|---|
| 1 | `DEMO_QUEUE_DIAG=1 -- writing step_diag.jsonl every 12 steps` | `DEMO_STEP_PACING=1 -- ... ENABLED` | `STT_OFF_THREAD=0 (default)` | 16 |
| 2 | same | `DEMO_STEP_PACING=0 (default) -- disabled` | same | 16 |
| 3 | same | `DEMO_STEP_PACING=0 (default) -- disabled` | same | 1 |

If a `[...]` line is missing or shows the wrong value, **stop and fix it** — a silently
wrong-mode session is exactly the failure this trip is meant to avoid. (Env vars are now
spliced into the tmux command explicitly; before 2026-07-30 an exported var could silently
fail to reach the process, which is what corrupted the 07-29 A/B.)

Model loading takes a few minutes with no output. `Access the Web UI directly at
http://localhost:8998` means it is ready.

**1b. Tunnel and connect.** From the Ubuntu VM:

```bash
ssh -L 8998:localhost:8998 wb-gpu-a1ultra
```

From the Mac (see CLAUDE.md's "macOS host + Ubuntu VM guest" section):

```bash
ssh -t -L 8998:localhost:8998 <ubuntu-user>@$(hostname -I on the Ubuntu VM) \
  "ssh -N -L 8998:localhost:8998 wb-gpu-a1ultra"
```

Then open **http://localhost:8998**.

**1c. Capture the two client-side artifacts before closing the tab.** Neither ever reaches
the VM, and both are lost on reload:

- **"Save audio"** → `moshirag-audio_startT<X>s-<timestamp>.webm`. Keep the filename
  intact — `scripts/analyze_recorded_audio.py` parses the `_startT` tag for alignment.
- **"Download audio diagnostics"** in the audio-stats panel → `client_audio_diag_*.json`.
  Rename to `client_audio.json`.

### Step 2 — per-session sanity checks (on the VM, before moving to the next session)

Catch a dud session now rather than after the trip. `SD=demo/sessions/<session-id>`:

```bash
SD=demo/sessions/2026-07-30T__-__-__Z

wc -l $SD/step_diag.jsonl          # ~1/s of session: expect ~100-200 for a 2-min session
head -2 $SD/step_diag.jsonl        # startup: lag_frames should be sizeable here
tail -2 $SD/step_diag.jsonl        # end of session: this is the number that matters
grep -c "LM buffer empty" $SD/server.log        # 0 means no turns registered — bad session
grep -c "RAG token" $SD/server.log              # expect ~3-4
uv run scripts/count_pad_stalls.py $SD/
uv run --all-extras scripts/summarize_demo_session.py $SD/
```

`summarize_demo_session.py` now prints a `pad` column and a `pad stall (steps)` aggregate
alongside `ttfat`. Sanity check the A2 fix here: **`ttfat` should now be seconds, not
~0.1–0.3 s**, and `pad × 0.08` should roughly equal it. If `ttfat` still reads ~0.1 s while
`count_pad_stalls.py` reports runs in the tens, the anchor is not firing — check that
`_vad_says_user_stopped` still mirrors upstream's `vad_neg`.

### Step 3 — pull everything back

```bash
make pull-sessions                 # new target: server.log, turns.jsonl, raw_events.jsonl,
                                   # step_diag.jsonl, conditioner.log for every session
```

Then hand-copy the two client-side files from the browser machine into the matching session
directory (they are on the Mac, not the VM):

```
demo/sessions/<session-id>/moshirag-audio_startT<X>s-<timestamp>.webm
demo/sessions/<session-id>/client_audio.json
```

**What to hand over for analysis:** the three session directories, complete. Minimum viable
set is `step_diag.jsonl` + `server.log` + `turns.jsonl` per session; the `.webm` and
`client_audio.json` are for the acoustic/playback cross-check and are nice-to-have on
session 3 only.

### Expected results

| # | Config | Expected if the diagnosis is right |
|---|---|---|
| 1 | pacing on, batch 16 | Reproduces 4 s → 8 s onsets. `lag_frames` climbs to ~50 during startup and **never returns to ~0**, growing through the session. `work_ms` < 80, `mean_period_ms` ≈ 80 |
| 2 | pacing off, batch 16 | Onsets collapse to 0.1–1.9 s and stop growing. `lag_frames` drains to 0–2 within the first ~15–30 s |
| 3 | pacing off, batch 1 | Same as 2 with a larger margin: `work_ms` well under 80, `lag_frames` at 0–1 almost immediately |

Read **`lag_frames`**, not `qsize` alone: it is derived from
`LocalSpeechToText.sent_samples`, the live `_recv_loop` counter, so it stays valid even if
audio backs up in the WebSocket or Opus reader rather than in `input_queue`. Multiply by
80 ms for wall-clock lag.

Also check, in every session: no `AcceleratorError` in `server.log`; and subjective fluency
during the model's own speech, since B1 removes the pacing that was believed to be smoothing
playback (if playback degrades in sessions 2–3, that is a genuine finding and belongs in
B3's decision, not a reason to restore pacing).

### If session 1 falsifies the hypothesis

`lag_frames` flat at 0–2 with pacing on means the backlog mechanism is wrong. **Stop; do not
draw conclusions from sessions 2 and 3**, and revert nothing yet — B1/B2 are independently
defensible (the pacing premise is still false; batch 16 still wastes compute), but they are
then not the fix. The next lever is the PAD/EPAD logit bonus (regression doc §3), and
`step_diag.jsonl`'s `text_tokens` field already carries the raw token stream needed to size
it — that is why D1 was folded into A1 rather than deferred.

Also check in every session: corrected `ttfat_s` in `turns.jsonl` now tracks the acoustic gap
(cross-check against a saved client recording with `scripts/analyze_recorded_audio.py`); no
`AcceleratorError`; and subjective fluency during model speech, since B1 removes the pacing
that was believed to be smoothing playback.

**If session 1 shows `qsize()` flat at 0–2** the backlog mechanism is wrong. In that case stop,
do not proceed to B1/B2, and pivot to the PAD/EPAD logit-bonus lever (regression doc §3) —
with D1 as its prerequisite.

---

## Open decisions needed before implementation

All four resolved 2026-07-30:

- **D-1. `ttfat_s` redefinition** — *redefine in place*, matching the paper's §3.1
  definition, with the old quantity preserved as `turn_switch_to_first_token_s`. Sessions
  logged before this change carry no `pad_stall_steps` field, which is the marker that their
  `ttfat_s` used the old anchor; `scripts/summarize_demo_session.py` says so explicitly rather
  than silently mixing the two.
- **D-2. Scope** — *one drop*, A/B by env toggle on a single VM trip.
- **D-3. Option E** — *deprecate now, delete later*, in a separate single-purpose commit once
  the A/B confirms the onset fix.
- **D-4. Phase D** — *keep separate*. D1 came along free with A1 (see Status above); D2–D8
  stay tracked here, unimplemented.
