# Demo turn-onset regression — independent re-analysis (2026-07-30)

**Status: RESOLVED and VM-validated (2026-07-30), merged to `main`.** Written from a
deliberately fresh review of the whole demo path — upstream source, both papers, and every
saved demo session in `demo/sessions/` — without assuming any prior conclusion in this repo.

§8 below asks for one decisive measurement before acting. It was taken, the diagnosis held,
and the fix was validated across three A/B sessions:

| | before | after |
|---|---|---|
| `ttfat_s` per turn | 4.45 / 0.07 / 3.37 / 7.10 s | **0.036 / 0.200 / 0.000 / 0.036 s** |
| client audio discarded | 5.334 s (7.2% of received) | **0.158 s (0.2%)** |
| step periods > 100 ms | 4 per session | **0** |

Two further defects surfaced during validation and were fixed alongside — **one upstream's,
one ours**:

- **Upstream's, latent — and the dominant cause of the artifact on *every* turn, not just the
  first:** `Channel.input_queue` is unbounded, so *any* backlog gets drained at ~1.78× real
  time, flooding the browser, which discards the surplus. Bounding it alone (B3) removed 81%
  of later-turn audio loss and 96% of the greeting's, before the blocking call below was
  touched. This is the long-standing "metallic audio" artifact, and it **predates this repo's
  patches** — the amplifier was always there, and the startup model load is a trigger no
  deployment avoids. See §7.2.
- **Ours:** `retrieval_backend.retrieve()` called synchronously inside an `async def`, blocking
  the event loop 0.45–0.63 s per `<ret>`. This is one of the *triggers* the unbounded queue
  amplified, plus a playback stall (underrun) of its own on every retrieval turn. Upstream's
  own equivalent properly `await`s; this repo's patch replaced it with a blocking call.

See [`demo-turn-onset-fix-plan.md`](demo-turn-onset-fix-plan.md) for the full arc, the
per-cause attribution table, and three conclusions of mine that measurement reversed along the
way.

**This document supersedes the framing of three CLAUDE.md sections.** Those sections are
kept in place, not rewritten, with correction pointers back here:

| CLAUDE.md section | What was wrong |
|---|---|
| "Real root cause of demo response lag: model-generation `<pad>` sampling drift" (2026-07-28) | The `<pad>` stall is real and correctly measured, but it is **not** learned model behaviour. It is a regression introduced by this repo on 2026-07-27. |
| "Real-time step pacing (added 2026-07-27)" | `_patch_server_state_step_pacing()`'s stated premise is provably false, and the patch is the leading cause of the regression. |
| "Client-side audio jitter-buffer diagnostic" / audio-quality thread | `liveBufferS ≈ 0 ms for 99.3% of samples` was read as "the jitter buffer is genuinely healthy now". A permanently-empty client buffer is the signature of a **starved** client, i.e. a server with zero real-time slack. |

---

## 1. Summary

The user-visible symptom — a long silence between the end of the user's question and the
model starting to speak, worst on retrieval turns — is:

1. **Not a retrieval problem.** Retrieval (0.6–0.8 s) and ARC encoding (0.2–0.5 s) both
   complete *inside* the gap and are off the critical path entirely.
2. **Not a RAG problem at all.** Stalls of the same magnitude occur on non-retrieval turns
   in the same sessions. This is front-end Moshi **turn-onset (turn-taking)** behaviour.
3. **Not learned model behaviour.** The paper measures TTFAT = 0.0 s and Full-Duplex-Bench
   turn-taking latency = 0.18 s for this exact model, and this repo's own sessions from
   2026-07-22/23 measure 0.1–1.9 s on the same stack.
4. **A regression this repo introduced on 2026-07-27**, whose leading cause is
   `scripts/instrumented_server.py`'s `_patch_server_state_step_pacing()`.

It went undiagnosed for weeks because **the demo's own `ttfat_s` metric starts its clock
after the stall has already ended**, and because every other diagnostic built for this
observes either the STT/transcript path (which is live and healthy, and structurally tells
you nothing about the main model) or the output path (downstream of the fault). Nothing in
this repo has ever observed the front-end model's own *input*.

---

## 2. What the delay actually consists of

Session `demo/sessions/2026-07-29T19-03-58Z/`, turn 1 (`STT_OFF_THREAD=0`,
`DEMO_STEP_PACING=1`, `gemini_api_moshi_style` / flash-lite):

| Event | server.log clock | Δ |
|---|---|---|
| User's last word (`"...World Cup?"`) | `19:05:26.27` | — |
| 53 consecutive `[VAD] ... LM buffer empty, remaining in user turn` steps | `26.09` → `30.16` | **+3.89 s of `<pad>`** |
| `[RAG] model emitted RAG token` | `30.16` | |
| First real text token (`'Sure'`) — **audible onset** | `30.56` | +0.40 s |
| `[RetrievalBackend] ... backend_latency=0.702s` | `31.26` | *already inside the gap* |
| `[Reference] ARC encoding received in 0.532s (streaming_sum (1, 13, 4096))` | `31.82` | *already inside the gap* |
| `[VAD] Waiting ended, switching to model` / `[Flushed LM buffer]` | `32.06` | *transcript display only* |

Two structural facts make this decomposition exact:

- **`Channel._output_loop` sends `out.pcm` over the WebSocket unconditionally, before the
  text branch.** `stt_wait_steps`, the VAD re-confirmation dip, and `[Flushed LM buffer]`
  gate only the *transcript*. User-perceived onset = the model's first non-pad text token.
- **Generation is never gated on retrieval** (paper §3.2), so retrieval/ARC latency can only
  matter if it exceeds the onset gap. It does not, by a wide margin.

All three turns of that session, measured end-of-user-speech → first audible token:
**4.3 s, 18.6 s, 7.4 s.** Pad before `<ret>`: **3.89 s, 17.13 s, 5.97 s** — i.e. 90–96% of
the wait happens before the model even asks for retrieval.

### 2.1 It is not specific to retrieval turns

Stall runs terminating in an ordinary word rather than `<ret>`, same sessions:
**17, 57, 81, 98, 106, 117 steps** (1.2–9.4 s at 80 ms/step). Retrieval turns only *look*
worse because `<ret>` is a loud log marker and because knowledge-intensive turns tend to
occur later in a conversation, where the effect has compounded (§4).

Paper Figure 4 confirms `<ret>` sits immediately before the first lead token in the training
template — there is no gap there by design, and the measured `<ret>` → first-word interval
is a healthy 0.24–0.88 s. The pathology is entirely upstream of `<ret>`.

---

## 3. Ground truth from the papers

`MoshiRAG` (arXiv 2604.12928v3), Table 1:

| Model | TTFAT | Keyword delay | E2EKD |
|---|---|---|---|
| MoshiRAG (Gemma 3 27B back end) | **0.0 s** | 3.1 s | 3.1 s |
| Vanilla Moshi | 0.0 s | 2.1 s | 2.1 s |

Table 2 (Full-Duplex-Bench), Turn Taking track: MoshiRAG latency **0.18 s**, TOR 0.83.

§3.1 defines TTFAT as "the delay between the end of a user's utterance and the moment the
model generates the first audio token of its response", and states the design assumption
explicitly: *"assuming that the retrieval is not triggered before the user query finishes,
the retrieval delay must be shorter than the E2EKD"*. So `<ret>` is expected at/near the
end of the user's query — which is what the 2026-07-22 sessions show and what current
sessions violate by 20–100×.

`Moshi` (arXiv 2410.00037) §3.4.4 / Appendix additionally documents the *trained* lever for
turn onset, should one ever be needed: *"forcing the sampling of a EPAD token will make
Moshi start talking immediately"*, and the paper itself applies "a small bonus on their
logits" to PAD/EPAD for rate control in TTS mode. Note `Channel._decode_text_token` lumps
token ids `0,1,2,3` together, so PAD vs EPAD is not currently distinguishable in our logs —
a prerequisite for that lever.

---

## 4. The regression: dates, magnitudes, mechanism

### 4.1 Onset latency vs. elapsed session time

"Onset" here = duration of the unbroken `LM buffer empty` run immediately preceding the
model's first real token (or `<ret>`), i.e. the pad stall itself.

| Session | `DEMO_STEP_PACING` | t≈15–25 s | t≈35–70 s | t≈85–155 s |
|---|---|---|---|---|
| `2026-07-22T17-49-24Z` | *(patch did not exist)* | 1.9 s | 0.3 / 0.6 s | — |
| `2026-07-22T19-24-40Z` | *(did not exist)* | 0.8 s | 1.9 / 0.1 s | 0.3 s (t=155) |
| `2026-07-22T21-12-52Z` | *(did not exist)* | 1.0 s | 0.5 / 0.6 s | 0.5 / 0.1 s (t=116) |
| `2026-07-28T07-23-04Z` | **on** | 1.6 / 4.5 s | 4.0 s | 5.2 / 5.9 s |
| `2026-07-29T07-10-18Z` | **on** | 4.4 / 4.4 s | 5.2 s | 6.3 / 7.9 s |
| `2026-07-29T07-19-59Z` | **on** | 1.2 s | 5.2 / 3.7 s | 8.2 / 9.3 s |
| `2026-07-29T18-37-07Z` | **on** | 4.1 s | 6.9 s | 6.5 / 8.0 s (t=126) |
| `2026-07-29T19-03-58Z` | **on** | 4.1 s | 17.1 s (t=38) | 9.7 s |

Pre-patch: flat or **decreasing** with session time, 0.1–1.9 s throughout.
Post-patch: ~4 s floor, growing monotonically to 8–9.7 s.

**Every measured onset is either ≤ 1.9 s or ≥ 4.0 s, with an empty band between.** A
stochastic sampling drift produces a continuum; a latched queue produces two modes. This is
the strongest single argument against the `<pad>`-sampling-drift explanation.

### 4.2 Mean step period (direct measurement)

Measured from consecutive `LM buffer empty` timestamps (one such line = exactly one
`_output_loop` iteration = one model step), intervals < 250 ms only:

| Session | pacing | mean ms/step | headroom vs. 80 ms budget |
|---|---|---|---|
| `2026-07-22T21-12-52Z` | off | **60.0** (n=54) | 25% |
| `2026-07-23T19-53-38Z` | off | **65.7** (n=131) | 18% |
| `2026-07-22T19-24-40Z` | off | **78.7** (n=365) | 1.6% |
| `2026-07-28T07-23-04Z` | on | **80.5** (n=283) | ~0 |
| `2026-07-29T07-10-18Z` | on | **81.0** (n=365) | ~0 |
| `2026-07-29T07-19-59Z` | on | **79.9** (n=358) | ~0 |
| `2026-07-29T18-37-07Z` | on | **80.8** (n=342) | ~0 |
| `2026-07-29T19-03-58Z` | on | **80.7** (n=367) | ~0 |

Mimi's frame rate is 12.5 Hz → the real-time budget is exactly 80 ms per step. The demo's
real per-step work is 60–79 ms. That 1–25% headroom is the **only** mechanism in the whole
pipeline that can drain a backlog.

### 4.3 Why `_patch_server_state_step_pacing()` removes it

The patch's docstring asserts that "nothing anywhere in the pipeline paces output to
real-time" and that "the server ships audio strictly faster than real-time, continuously,
with nothing downstream to throttle it back down". **That premise is false as stated** — but
see the correction at the end of this section: the faster-than-real-time delivery it
describes is real, just transient rather than continuous, which is a materially more
sympathetic reading than this section originally gave it. From real
upstream source (`moshi/moshi/server.py`):

```python
def _gather_step_inputs(self) -> BatchInput[Any] | None:
    ...
    try:
        inp = occupant.input_queue.get_nowait()      # non-blocking
    except asyncio.QueueEmpty:
        continue
    ...
    if not active:
        return None                                  # -> run_one_step returns False

async def _step_loop(self):
    while True:
        ran = await self.run_one_step()
        if ran:  await asyncio.sleep(0)
        else:    await asyncio.sleep(0.005)           # idle poll
```

The step loop is **input-driven**: it cannot outrun the client's real-time audio upload. The
client's uplink *is* the throttle and `Channel.input_queue` is the buffer.

Given that, the patch behaves as a **one-way latch**:

```python
ran = await original_run_one_step(self)
if ran:                                  # True only when a frame WAS queued
    remaining = frame_period_s - (time.monotonic() - step_start)
    if remaining > 0:
        await asyncio.sleep(remaining)   # floor the period at 80 ms
```

- Queue empty → `ran is False` → no sleep. Harmless; the patch is invisible in the healthy case.
- Queue non-empty → drain rate is capped at exactly the production rate. **The backlog can
  never shrink**, and every step whose own compute exceeds 80 ms ratchets it further up.

Before the patch, the 60–79 ms drain rate made `input_queue` self-draining (absorbing at
zero) — a backlog was always transient. After it, drain capacity is zero.

**Correction (VM session 2, `2026-07-30T19-33-05Z`) — the patch's author saw something
real.** With pacing off, measured `mean_period_ms` is ~66 ms *while a backlog is draining*
and snaps to ~80 ms the instant `lag_frames` reaches 1. So during a drain the server does
consume an 80 ms frame every 66 ms and genuinely ships audio **~21% faster than real time** —
audible, and independently reported by the operator as "slightly fast, metallic artifacts
more noticeable", concentrated in the first ~25 s of the session.

So the honest statement is narrower than "the premise is false": faster-than-real-time
delivery is real but **transient, confined to backlog drain**, not continuous or inherent.
The patch mistook a transient for a steady state and suppressed it by guaranteeing the drain
never completes — trading a ~25-second startup audio artifact for a permanent multi-second
latency. Both problems are avoidable at once by *discarding* the startup backlog instead of
draining it; see `demo-turn-onset-fix-plan.md`'s B3 decision.

### 4.4 Where the initial ~4 s comes from

Measured `[Slot] acquired slot` → first model output, every session:

| pacing off | pacing on |
|---|---|
| 5.00 s, 5.58 s | 4.44, 4.62, 4.54, 4.85, 4.94, 4.73 s |

The client streams audio throughout this window (STT priming in
`LocalSpeechToText._run_codes`' `while self._frame_id < self._prime_cap` loop, CUDA-graph
capture, the first `max_delay` steps that return `None`), so ~50–60 frames queue up before
the model produces anything. That backlog exists in **every** session, pre- and post-patch.

- Pre-patch it was burned off within roughly the first 15–30 s of conversation — which is
  exactly why the 2026-07-22 onsets *decrease* over the session.
- Post-patch it becomes the permanent floor. **In 4 of 6 pacing-on sessions the first
  measured onset matches the startup window to within 0.6 s.**

### 4.5 Corroborating signals already in the repo, previously misread

- `liveBufferS = 0 ms for 99.3% of samples, max 37 ms` (session `2026-07-28T07-23-04Z`),
  recorded in CLAUDE.md as "the jitter buffer itself is genuinely healthy now". A
  permanently-empty playback buffer means the client is **starved** — the server is
  delivering strictly on demand with no slack.
- Within-generation word cadence 0.243 s → 0.309 s post-fix (**+27%**), recorded as "close
  to the pre-fix baseline". That is the same rate mismatch surfacing as stretched playback,
  and it is consistent with the user's "did not feel fluent" report.

### 4.6 Contributing factor: `--batch-size 16`

`run_demo.sh` never sets `--batch-size`, so the demo runs at `moshi.server`'s default of 16.
`BatchRunner.run_step()` computes the **full batch tensor every step** — `exec_mask` gates
state updates, not compute — so a single-user demo pays ~16× the necessary work per step
(logged `batched step (1/16 active) took 81.8–89.9ms`). The eval path already hardcodes
`batch_size=1`. This is the cheapest available way to buy back real-time headroom, and it
also makes the pipeline robust to the startup backlog without any queue surgery.

`specs/moshirag-evals-requirements.md`'s "Generation parameters" section currently justifies
leaving it at 16 as "the demo needs headroom for concurrent live sessions". That rationale
needs correcting: unused slots are not free, they are a per-step multiplier on the one
budget that matters.

---

## 5. Hypotheses eliminated with new evidence

### 5.1 Acoustics / audio plumbing — eliminated

The saved client recording for `2026-07-29T07-19-59Z` was decoded (`ffmpeg`, 120.4 s,
aligned via the filename's `_startT1.905s` tag + server.log's `new WebSocket client
connected` at `07:21:35.87`) and its 80 ms RMS envelope computed in dBFS:

| Window | median | fraction < −65 dBFS |
|---|---|---|
| stall run, 65 steps → `<ret>` | −77.5 dBFS | **100%** |
| stall run, 47 steps → `<ret>` | −77.9 dBFS | **100%** |
| stall run, 106 steps | −76.8 dBFS | **100%** |
| stall run, 117 steps | −77.8 dBFS | **100%** |
| *(reference: user actively speaking)* | −25.5 dBFS | 13.8% |

With `--power-threshold -65` in effect (this repo sets it; see §7.1), the front-end model is
being fed **exact digital zeros** for the entire duration of every stall. No noise floor, no
speaker→microphone echo, no gate chatter around the threshold. This eliminates the whole
audio-level / echo / AGC / gate-artifact family of hypotheses.

Independently reinforced by the Moshi paper's instruct-finetuning augmentation, which covers
precisely the live-microphone conditions: random gain **−24 dB to +15 dB** on the user
stream 50% of the time, Deep Noise Suppression noise 30% of the time, silent sections of up
to **30 s**, **emulated echo of Moshi into the user's microphone** (scale 0–0.2, delay
100–500 ms), and reverb. Base Moshi is explicitly trained to be robust here.

*Side observation, not the cause:* ~14% of 80 ms frames **inside** the user's real speech
also fall below −65 dBFS and are being zeroed. Worth knowing; it does not produce a
9-second wait after clean silence.

### 5.2 "Learned `<pad>` sampling drift" — refuted

Three independent ways: the paper's own TTFAT = 0.0 s / turn-taking latency = 0.18 s (§3);
this repo's own 2026-07-22 sessions at 0.1–1.9 s flat (§4.1); and the bimodal distribution
with an empty band between 1.9 s and 4.0 s (§4.1). The previously-noted correlation with
conversation depth is real but is a **consequence** of the ratchet in §4.3, not a sampling
property. `temp_text`/`top_k_text` correctly showed nothing, because they were never the
relevant variable.

### 5.3 Conditioning correctness — verified fine

Checked against paper Eq. 2 (`h'_i = h_i + h^ref_{i-(i_ret + d·f_r)}` inside the injection
window, `h'_i = h_i` outside it):

- `ConditionFuser.get_streaming_sum()` and `LMGen.apply_pending_streaming_sum_condition()`
  zero `condition_streaming_sum` outside the window, matching `h'_i = h_i`. Correct.
- `force_streaming_sum=True` supplies a zero tensor at init when the ARC conditioner is
  skipped (`skip_conditioners = ["reference_with_time"]` under `REFERENCE_ENCODER_URL`).
  Consistent — the log's `streaming_sum=(16, 1, 4096)` is exactly that forced zero tensor.
- Reference lengths in real sessions are T = 12–13 steps (≈1.0 s of streamed injection),
  from ARC-Encoder's 4× compression of ~50-token references. Well inside budget; the
  paper's own worst case ("a 250-token reference could correspond to a 20-second embedding
  sequence") is not being hit.

### 5.4 Not the driver: `STT_OFF_THREAD`

Sessions split cleanly by `STT_OFF_THREAD` at first glance, but that split is confounded:
the `STT_OFF_THREAD=1` sessions are the ones exhibiting the separately-tracked
dropped-user-utterance bug, so whole turns are missing from the stall statistics, and two of
them have n=2/n=15 usable intervals. The step-period measurement in §4.2 is the
non-confounded signal, and it tracks `DEMO_STEP_PACING`, not `STT_OFF_THREAD`.

---

## 6. Why this was never found: the instrumentation trap

### 6.1 The demo's `ttfat_s` clock starts after the stall

`scripts/instrumented_server.py`'s `_ChannelState.on_utterance_end()` sets `_ttfat_start`,
and is called from `_patch_turn_manager`'s wrapper when `TurnManager._update_active_speaker`
**returns** `"model"`. Following real upstream source, that return is only reachable via:

```python
elif vad_neg and current_speaker == "user":
    if self.model_text_buffer:            # <-- only non-empty once a real token was produced
        self._pending_speaker = "model"
        self._wait_counter = self.stt_wait_steps
    else:
        logger.info("[VAD] User stopped speaking but LM buffer empty, remaining in user turn")
```

The `else` branch **is** the stall, and it starts no clock. So `_ttfat_start` is set only
after the stall has ended *and* after the `stt_wait_steps` hold, and the metric then measures
the interval to the next non-pad token — one or two steps.

Consequence, on the exact session analysed in §2:

| Turn | `turns.jsonl` `ttfat_s` | real end-of-speech → audible |
|---|---|---|
| 1 | **0.1613** | **4.3 s** |
| 2 | **0.3216** | **18.6 s** |
| 3 | **0.0793** | **7.4 s** |

Both documented `ttfat_s` root-cause fixes in CLAUDE.md ("RESOLVED: `ttfat_s` always exactly
0.0") were on the **eval** path (`_TimedInferenceJob._finalize_ttfat`). The demo path's zero
point was never audited, and it is anchored to an event that by construction happens *after*
the delay users complain about.

### 6.2 The transcript and VAD prove nothing about the front-end model

Two independent audio consumers exist per connection:

```
_recv_loop:  ws -> opus_reader -> chunk
                 ├─ await self.stt.send_audio(chunk[0,0].cpu().numpy())   # RAW, separate 1B STT model
                 └─ input_queue.put(filter_by_power(chunk))               # -> front-end Moshi
```

`[Display User]` text and `TurnManager.update_vad` both come from the **STT** model, fed
directly in `_recv_loop`, never queued. So a perfect live transcript and
`history=['1.000','1.000','1.000','1.000']` are fully compatible with the front-end model
being seconds behind. Every VAD/transcript-based diagnostic in this repo is measuring the
healthy path.

### 6.3 The actual blind spot

`Channel.input_queue` depth, and step index vs. wall clock, have **never** been logged
anywhere in this repo. That is the one place the fault is directly visible.

The cost of that blind spot: retrieval-latency work, GPU-contention/MIG/dual-GPU
provisioning, event-loop starvation, the three-crash CUDA-graph chain, Option E, and the
jitter-buffer/pacing work. Each individual fix was real and correctly reasoned from the
evidence available. But together they address components summing to roughly a second of a
4–19 s delay — and the two largest (Option E's second-GPU requirement, STT-off-thread) exist
only to serve a hazard created by another self-inflicted patch.

---

## 7. Would upstream have worked out of the box?

For this specific symptom, **largely yes** — there is no pacing patch upstream, so its step
loop self-drains, which is what the 2026-07-22 sessions were measuring. But "out of the box"
is not what this repo runs. Real deviations, ordered by risk:

1. **`--batch-size` left at upstream's default 16** — see §4.6.
2. **`--stt local` instead of the README's `--gradium-stt`.** Upstream's recommended stack
   puts ASR on a remote service (and notes it "will reduce `stt_wait_time` automatically").
   Local STT adds a second 1B model + Mimi on the same GPU **and the same event loop** —
   which is what consumed the real-time headroom, and is the origin of the entire
   off-thread/CUDA-graph/Option-E branch.
3. **Retrieval back end.** Upstream recommends a local vLLM Gemma-3-27B and warns "MoshiRAG
   is sensitive to retrieval delays over 3 seconds". Current flash-lite latencies
   (0.6–0.8 s) are fine; the earlier `gemini-3.5-flash` figures of 1.8–3.6 s were genuinely
   past the paper's own 1.5 s degradation threshold — a real but separate quality problem.
4. **Checkpoint provenance.** `base` resolves to
   `gs://${GCS_BUCKET}/checkpoints/base/moshirag-base-bf16`, not verified byte-identical to
   `kyutai/moshika-rag-pytorch-bf16`. Worth confirming before attributing anything to model
   behaviour.

### 7.1 One upstream bug this repo already fixed

`moshi.server` defaults `--power-threshold` to `None`, while `run_inference.py` defaults it
to `-65`, and the paper **trains** with an 80 ms-window gate: *"Audio segments with a
root-mean-square level below −65 dBFS are zeroed out"* (§4.2). Upstream's README server
command omits the flag entirely, so the reference live deployment feeds the front-end
un-gated audio, unlike training. This repo sets `-65` via `_DEFAULT_GENERATION` /
`print_demo_env.py`. Worth reporting upstream.

### 7.2 One latent bug upstream shares

The 3.9–4.9 s startup backlog (§4.4) exists regardless of the pacing patch. Upstream
survives it on headroom alone. There is no queue cap, stale-frame drop, or resync anywhere
in `_recv_loop` / `input_queue`, so on slower hardware or with several slots genuinely
active, upstream would latch the same way.

---

## 8. The one decisive experiment still missing

Everything in §4 is strong circumstantial evidence — a coincident date, a crossed
step-period threshold, a quantitative match between the startup window and the initial
stall, a bimodal distribution, a ratchet mechanism that predicts the observed growth, and a
starved client buffer. **It is not proof.** The server logs cannot distinguish "the model
heard the question 4 s late" from "the model heard it on time and chose to wait 4 s".

The distinguishing measurement is `Channel.input_queue.qsize()` (plus step index vs. wall
clock), sampled per step and written to the session log. One live demo session settles it:

- **Backlog hypothesis correct** → `qsize()` climbs to ~50 frames during startup, never
  returns to 0 with pacing on, and grows through the session; and collapses to 0–2 with
  `DEMO_STEP_PACING=0`.
- **Backlog hypothesis wrong** → `qsize()` sits at 0–2 throughout even with pacing on, and
  the stall is genuinely the model declining to speak on time-aligned input. In that case
  the next lever is the PAD/EPAD logit bonus (§3), not sampling temperature.

Do this **before** changing anything else. See `docs/demo-turn-onset-fix-plan.md` for the
sequenced work.

---

## Appendix — reproducing the measurements

All of these run against committed session logs (`demo/sessions/` is gitignored; these were
run against the local working copy) and need no GPU.

Stall runs and their terminators, per session:

```bash
sed -e 's/\x1b\[[0-9;]*m//g' demo/sessions/<ID>/server.log | awk '
function ts(x){split(x,t,":"); return t[1]*3600+t[2]*60+t[3]}
/new WebSocket client connected/ {t0=ts($1)}
/LM buffer empty/ {if(stall==0)stallstart=ts($1); stall++}
/RAG token|buffering model text/ {
  if(stall>0) printf "t_sess=%5.1fs stall=%3d (%4.1fs) %s\n",
    stallstart-t0, stall, ts($1)-stallstart, ($0~/RAG/?"ret":"word");
  stall=0 }'
```

Mean step period (one `LM buffer empty` line = one model step):

```bash
sed -e 's/\x1b\[[0-9;]*m//g' demo/sessions/<ID>/server.log | grep "LM buffer empty" | awk '
function ts(x){split(x,t,":"); return t[1]*3600+t[2]*60+t[3]}
{s=ts($1); if(p>0 && s-p<0.25){tot+=s-p; n++} p=s}
END{printf "mean_step=%.1f ms (n=%d)\n", tot*1000/n, n}'
```

Startup backlog window:

```bash
sed -e 's/\x1b\[[0-9;]*m//g' demo/sessions/<ID>/server.log | awk '
function ts(x){split(x,t,":"); return t[1]*3600+t[2]*60+t[3]}
/acquired slot/ {slot=ts($1)}
/Flushed LM buffer|Display Model/ {if(f==0 && slot>0) f=ts($1)}
END{printf "slotAcquired->firstOutput=%.2fs\n", f-slot}'
```

Acoustic envelope of the saved client recording (needs `ffmpeg`; see CLAUDE.md's note on the
user-writable `apk --root` prefix used for this):

```bash
ffmpeg -i demo/sessions/<ID>/moshirag-audio_startT<X>s-*.webm -ac 1 -ar 24000 -f f32le rec.raw
# then: 80 ms (1920-sample) window RMS -> dBFS, aligned to
# server.log's "new WebSocket client connected" + <X> seconds
```

`scripts/count_pad_stalls.py` already automates the stall-run counting and remains valid.
