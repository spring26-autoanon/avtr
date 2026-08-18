# Serving experiment results

| date | exp | ckpt | scale | stt-wait | patch | native fires | backstops | grounded/asked | notes | log |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-11 | E0 | 4c2-700 | 2.0 | 2.0 | e0_session_ret | 1 synth | - | 0/1 | DESTABILIZED: lone t=0 synth ret -> 14s stall + unprompted monologue; preseed clobbered by empty-ref race (lost by 50ms). Verdict: session-start synthetic ask dead regardless of forcing. | replay/auditions/logs/2026-08-11_E0_4c2-700_destabilized.log |
| 2026-08-11 | E1 s1 | 4c2-700 | 2.0 | 2.0 | e1_silent_backstop | 1 | 4 | 1?/4 | Backstop mechanics perfect (engage ~2.5s, refs question-matched, dedup works). Brooklyn "win" ambiguous (leading-question echo). Houseplants: double-inject (backstop+native 3.5s apart) mid-utterance -> SCREECH. | replay/auditions/logs/2026-08-11_E1_4c2-700.log |
| 2026-08-11 | E1 s2 | 4c2-700 | 2.0 | 2.0 | e1_silent_backstop | 0 | 2 | 0/2 | Both notes correct + ignored: deflection then dead air. | replay/auditions/logs/2026-08-11_E1_4c2-700_s2.log |
| 2026-08-11 | E1 s3 | 4c2-400 | 2.0 | 0.5 | e1_silent_backstop | 0 | 1 | 0/1 | HEALTHY checkpoint destabilized: deflect -> 11s silence -> screech after unsolicited injection. | replay/auditions/logs/2026-08-11_E1_4c2-400_destabilized.log |

**E1 VERDICT: TOKEN-GATED, and stronger — unsolicited injection is actively destabilizing (stalls/screech on both a stall-prone and a healthy checkpoint). Silent backstop not viable. E2 (forced single ret at boundary) is required and decisive.**
| 2026-08-11 | E2 s1 | 4c2-700 | 2.0 | 2.0 | e2_forced_ret (v1) | 0 | 4 armed, 0 consumed | 0/4 | FIELD DEFECT: TTL 2.0s expired before any word token (deep-rung words come 4+s late or never); consume-on-word deadlocks on stalls. Redesign: consume-on-pad. | replay/auditions/logs/2026-08-11_E2_4c2-700_s1.log |
| 2026-08-11 | E2 s2 | 4c2-700 | 2.0 | 2.0 (delay 4.0) | e2_forced_ret v3 | 1 native follow-up | 4+ | 1/5 | Forcing mechanism PROVEN (engage->forced ret ~0.15s, native machinery clean). 700's delivery stalls remain the bottleneck; post-answer notes don't retro-correct. Greeting got backstopped ("?"). | replay/auditions/logs/2026-08-11_E2_4c2-700_s2.log |
| 2026-08-11 | E2 s3 | 4c2-400 | 2.0 | 0.5 (delay 4.0 STALE ENV) | e2_forced_ret v3 | 0 | 3 | 0/3 | Delay env didn't take (stale 4.0): fast wrong answers always beat the rescue; confirms post-answer notes don't reopen topics. | replay/auditions/logs/2026-08-11_E2_4c2-400_delaybug.log |
| 2026-08-11 | **E2 s4** | **4c2-400** | 2.0 | 0.5 (**delay 0.8**) | e2_forced_ret v3 | ~1 | ~7 | **5/7 grounded** | **THE WIN: engage ~0.9s, forced fire, note injected ~2s = ahead of her answer. Five consecutive note-grounded answers incl. the work question. First-asks sometimes race (re-ask lands); minor blips/repetition.** | replay/auditions/logs/2026-08-11_E2_4c2-400_s4.log |

**E2 VERDICT: PROVEN. Believed-ask restores note-reading and delivery on a healthy checkpoint when the note beats the answer (delay ~0.8s at 4c2-400). Composite system = healthy checkpoint + forced-fire backstop + short delay -> guaranteed grounded answers in her voice. 700 remains delivery-limited regardless (stall pathology, not gating). Polish items: backstop refractory window, first-ask race tuning, statement-gate already holds.**
| 2026-08-11 | E3-A | 4b-300 | 2.0 | 0.5 (delay 1.5) | e2_forced_ret v3 | 2 native (1 mid-question) | ~6 | mostly grounded | Non-interference QUALIFIED PASS: natives fire and complete (Florida 0.14s post-question, clean). BUG: mid-question native erased by flag-reset-at-arming -> duplicate force. Rescue works on 4b (library->Director). Late-session screech again (~9 injections/2min — cumulative-swap hypothesis). | replay/auditions/logs/2026-08-11_E3_4b-300.log |
| 2026-08-11 | E3-B v4.1 | 4b-300 | 2.0 | 2.0 (delay 1.5) | e2 v4.1 | 2 native, NO dups | several | mixed | v4.1 FIELD-CONFIRMED: mid-question natives not duplicated; attribution lines in log. Session content poisoned by opening hallucination ("Ruri" woodcarver) -> context cascade. | 2026-08-11_E3_4b-300_v41_ruri.log |
| 2026-08-11 | LOCK | 4c2-400 | 2.0 | 0.5 (delay 0.8) | e2 v4.1 | 4 native | 4 forced | ~7/8 grounded | Final session: natives+forced interleave cleanly, she asks questions back, all persona+factual grounded. E4: Bitcoin decline HALF-FLATTENED (confabulated number + disclaimer). Statements get no reply (backstop is ?-gated; 400's statement reflex thin). Voice session-variable. Opening hallucinations correlate with backstopped first exchange (Ruri/Borges) -> protocol: open with a statement. | 2026-08-11_LOCK_4c2-400_v41_final.log |

## SYNTHESIS (program complete, 2026-08-11 ~05:40 UTC)

**Mechanism verdicts:** E0 (session-start synthetic ask) = destabilizing, closed. E1 (unsolicited injection) = ignored AND harmful, closed. **E2 (believed-ask backstop) = PROVEN** — a forced single ret at the boundary, landing its note before the answer forms, yields grounded answers in the model's own delivery.

**LOCKED COMPOSITE CONFIG:** 4c2-400, MOSHI_LORA_SCALING=2.0, MOSHI_LORA_RANK=64, stt-wait 0.5, MOSHI_BACKSTOP_FORCE=1, MOSHI_BACKSTOP_DELAY=0.8, patch e2_forced_ret v4.1 (LOOKBACK 8.0, REFRACTORY 6.0).

**Success criteria:** (1) >=90% grounding: re-homed from 700 (delivery-pathology, unrescuable) to 400: ~80-85% first-ask, ~100% with one re-ask — NEAR PASS. (2) Non-interference: PASS (v4.1: natives clean, zero duplicates, exact attribution). (3) Declines: PARTIAL — confabulate-then-disclaim; sentinel mitigation designed, untested. (4) Backstopped-note latency ~1.6s post-question — PASS.

**Operating protocol (hard-won):** open with a STATEMENT not a question (backstopped first exchanges hallucinate: Ruri/Borges); question is the last thing you say; keep the ball moving (statements get no reply); short sessions; reconnect on a weird opening or flat voice roll; encoder-safe pkill only.

**Checkpoint law (confirmed across 6 rungs + 4b):** the backstop guarantees knowledge, never speech — it amplifies delivery-healthy checkpoints (400, 4b-300) and cannot rescue stall-pathology rungs (500/600/700/800). Checkpoint choice owns voice and delivery; the backstop owns grounding.

**The one remaining training run** (recommendation, evidence-based): rank 64, 4c recipe, targets = (a) fire-rate via synthetic question-variety (many voices/phrasings per answer), (b) contradiction clips (reference disagrees with memorized answer) to keep asking valuable through crystallization, (c) casual statement-response dialogue for conversational carry. Success metrics: backstop engagement rate driven down on the LOCK protocol; statement-responsiveness; ambient-reference clips NOT required (E2 works without them).
| 2026-08-11 | VOICEHUNT | 4c2-700 | 2.0 | 2.0 (delay 0.8) | e2 v4.1 | 0 native | 6 forced (3 sessions) | 0 delivered | CONCLUSIVE CLOSE: every note generated+injected, mouth still died 2-3 words in ("Yeah, I—"). Padding-attractor stall is checkpoint-intrinsic; backstop cannot rescue delivery. Also: encoder cold-start 1.74s cost the first note race. | (sessions in serve.log era 05:48) |
| 2026-08-11 | VOICEHUNT | 4c2-600 | 1.5 | 0.5 (delay 0.8) | e2 v4.1 | 2 native | 7 forced | ~8/9 grounded | Zero stalls, statement-carry, humor (martial-arts joke, pokemon cards), Bitcoin decline PASSED natively. Identity slips: "I am a chatbot", "I don't have vision", "haven't visited" — 1.5 attenuates identity. 1 confab (World Cup US wins). | 2026-08-11_voicehunt_600at15.log |
| 2026-08-11 | VOICEHUNT | 4c2-600 | 2.0 | 0.5 (delay 0.8) | e2 v4.1 | 0 native | 4 forced | 4/4 grounded | BEST VOICE OF SPRINT (user: um, so, likes). Identity restored (no chatbot slips), zero stalls, statement replies. Cost: wrong-first-then-correct (Florida "Yes I am", houseplants) — at 2.0 she answers before any note can land; only mid-question native fires beat the buffer, and there are none. | 2026-08-11_voicehunt_600at20.log |
| 2026-08-11 | CONFIRM | 4c2-600 | 2.0 | 0.5 (delay 0.8) | e2 v4.1 | 0 native | ~12 forced + 1 refractory skip | most grounded | Long-session confirm: voice+carry hold, refractory works on double questions, one dropped answer (favorite place — re-ask rescued). NEW FAILURES: Bitcoin decline FULLY flattened ("six thousand" + doubled down "yes that is the live price"); name split ("people call me Danielle sometimes... going by Moshi") — partly a ref-template bug (Gemini said "Danielle goes by Moshi", confused by moshi: transcript labels). | 2026-08-11_confirm_600at20_v41.log |

## AMENDMENT (2026-08-11 morning): voice-optimal config

The fire-vs-imprint election is now QUANTIFIED on one checkpoint: 4c2-600 natives 2 -> 0
going 1.5 -> 2.0 scaling. Voice, identity, and register all arrive at 2.0; the native
trigger and the trained decline both die there (decline survival rode on native fires).

REVISED LOCKS:
- DEMO PRIMARY: 4c2-600 @ 2.0, rank 64, stt-wait 0.5, BACKSTOP_FORCE=1 DELAY=0.8, v4.1.
  Best voice + ums + statement-carry + healthy delivery; 100% backstop-carried grounding.
  Known flaws: wrong-first-then-correct on fast identity answers; declines flatten (do not
  ask live-data questions in the demo, or demo the flatten deliberately); name question
  yields Moshi/Danielle split.
- DECLINE-SAFE FALLBACK: 4c2-400 same serve config (3/8 native, declines half-survive,
  weaker/variable voice, no statement-carry).
- 700/800: permanently closed (delivery-stall under ideal composite conditions).
- Ref-template fix for the name split (1 line): "the speaker labeled 'moshi' in the
  transcript IS Danielle" in both reference templates.
- LoRA-interpolation "650" (600/700 average) remains the untested voice dial if more
  timbre is wanted without 700's stalls.
