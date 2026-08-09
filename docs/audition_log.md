# Audition log — every live session, indexed

Raw serve logs live in `replay/auditions/logs/` named `<date>_audition_<model>_<note>.log`.
Convention going forward: after every session, `cp ~/serve.log
~/moshi-finetune/replay/auditions/logs/<date>_audition_<run>_<ckpt>_<note>.log` on the box,
then rsync down. One line per session here, newest last.

## Serving-stack caveats that contaminate early entries

- **All sessions before ~03:00 UTC 08-09 ran WITHOUT the Danielle persona template** — the
  backend answered personal-question fires with "I am Moshi, an AI assistant" references.
  Identity verdicts from those sessions are lower bounds only.
- Track A "overlay" sessions ran the plain-moshika adapter ON the moshika-rag base — proven
  invalid for voice/filler judgment (trigger head = noise, injections crash sessions).
- Several sessions were cut by wifi/service-worker drops (5 Mbps link; "maximum capacity"
  page = stale service worker, not the server).

## Sessions — Sat/Sun 2026-08-08/09 (times UTC)

| # | Time | Model · ckpt | Stack | Duration | Trigger | Voice | Identity | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | ~01:10 | **4a · 600** (r64, 4e-6, 100s) | rag+enc, no template | **died 1:39** | 1 fire, grounded WC ref, round trip ~2.3 s | few words only | — | severe stalls from t=15s; 99s ceiling CONFIRMED |
| 2 | ~01:26 | **4b · 300** (r32, 2e-6, 300s) | rag+enc, no template | **5+ min** ✅ | **14 fires**, correct grounded chain (1930 Uruguay, Brazil ×5), declines weather | ❌ AI voice | ❌ "AI" (template-contaminated) | fluent; audio cut in/out (wifi); the behavior champion |
| 3 | ~01:5x | **4a · 400** | rag+enc, no template | 3+ min (user ended) | 0 fires | ✅ Danielle | freezes on "what's your name" | confabulates factuals; light-dose coasts past 100s |
| 4 | ~02:0x | **4a · 500** | rag+enc, no template | ~3 min | 0 fires, 239 stall lines | ✅ Danielle | ❌ no personal info | same profile as 400 |
| 5 | ~02:11–02:51 | **Track A · 2000 overlay** (r128 on rag base) | rag, **encoder killed**, no template | sessions die ~2 s after any fire | fires on EVERYTHING incl. "hi" and name (noise trigger) | (invalid stack) | Gemini refs literally "My name is Moshi" | proves: template missing + dead encoder kills sessions + overlay invalid |
| 6 | pending | **4b · 300 retest** | rag+enc, **template FIXED** | | | | **the money test**: name → Danielle-facts reference? | |
| 7 | pending | **Track A · 2000 mainline** | plain moshika base, no RAG stack | | n/a by design | **the ums question** | name test | first valid Track A audition |

## Standing findings from tonight

1. **99 s ceiling real & dose-coupled**: heavy-dose 100s adapters die at trained horizon (4a-600 @ 1:39); light-dose coast (4a-400 3+ min); 300 s training clears it (4b 5+ min).
2. **Loss mask works**: triggering restored from Stage-3 coin-flip to abundant (4b: 14 grounded fires).
3. **Voice needs rank-64 dose**: present at 4a-400/500, absent in all 4b.
4. **Identity in weights = half-written at best** (freeze on name at full dose); persona-by-retrieval unblocked by the template fix (verify in session 6).
5. **lr 4e-6 kills the trigger by step 400** (4a family); 2e-6 preserves it (4b family). Rank contribution untested (4a-200 probe skipped).
6. **Serving stack must be treated as a fixture**: base-control session first, template verified, encoder up, no rsync during sessions, hotspot > 5 Mbps wifi, fresh private window per session.

## Sunday 2026-08-09 — 4c ladder walk (times UTC; all sessions rag+enc, Danielle template in BOTH prompt files after 16:5x fix)

| # | Time | Model · ckpt · scale | Duration | Trigger | Voice | Identity | Notes |
|---|---|---|---|---|---|---|---|
| 8 | 15:52+16:01 | 4c·600 @2.0 | dead | 1 WC fire | — | — | stall-wall, won't take turn (2 sessions, quiet box confirmed) |
| 9 | 16:2x | 4b·300 @2.0 (control) | fine | greeting junk-fire + good WC | AI | — | stack proven innocent; base-Moshi rambling in silence |
| 10 | 16:2x | 4c·300 @2.0 | fluent | late fires → speaks WRONG answer first ("That's right" to Florida, "LA area") | partial, ums | ✗ | non-responses under sustained probing |
| 11 | 16:41 | 4c·400 @2.0 | 2 stalls | 1st-ask fire on Florida-Q, mid-answer self-correct (LA→Brooklyn) | best @2.0 per ear | via fires only | "accountant" confabs on work |
| 12 | 16:36+16:56+16:57 | 4c·500 @2.0 ×3 | ok/parrot/ok | ~1 fire/session, high variance; houseplants fire clean | unrated | via fires only | parroting failure in s2 |
| 13 | 17:00 | **4c·600 @1.5 — LOCKED for Track B** | good | **3 fires/2.5min, both persona fires early+clean** (houseplants: fired before word finished, personality-rich answer) | **"sounds like Danielle"** | ✅ via retrieval | WC fire froze on delivery → loops filler; recovery = ask next question |

## Sunday standing findings

7. **Fire timing matures with dose**: 300 fires after answering (wrong answer spoken) → 400 mid-answer (self-corrects) → 500 before answering → 600 earliest (mid-question). Eq.3 delay training visible rung by rung.
8. **MOSHI_LORA_SCALING is a serve-time dose dial**: 600 unusable @2.0, best-of-sprint @1.5 (~450-step effective dose w/ 600's fire maturity). Voice survives 1.5.
9. **Template bug found+fixed**: code defaults prompt_style=original → reference_prompt_template.txt (had NO Danielle). Danielle block now in BOTH template files. First live persona reference 16:36 ("I grew up in Brooklyn…" spoken correctly).
10. **Session variance is large** — single sessions are coin flips; judge on 2-3.
11. **pkill -f "moshi.server" also kills server_conditioner** (the encoder assassin). Safe: pkill -f "moshi.server --hf".
12. **Freeze recovery**: a fresh question un-freezes; filler-loop ("I think that's about it") = frozen state marker.
