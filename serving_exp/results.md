# Serving experiment results

| date | exp | ckpt | scale | stt-wait | patch | native fires | backstops | grounded/asked | notes | log |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-11 | E0 | 4c2-700 | 2.0 | 2.0 | e0_session_ret | 1 synth | - | 0/1 | DESTABILIZED: lone t=0 synth ret -> 14s stall + unprompted monologue; preseed clobbered by empty-ref race (lost by 50ms). Verdict: session-start synthetic ask dead regardless of forcing. | replay/auditions/logs/2026-08-11_E0_4c2-700_destabilized.log |
| 2026-08-11 | E1 s1 | 4c2-700 | 2.0 | 2.0 | e1_silent_backstop | 1 | 4 | 1?/4 | Backstop mechanics perfect (engage ~2.5s, refs question-matched, dedup works). Brooklyn "win" ambiguous (leading-question echo). Houseplants: double-inject (backstop+native 3.5s apart) mid-utterance -> SCREECH. | replay/auditions/logs/2026-08-11_E1_4c2-700.log |
| 2026-08-11 | E1 s2 | 4c2-700 | 2.0 | 2.0 | e1_silent_backstop | 0 | 2 | 0/2 | Both notes correct + ignored: deflection then dead air. | replay/auditions/logs/2026-08-11_E1_4c2-700_s2.log |
| 2026-08-11 | E1 s3 | 4c2-400 | 2.0 | 0.5 | e1_silent_backstop | 0 | 1 | 0/1 | HEALTHY checkpoint destabilized: deflect -> 11s silence -> screech after unsolicited injection. | replay/auditions/logs/2026-08-11_E1_4c2-400_destabilized.log |

**E1 VERDICT: TOKEN-GATED, and stronger — unsolicited injection is actively destabilizing (stalls/screech on both a stall-prone and a healthy checkpoint). Silent backstop not viable. E2 (forced single ret at boundary) is required and decisive.**
| 2026-08-11 | E2 s1 | 4c2-700 | 2.0 | 2.0 | e2_forced_ret (v1) | 0 | 4 armed, 0 consumed | 0/4 | FIELD DEFECT: TTL 2.0s expired before any word token (deep-rung words come 4+s late or never); consume-on-word deadlocks on stalls. Redesign: consume-on-pad. | replay/auditions/logs/2026-08-11_E2_4c2-700_s1.log |
