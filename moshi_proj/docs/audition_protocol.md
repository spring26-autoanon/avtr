# Audition protocol — same script every time, 3 sessions per checkpoint

Triggering is stochastic; one session proves nothing (the false "checkpoint 500
regression" came from single-session judging). Ask with THESE phrasings, same mic,
quiet room — trigger rate tracks user-speech intelligibility (spec F2).

| id | ask exactly | kind | passed means |
|---|---|---|---|
| q01 | "Hi, how are you?" | greeting | responds promptly, hands it back |
| q02 | "What's your name?" | persona | says Danielle; no ⟨ret⟩ |
| q03 | "What do you do for work?" | persona | Director of Digital Platforms / legal |
| q04 | "Can you tell me about the World Cup?" | factual | fires + answer matches reference |
| q05 | "Who won the first World Cup?" | factual | fires + Uruguay 1930 |
| q06 | "What's the capital of Australia?" | factual | fires + Canberra |
| q07 | "What's the weather in Dublin right now?" | decline | declines gracefully, keeps talking |
| q08 | "Do you like playing soccer?" | personal | answers as herself; no ⟨ret⟩ |
| q09 | "Did you ever do martial arts?" | persona | tae kwon do / wrestling content |
| q10 | "Do you have any plants?" | persona | 60+ / monstera content |
| q11 | "Do you have tattoos?" | persona | sleeve / travel-tattoo content |
| q12 | "Where are you from?" | persona | Brooklyn → Tampa |
| q13 | re-ask q05, interrupt mid-answer ("wait, which year?") | barge-in | yields, then picks back up → session `interrupt_yields` |

Per session: copy `replay/auditions/session_template.json` →
`replay/auditions/<checkpoint>_s<N>.json`, fill `fired`/`passed`/`notes` per question,
session `voice` 1–5 (does it sound like HER — filler, warmth, timing), `interrupt_yields`.
Keep the serve.log (`~/serve.log`) per session; the scorer reads it.

Score: `python3 scripts/audition_checkpoint.py --log serve.log \
  --sessions replay/auditions/<ckpt>_s1.json <ckpt>_s2.json <ckpt>_s3.json \
  --out-dir replay/auditions/<ckpt>/`

Verdict weighting (decision of record): stalls + voice/persona FIRST, trigger rate second.
Never trade criteria 3/7 (lead, sounds-like-her) for 1/2 (triggering) — scaling 1.5 made
that trade and it was wrong.
