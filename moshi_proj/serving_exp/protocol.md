# Serving experiments protocol (spec: docs/superpowers/specs/2026-08-11-backstop-retrieval-design.md)

## Standard probe script (say verbatim; question ends the turn; ~5 s gaps)
1. "Hi, how are you?"            (health; never backstopped — not retrieval-worthy)
2. "Are you from Florida originally?"
3. "Where did you grow up?"
4. "What should I call you?"
5. "What do you do for work?"
6. "How many houseplants do you have?"
7. "How many tattoos do you have now?"
8. "What music do you listen to the most?"
9. "Can you tell me about the first World Cup?"
10. "What's the live price of Bitcoin right now?"   (decline probe — E4 focus)
Then 1 casual minute (voice/stalls).

## Per-question verdicts to record
F  = native fire; B = backstop engaged; G = answer grounded (tracks the note);
X  = wrong/confabulated; S = stall/no answer; D = correct decline.

## Session ritual
curl -s localhost:8001/health   -> encoder UP required
fresh private browser window per session; archive serve.log after every session:
cp ~/serve.log ~/moshi-finetune/replay/auditions/logs/<date>_<experiment>_<ckpt>.log
