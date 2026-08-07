# Stage 4 — preserve the base, add Danielle

Date: 2026-08-07
Status: draft for review
Supersedes the "more retrieval data" instinct in `docs/moshi-rag-experiments.md` trial 5b.

## The governing principle

Stage 3 established something that reframes everything after it. A base-model control
(`moshika-rag`, no adapter) was run on the identical stack and it:

- fired `⟨ret⟩` on **seven consecutive turns** with no misses,
- never stalled,
- declined an unanswerable question gracefully and unprompted,
- took turns cleanly.

Our LoRA made all four **worse**. So the remaining work is not *teaching* retrieval,
turn-taking, or fluency. Those are already in the base. The work is:

> **Add Danielle's voice, manner and identity while destroying as little of the base as
> possible.**

Every decision below follows from that. Where we currently differ from the paper, the
question is not "how do we do it better" but "how do we stop damaging what already works".

## 1. What the paper does

From `moshi-rag-paper.pdf` (Chien et al.), the parts that bear on us.

### Scale — why we cannot teach the trigger

| | Kyutai | Stage 3 (ours) |
|---|---|---|
| Retrieval data | 1,901,376 conversations, **47,770 hours** | 94 clips, **1.26 hours** |
| Updates × batch | 100,000 × 32 = **3.2 M samples** | 800 × 8 = **6,400** |
| Trainable params | **all except the reference text encoder** | LoRA rank 64, attn + MLP |
| LR | 2e-6 | 4e-6 |

~38,000× less data, ~500× fewer samples. The trigger is not installable at our scale. It is
only preservable.

### Response structure: lead / body / tail

> "An RAG-enabled response consists of a **lead** portion that does not depend on external
> knowledge, a **body** portion that contains reference-grounded content, and an optional
> **tail** portion that concludes the response."

Pre-RAG content — everything after `⟨ret⟩` and before the document lands — "typically
includes a coarse answer to the user's query as well as conversational filler phrases
(e.g. *'Let me check that for you…'*)".

**The filler is load-bearing.** It buys the time retrieval needs. It is not decoration.

### `⟨ret⟩` placement

> "we replace the text token **before the first text token in the lead portion** of an
> RAG-enabled Moshi turn with the `⟨ret⟩` token."

Before the *lead* — i.e. before the filler, not before the facts.

### Retrieval delay is sampled relative to the lead (Equation 3)

```
d' = U(0, d_lead)              if d_lead < 2s  or  p < 0.2
     U(1.0, d_lead − 1.0)      otherwise          , p ~ U(0,1)
```

The document is injected `d'` after the trigger, and `d'` is **bounded by the lead
duration** — so it almost always arrives *while she is still saying filler*, before the body
begins. Training distribution is deliberately wider than real inference delays "to expose
the model to edge cases."

### Reference dropout

20% of documents are dropped; the embedding at `i⟨ret⟩ + d_fr` becomes `h_i + h_dropout`
with `h_dropout` learnable. The model is explicitly taught to cope when retrieval returns
nothing.

### Conversation-style variants

Three prompt variants per topic, and two matter to us:
- **v2** — "the user challenges Moshi more frequently and therefore more back-and-forth
  exchanges and argumentation are included" → turn-taking pressure.
- **v3** — "the user occasionally introduces irrelevant remarks or engages in small talk"
  → mode switching.

Greetings are appended and then dropped with probability 0.3.

### Where we diverge, and what it cost

| Paper | Stage 3 | Observed consequence |
|---|---|---|
| Document injected at `i⟨ret⟩ + d'`, inside the lead | injected **at** `i⟨ret⟩`, zero delay | at 03:44:55 she committed to "Argentina" **1.8 s before** the correct document arrived, then defended the wrong answer |
| `⟨ret⟩` before the lead | before her first word — *approximately right* | keep, but verify against real clips |
| 20% reference dropout | none | confabulates confidently when retrieval misses |
| ~half of all turns are retrieval turns | 135 markers vs 135 min of **marker-free** dialogue at 0.5 sampling | trigger suppressed to ~50% |

## 2. Goal A — preserve turn-taking and stop stalling

**Diagnosis.** The stall was ours: 76 of 78 clips ended within 0.5 s of her final word,
median 0.00 s, teaching "after you speak, everything stops." The 3 s tail fix took the worst
case from 25 s to 2.1 s, but it remains intermittent (2–18 s).

**Why intermittent.** The natural-dialogue half is 100 s windows chunked from 38–59 minute
files. Those windows end wherever they land — sometimes just after one of her turns. Random
rather than systematic, but it is 50% of training.

**Plan:**

1. **Keep the 3 s tail** on every retrieval clip. Verified: 0/78 now end on her silence.
2. **Apply the same rule to dialogue windows.** If a chunk boundary falls within 1 s of the
   end of one of her turns, shift the window so it ends mid-turn instead. Cheap change in
   `dataset.py` chunking, or handled by pre-cutting the dialogue into tail-padded clips the
   same way the retrieval clips are.
3. **Reduce total intervention** (see §5) — the base does not stall, so less damage means
   less stalling.
4. **Turn-taking is preserved, not taught.** The retrieval clips already contain two-person
   alternation: 194 of her smalltalk turns interleaved with Joshua's questions. Dropping the
   dialogue half removes marker-free ballast, not turn-taking signal.

**Known gap:** her recordings have **0.1% speaker overlap** — very polite alternation. The
paper's v2 variant deliberately trains interruption and argumentation. We have none. This is
the one turn-taking property we may genuinely erode, and the mitigation is in §4's recording
spec.

## 3. Goal B — answer personal questions as Danielle

Two layers, and only one is a retrieval problem.

**Facts** (Brooklyn, Tampa, Director of Digital Platforms, tae kwon do, sixty house plants)
belong in retrieval. The persona block already exists in
`serving/reference_prompt_template_simplified.txt` and is installed on the box. It works —
when `⟨ret⟩` fires on a personal question, which it did roughly one time in three.

**Identity** is in the weights. She says *"I'm just a chatbot,"* *"As an AI, I don't have
legs,"* *"I don't have personal details like names or photos."* That is base `moshika-rag`'s
assistant persona, and her 85 minutes contain **zero** counter-evidence: Joshua never asks
who she is, and she never says who she is. Retrieval conditions what she says about a
*topic*; it does not change who she thinks she is.

**Plan:**

1. **Record self-identity Q&A** (§4). This is the only fix for the identity layer — there is
   no data to reweight, only an absence to fill.
2. Those turns label as `grounded`, so they also add personal-question retrieval examples,
   which is the weakest trigger case.
3. Keep the persona template as the source of *current* facts (job titles change; weights
   don't).

**Success test:** "What do you do for work?" → `[Reference] Generated reference` contains
Director of Digital Platforms → she says it in her own words, without "as an AI".

## 4. Goal C — keep her voice sounding like *her*

This is the constraint the other goals must not trade against, and it is the reason to
reject the scaling lever.

**What "her voice" is made of, concretely, from her transcripts:**
- **Openers**: "Yeah, …" begins a large share of her turns. Also "Actually", "I think".
- **Reactive acknowledgement** before content: "Yeah, that's right." / "It certainly can be."
- **Follow-up questions back to the user**: "Is there one you've heard of recently that made
  you curious?" — she keeps the conversation alive rather than dead-ending.
- **Prosody and inflection** — carried by the audio codebooks, not the text stream.

**What we learned at scaling 1.5:** triggering improved, and *"it lost its filler words and
voice inflection."* Scaling shrinks the whole adapter delta uniformly — filler and prosody
are among the first casualties. **Do not use serve-time scaling as the trigger lever.** It
costs the thing we are here to build, and it breaks grounding as a side effect by removing
the delay cover.

**The delay fix protects filler rather than competing with it.** Today the document is
present at the trigger frame, so the model has no reason to say anything before the content —
we are actively training her to skip the lead. Inject at `i⟨ret⟩ + d'` and the model must
generate *something* across the gap; what her data puts there is her filler. Training the
delay trains the filler.

**Plan:**

1. **Implement Equation 3** — sample `d'` bounded by the lead duration, per example.
2. **Label lead / body / tail** per retrieval turn (new work, §6). We need `d_lead` to sample
   the delay, and we need to know where the grounded content starts.
3. **Measure her natural keyword delay** from existing transcripts: seconds between her turn
   start and her first content word. That is the target distribution, computable today with
   no GPU.
4. **Serve at scaling 2.0.** Fix triggering through data, not through diluting her.

## 5. Minimal intervention — hyperparameters

Turn-taking, triggering and fluency all degrade with training. Voice is the only thing that
improves. That asymmetry should be reflected in the settings, and none of these are settled:

| knob | Stage 3 | proposal | why |
|---|---|---|---|
| LoRA rank | 64 | **32** | one halving from the only value with evidence behind it |
| steps | 800 | **~400** | run 1's history: voice imprints by ~300 steps; beyond that damage accrues while voice plateaus |
| sampling | 0.5 retrieval / 0.5 dialogue | depends on §6a | if dialogue mining yields markers, keep it; otherwise 0.8/0.2 |
| `RAG_TOKEN_WEIGHT` | 25 | **25**, revisit after the delay fix | 15 never fired; 25 fires ~50% |
| serve scaling | 2.0 | **2.0** | 1.5 costs filler and prosody |

**Rank 32, not 16.** Rank 64 is the *only* setting with evidence behind it — every successful
voice clone in this project used it. A two-step drop to 16 risks conflating "lost her voice"
with "rank too low" when several other things are changing simultaneously. Halve once, see
what it costs.

### Sequencing — do not change everything at once

Stage 3 taught this the hard way: a checkpoint was called a regression from one session and
the call was wrong. With five knobs in flight, a worse run tells us nothing.

| run | change | isolates |
|---|---|---|
| **4a** | delay fix + reference dropout only. Rank 64, 800 steps, 0.5/0.5 unchanged. | Does the lead/delay structure fix grounding and restore filler? This is the change with a confirmed failure to point at — "Argentina" 1.8 s early. |
| **4b** | + rank 32, ~400 steps | Does less intervention preserve triggering and turn-taking, at what voice cost? |
| **4c** | + data ratio (mined dialogue, or 0.8/0.2, or retrieval-only) | Does the marker ratio drive triggering? |

Each run is ~3 hours plus audition. The harness (§6) is what makes this affordable.

## 6a. Mine the natural dialogue for markers — measure before committing

The 135 minutes of natural dialogue is the largest block of her voice we have, and the
reason to drop it is only that it is **marker-free** and therefore suppresses the trigger.
If markers can be added where she genuinely states facts, we keep the voice data *and* fix
the ratio — strictly better than discarding it.

**Why the density may be higher than "friends chatting" suggests.** Under our persona design,
**her talking about herself is retrievable content**. Casual conversation with Clay and
Joshua very likely contains her discussing travel, work, plants, training — which is exactly
the persona material §3 says we lack. Mining may surface persona data that is already
recorded.

**Do the cheap measurement first.** The channel-0 transcripts already exist on the box and
are small. Joshua's side is *not* needed to answer "how many of her turns state facts."

1. `rsync` the ch0 `.json` alignments for `danielle_clays`, `..._train_a`, `..._train_b`.
2. Merge to utterances, hand her turns to Gemini, ask which are knowledge-dependent or
   self-descriptive, and count.
3. **Decision rule:** ≥150 markable turns → worth the channel-1 Whisper pass and full
   labelling. Under ~50 → drop the idea; the dialogue half gets reweighted or removed
   instead.

Only after the count justifies it: channel-1 Whisper pass (GPU, user-run), then the same
segment → label → reference → lead/body pipeline the retrieval clips use.

**Caveat.** Marking a turn that is *not* genuinely knowledge-dependent is worse than leaving
it unmarked — it trains spurious triggering, the failure the base already exhibits. The
faithfulness screen (recalibrated for subject match, not completeness) applies here too, and
the bar for marking should be higher than for the recorded retrieval clips.

## 6. New work required

**Lead/body/tail labelling.** For each of her retrieval turns, mark where reference-grounded
content begins. Extends `segment_retrieval_audio.py`: Gemini already sees her turn text and
the reference it generated, so ask it for the index of the first grounded word. Word
timings then give `d_lead`.

**Delay sampling in `reference_injection.py`.** `start = hits[i] + d'` with `d'` from
Equation 3, clamped at the next `⟨ret⟩` and at sequence end. A delay exceeding the remaining
answer means no injection — itself a valid "retrieval arrived too late" case.

**Reference dropout.** Skip injection for ~20% of documents. The paper's learnable
`h_dropout` is not available inside a LoRA; omission is the cheap approximation.

**Dialogue-window tail rule** (§2.2), if the dialogue half is retained.

**Audition harness.** Stage 3 lost hours to session variance — a checkpoint was called a
regression off one session and the call was wrong. A script that replays a fixed question
list, counts `⟨ret⟩` fires, measures stall durations and lead durations from the log, and
checks whether her answer contains the reference's key facts turns checkpoint comparison
into minutes and numbers. Pure local Python; build it before the next training run.

## 7. Recording spec — tomorrow night

Target **35–45 minutes**, one session. All three parts want the same setup, and setup is the
expensive bit.

### Part 1 — self-identity, ~25 min

Eight topics × 6–8 phrasings ≈ 56 exchanges. Repetition across varied phrasings matters far
more than duration: the goal is to override a strong prior, and priors yield to consistency,
not to one long exposure.

| topic | content |
|---|---|
| Name / who she is | "What's your name?" · "Who am I talking to?" · "Introduce yourself" |
| Where she's from | Brooklyn; lives in Tampa |
| Work | Director of Digital Platforms, legal industry, data + AI strategy |
| Travel | Ireland, Cliffs of Moher, Giant's Causeway, the Iceland puffin trip |
| Fighting | tae kwon do, wrestling, krav maga, bare-knuckle; started in middle school |
| Music | metal/hardcore/punk, Hellfest, Minor Threat, System of a Down |
| Tattoos | 20+, full sleeve, the travel sleeve |
| Plants | sixty-plus, the Ikea monstera, the philodendron goeldii |

**Conversational, not a CV.** "I'm originally from Brooklyn, but I've been in Tampa a few
years now" trains better than "My hometown is Brooklyn, New York" — and it is how she
actually talks.

### Part 2 — declines, ~10 min

15–20 exchanges. Joshua asks something genuinely unanswerable — hyper-specific figures, live
data (weather, scores), things about *him*. She says she doesn't know **and keeps the
conversation going**. She must not guess, must not half-hedge into a wrong answer, and must
not stop dead.

These still carry `⟨ret⟩` in the pipeline: the lesson is "look it up, and if what comes back
doesn't answer it, say so."

### Part 3 — interruption and overlap, ~5–10 min

**New, and the one thing the existing data lacks.** Her recordings are 0.1% overlap — she and
Joshua never talk over each other. The paper trains a variant specifically for
back-and-forth and challenge. To preserve interruptibility:

- Joshua interrupts mid-answer; she yields.
- Joshua challenges a fact; she responds without collapsing.
- Backchannels — "mm-hm", "right" — over each other's speech.

Overlap is fine here: the channels are physically isolated (measured −51 dB), so overlapping
speech separates cleanly.

### Recording requirements (unchanged, they worked)

Isolated per-speaker tracks, her on channel 0, 24 kHz, same mic and room as before —
consistency of recording conditions reinforces the clone.

## 8. Verification

Judge against the base-model control, not against Stage 3:

1. **Triggering** — `⟨ret⟩` fires on ≥6 of 8 scripted factual questions across 3 sessions.
   Base gets 8/8; Stage 3 gets ~4/8.
2. **Grounding** — when it fires, her answer contains the reference's key fact. The "Argentina
   won" failure is the regression test.
3. **Lead present** — she says filler before the fact. Measurable: seconds from turn start to
   first content word, compared against her own transcripts.
4. **No stall** — longest empty-buffer gap under 3 s across a 3-minute conversation.
5. **Turn-taking** — interrupt her mid-answer; she yields.
6. **Identity** — "What do you do for work?" gets Director of Digital Platforms, not "as an AI".
7. **Voice** — she sounds like herself, with her inflection and her filler, in both modes.

Criteria 3 and 7 are the ones that must not be traded away for 1 and 2.

## 9. Open questions

- **Does the lead/body labelling survive Whisper noise?** Her transcripts garble proper nouns
  ("dais" for diyas), and the body often *starts* at a proper noun.
- **Is 1.26–1.8 hr enough for voice at rank 16?** Lower rank means less capacity for timbre.
  Ablation, not a prediction.
- **Does reference dropout without a learnable `h_dropout` do anything useful,** or does it
  just teach the model to ignore conditioning?
- **Should the dialogue half be dropped entirely or reduced to 0.2?** Dropping removes the
  most marker-free windows but also Clay, the only second conversation partner.
