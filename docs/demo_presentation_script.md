# Demo presentation script — "Giving Moshi a Voice: Danielle meets moshi-rag"

Spoken script, ~7 minutes before demos. Sections map to the three slides
(1: Technical Pipeline, 2: Full-Duplex System Workflow, 3: Stages of Testing).
Deliver conversationally; the bold lines are the beats to hit even if you improvise.

---

## Opening (no slide, ~30 s)

Most voice assistants are a pipeline of three separate systems: speech-to-text, a
text language model, and text-to-speech. That's why they feel like walkie-talkies —
you speak, you wait, they speak. **Moshi is different: it's a single model that
listens and speaks at the same time**, full duplex, like a real phone call. Our
project asked two questions on top of that: **can we make Moshi speak as a specific
real person — Danielle — and can that persona coexist with reliable, live
retrieval-augmented generation?** Today you'll hear both.

---

## Slide 1 — The technical pipeline (~2 min)

Moshi is an RQ-Transformer: every 80 milliseconds it emits one text token plus
eight audio codec tokens, so text and sound are generated together, in her voice,
in real time. It runs at 12.5 frames per second and hears the user as raw audio —
there's no transcription step between your mouth and the model.

We didn't retrain this 7-billion-parameter model. **We trained a LoRA adapter — a
small, low-rank delta on the attention layers** — on recordings of one consented
speaker, Danielle. The data pipeline mattered more than the math:

- **Stereo structure is the curriculum.** Danielle's isolated voice on the left
  channel — the channel Moshi learns to *be* — and her conversation partner on the
  right, the channel Moshi learns to *listen to*. Our first attempt used monologues
  with a silent partner channel, and the model learned exactly that: to monologue
  over you. Real two-speaker dialogue fixed turn-taking.
- **Word-level transcripts** align every spoken word to its moment, so the text
  stream and audio stream stay locked.
- **Her disfluencies are data.** Whisper politely deletes "um" and "like" from
  transcripts, and early fine-tunes lost her fillers. We kept the register in the
  audio and trained long-form, so the ums live in the adapter.

For retrieval, we built on **moshi-rag**: the model is trained to emit a special
retrieval token — think of it as the model *raising its hand mid-conversation*
when it wants facts. That token triggers a backend LLM to write a short reference
document, which an ARC encoder — a compressed language-model encoder — turns into
embedding vectors that are summed directly into Moshi's hidden state. **The model
never reads text; knowledge is injected as vectors, mid-sentence, in about one
second.**

---

## Slide 2 — Full-duplex system workflow (~1.5 min)

Live, the system is four services in concert. The browser streams your microphone
to the GPU server over a websocket. Moshi ingests your audio frame by frame while
generating her own — both directions at once; a voice-activity detector arbitrates
turns, and her replies buffer while you're still talking so she can come in the
moment you stop.

When the model raises its hand — emits the retrieval token — the server sends the
conversation so far to a fast backend LLM (Gemini), which writes a reference in
under half a second. The ARC encoder embeds it in a quarter second, the vectors sum
into her hidden stream, and her next words are grounded. **Round trip: under two
seconds, while the conversation keeps flowing.**

The same machinery carries her persona. Personal questions — who are you, where
are you from, what do you do — were trained as *marked retrieval turns*, so she
raises her hand for her own biography, and the backend answers from a persona
reference we control. You'll see her correct herself live when retrieval lands —
that flicker from a guess to the right answer is the system working in real time.

---

## Slide 3 — Stages of testing: what broke and what we learned (~2 min)

The journey was five fine-tunes and about thirty checkpoint auditions. Three
findings shaped everything:

**First: the 99-second ceiling.** Early adapters died at exactly 1 minute 39 into
every session. That's not a bug in the server — it's 100 seconds, the training
clip length. The model had learned that conversations *end* at the horizon it was
trained on. Training on 300-second windows — matching the base model's native
horizon — fixed it: same adapter recipe went from dying at 1:39 to conversing past
five minutes.

**Second: catastrophic forgetting of retrieval.** Fine-tuning on casual speech
quietly taught the model to *never* raise its hand — every casual clip was a
lesson that the retrieval token is wrong. We fixed it with a loss mask: on
unlabelled data, the retrieval token is simply removed from the loss, so voice
training can't vote against retrieval. Paired with a gentler learning rate, the
trigger survived.

**Third: dose is everything, and it's tunable at serve time.** More training steps
buy more voice and better retrieval timing — until the model stops taking turns at
all. Our best checkpoint was actually *overtrained* and unusable at full strength;
scaling the adapter down at inference — a single dial, no retraining — restored
fluent speech while keeping the mature retrieval behavior. The final model is that
checkpoint, served at reduced scale: **voice, turn-taking, and retrieval from one
adapter.**

---

## Demo setup (~30 s)

Two models, live from the GPUs. **Track A** is the pure voice clone on base Moshi —
listen for her tone and her fillers; it has no retrieval by design. **Track B** is
moshi-rag wearing her voice: watch the reference panel when she raises her hand —
you'll see the exact document injected into her hidden state as she speaks. When
she's asked where she grew up, the answer comes from retrieval, in Danielle's
voice — which is the whole project in one sentence.

[→ run demos; recovery moves: fresh question un-freezes her; reconnect resets a
flat session]

---

## Numbers to have ready for Q&A

- Base: Moshi 7B (moshika); RAG stack: kyutai/moshika-rag-pytorch-bf16
- Adapter: LoRA rank 64 on attention layers, ~0.5% of model weights
- Data: ~4.5 h total — casual dialogue, factual/retrieval sessions with 248 marked
  retrieval turns, two persona interview sessions
- Training: single A100-80GB, 800 steps at 300 s windows ≈ 9.5 h; loss mask +
  Eq. 3 delay + reference dropout on
- Serving: retrieval round trip < 2 s (LLM ~0.5 s + ARC encode ~0.2 s + injection);
  frame rate 12.5 Hz; audio codec Mimi at 24 kHz
- Fire timing matures with dose: late→mid→early relative to the question as
  checkpoints deepen (Eq. 3 delay visible on the ladder)
