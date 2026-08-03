# Landscape: full-duplex speech, voice cloning, and where this project sits

**Draft for review (2026-08-03).** Background reading, not spec — context on the
state of conversational speech technology and what's distinctive about what this
repo measures. Product and model claims were verified against the sources listed
at the end on the date above; they will rot, and the hosted products in
particular change under stable brand names.

This repo is one half of a two-repo effort. The end state of the demo is a
MoshiRAG checkpoint **fine-tuned on a target speaker's voice and persona while
keeping the retrieval behavior intact** — training lives in the sibling
`moshirag-finetune` repo; this repo is the measurement framework that says
whether the fine-tune broke anything. The sections below are context for why
that combination is hard.

## Speech-to-speech models

### Live voice agents: full-duplex vs. cascade

Almost every "voice AI" product is a **cascade**: VAD → ASR → LLM → TTS, four
models in a row. It works, it's easy to assemble from parts, and it has three
structural problems.

- **Latency is additive, and endpointing is a guess.** Nothing can start until
  the VAD decides the user has stopped talking. That decision is a gamble on
  silence: tune it short and the agent talks over people mid-thought; tune it
  long and every reply feels sluggish. The tuning knob everyone reaches for
  trades one failure for the other — it doesn't remove either.
- **Paralinguistics die at the ASR boundary.** Hesitation, emphasis, sarcasm,
  laughter, and rising uncertainty all become flat text. The TTS at the other
  end then invents prosody from scratch, with no knowledge of how the user
  actually said it.
- **Barge-in is bolted on.** Interruption is handled outside the models — detect
  user speech, kill playback, discard the buffered audio. The LLM usually
  doesn't even know it was cut off, or where.

**Native speech-to-speech** models remove the ASR/TTS seam: audio tokens in,
audio tokens out, one model. That fixes the paralinguistic loss and collapses
the latency stack. Most production systems stop here — they are still turn-based
underneath, with a server-side VAD deciding whose turn it is and interruption
handling layered on top. This tier has gotten genuinely good: Qwen3.5-Omni added
turn-taking *intent* recognition that distinguishes a backchannel ("uh-huh")
from a real semantic interruption. But the model is still fundamentally taking
turns.

**Full-duplex in Moshi's sense** means there is no turn-taking state machine at
all. The model emits audio *and* listens on a fixed clock (12.5 Hz, 80 ms per
step), modeling both streams jointly and continuously. Silence is something the
model actively produces, not a state it waits in. Backchannels, overlapping
speech, and interruptions aren't special cases handled by surrounding code —
they're just what the model does, because that's what its training data did. The
MoshiRAG paper reports **0.18 s** turn-taking latency on Full-Duplex-Bench and
**0.0 s** time-to-first-audio-token.

The catch is that the clock never stops. In a cascade you can put a two-second
lookup between the ASR and the LLM and simply accept the pause. In a full-duplex
model there is no gap to hide it in. Anything you add either fits inside the
80 ms step budget or runs asynchronously beside the loop. That constraint is
what MoshiRAG's asynchronous retrieval design exists to satisfy (see the
README's "How MoshiRAG works"), and it's why retrieval here looks nothing like
retrieval in a text RAG stack.

That cost is measurable in systems that *don't* work this way.
Full-Duplex-Bench-v3 evaluates multi-step tool use by voice agents under
realistic disfluency; the fastest system tested was Gemini Live 3.1 at **4.25 s**
latency, with a cascaded Whisper→GPT-4o→TTS baseline slowest at **10.12 s**.
Those are chained multi-call tasks rather than a single lookup, so they are not
a like-for-like comparison against MoshiRAG's ≤2 s single-retrieval budget — but
they show the shape of the problem clearly enough: when knowledge access is
synchronous, the conversation waits for it, and the wait is measured in seconds.

### Voice cloning: zero-shot vs. SFT

Two ways to make a model speak in a particular person's voice.

**Zero-shot (speaker-conditioned).** Encode a short reference clip into a
speaker embedding and condition synthesis on it at inference. New voice in
seconds, no training, no per-speaker checkpoint. Chatterbox — Resemble AI's
MIT-licensed open TTS — clones from about 5 seconds of audio, and that is the
norm for modern open TTS. For timbre it works remarkably well. What it captures
less well is the *person*: pacing, filler habits, how long they hold a pause,
which words they stress.

**SFT (fine-tuning).** Train on tens of minutes to hours of target-speaker data
and update the weights. It costs a training run and a checkpoint per speaker,
and it can reach things conditioning struggles with: prosodic habits, idiolect,
phrasing, and — in a full-duplex model specifically — *turn-taking behavior*.
When this person interrupts, how long they let a pause run, whether they
backchannel while listening. Those are properties of the conversational policy
rather than of an acoustic decoder.

There is a structural difference worth being precise about. In a cascade, voice
is a module: swap the TTS box and keep everything else. In a full-duplex speech
LM there is no TTS stage to swap — voice isn't downstream of the language model,
it's the same weights doing the same generation. That doesn't make zero-shot
conditioning impossible (it's an active research direction, and prototypes
exist), but it does mean voice can't be treated as a detachable component the
way a cascade allows. Giving this model a specific voice means touching the
weights that also do the turn-taking, the inner monologue, and the `<ret>`
prediction that triggers retrieval.

The target speaker is a consenting participant, and the demo is identified as a
synthetic voice to everyone it's shown to. That is the same bar the commercial
providers set: OpenAI's custom-voice program requires a recorded consent
statement from the voice donor.

## Competitive comparison

Rows are pinned to specific named versions. The comparison is on *architecture*,
which is stable across releases, rather than on quality benchmarks, which are
not. This is a comparison against systems a general audience is likely to have
used — not a survey of the research literature.

| System | Duplex behavior | Voice control | Knowledge grounding | Weights |
|---|---|---|---|---|
| **OpenAI GPT-Realtime** | Native S2S, turn-based with server VAD + barge-in | Preset catalogue; custom voices exist but are gated to approved customers and require a recorded consent statement | Function calling / file search — synchronous, the turn waits | Hosted, closed |
| **Gemini Live 3.1** | Native S2S, turn-based with server VAD + barge-in | 30+ preset voices, 24+ languages; no cloning offered | Function calling + Grounding with Google Search — synchronous, the turn waits | Hosted, closed |
| **Qwen3.5-Omni** | Turn-based (Thinker–Talker), with turn-taking intent recognition for backchannel vs. real interruption | Small preset voice set (3 in Qwen3-Omni Instruct); no cloning | Function calling; retrieved text enters the context window and pays full prefill | Open (Apache-2.0) |
| **Chatterbox (Turbo / Multilingual v3)** | Not a dialogue system — TTS only; inherits whatever cascade wraps it | Zero-shot cloning from ~5 s, plus emotion control and paralinguistic tags | None — belongs to the LLM upstream of it | Open (MIT) |
| **This project** | **Full-duplex** — continuous parallel audio + text streams, no turn state machine | **Target-speaker voice + persona via SFT** of the speech LM itself | **`<ret>`-triggered async retrieval** that never enters the context window — the conversation doesn't wait for it | Open, self-hosted |

Read down the columns and the gaps are consistent. **None of the four systems
above puts a specific person's voice into a live conversational agent** — the
hosted models restrict custom voices to approved customers, Qwen ships a fixed
voice set, and Chatterbox clones beautifully but has no conversation attached.
**Where knowledge grounding exists, it is synchronous**: a tool call the turn
waits on, or context tokens the model must consume before continuing. And none
of them is full-duplex in the architectural sense — they take turns well, but
they take turns.

What this project combines is an open full-duplex model, a speaker-specific
fine-tune, and domain-specific asynchronous retrieval. The three constrain each
other: the fine-tune that installs the voice operates on the same weights that
decide when to retrieve, so "the voice got better" and "retrieval quietly
degraded" are things that can happen in the same training run. Whether they hold
together at once is an empirical question, and answering it is what this repo
measures — base and fine-tuned checkpoints scored on both sides.

**What we give up.** GPT-Realtime and Gemini Live are much stronger general
reasoners, far better multilingual, better at long-horizon tool use, and someone
else keeps them running. Our accuracy ceiling is Moshi's backbone, and the
retrieval integration is lossy by design — the MoshiRAG paper attributes part of
its own accuracy gap to exactly that. The trade is deliberate: raw model
capability for duplex fidelity, a specific voice, and private domain knowledge
on our own hardware.

## Sources

Verified 2026-08-03.

- **MoshiRAG** — arXiv 2604.12928; ICML 2026 poster; `github.com/kyutai-labs/moshi-rag`
- **Moshi** — arXiv 2410.00037
- **Full-Duplex-Bench** — arXiv 2503.04721; v3 (tool use under real-world disfluency) arXiv 2604.04847
- **Qwen3-Omni** — arXiv 2509.17765; `github.com/QwenLM/Qwen3-Omni` (Apache-2.0, 3 preset voices, function calling)
- **Qwen3.5-Omni** — arXiv 2604.15804 (turn-taking intent recognition)
- **Chatterbox** — `resemble.ai/learn/models/chatterbox` (MIT; ~5 s zero-shot; Turbo 350M Dec 2025; Multilingual v3 Jun 2026)
- **Gemini Live API** — `ai.google.dev/gemini-api/docs/live-api/capabilities`; Vertex AI Live API docs
- **OpenAI voices / custom voices** — `platform.openai.com/docs/guides/text-to-speech`; OpenAI, "Navigating the challenges and opportunities of synthetic voices"
