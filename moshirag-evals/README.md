# MoshiRAG Eval Pipeline

Evaluation harness for [kyutai-labs/moshi-rag](https://github.com/kyutai-labs/moshi-rag) — measures answer quality and latency of the retrieval-augmented voice model against benchmark question sets (LlamaQ, WebQ, and others), plus a live demo stack for manual testing.

Full requirements/spec: [`specs/moshirag-evals-requirements.md`](specs/moshirag-evals-requirements.md) — the authoritative source of truth for this project. Setup and run commands are in [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## How MoshiRAG works

MoshiRAG is built on [Moshi](https://arxiv.org/abs/2410.00037), a full-duplex speech language model: it runs two parallel token streams — an inner-monologue text channel and an audio output channel — on a fixed step clock (12.5 Hz, 80 ms/step) with no separate turn-taking state machine. There's no point in the loop where the model "waits its turn"; it's always producing audio, and text and speech are just two views onto the same generation process.

Retrieval sits inside that loop rather than in front of it:

- **Only if the model predicts it.** Nothing outside the model decides whether to retrieve. Retrieval triggers when generation itself produces the `<ret>` token — a learned signal that the model doesn't have enough grounding to keep answering.
- **In the background, not in the way.** Once `<ret>` fires, the retrieval call runs as a background task while the front-end keeps generating and keeps listening — full-duplex the whole time. Nothing blocks on the network round-trip.
- **Compressed conditioning injection, not context-window concatenation.** Standard RAG — including prior RAG-augmented speech systems, which the MoshiRAG paper notes were built for non-full-duplex settings — works by inserting retrieved text into the model's context as tokens: synchronous by construction, since the model has to consume them before continuing. MoshiRAG instead compresses the retrieved reference ~4x via a pretrained ARC-Encoder and sums the resulting embeddings directly into the transformer's streaming input over `l` steps, starting `d` seconds after `<ret>` — the retrieved text never becomes context-window tokens at all. This is genuinely lossy (the paper attributes part of its own accuracy gap to "information loss introduced during RAG integration"), but it's the tradeoff that makes retrieval compatible with an 80 ms/step real-time budget instead of stalling it: the paper reports a **0.0 s** time-to-first-audio-token even on turns that retrieve.

This repo's eval suite covers both sides of that mechanism. Knowledge evals measure whether retrieval actually helps: `open_audio_bench` (TriviaQA/WebQ/LlamaQ), `halu_eval_audio` (hallucination), and `gsm8k` (out-of-domain math reasoning) all come from the MoshiRAG paper's own benchmark suite — the paper reports retrieval mattering substantially even for GSM8K (vanilla Moshi scores 2.1% with no retrieval vs. MoshiRAG's 33.9%+). This repo's `gsm8k` eval currently scores exact-match on the final number rather than the paper's ref./resp. split, and doesn't force retrieval on. Latency evals measure whether the async-retrieval design holds up in practice: `ttfat` (does retrieval ever stall first-audio-token?) and `retrieval_breakdown` (where does retrieval time actually go — ASR wait, API call, context injection?) are both implemented. `e2ekd`/keyword delay (how long until a retrieved fact is actually spoken) and `duplex.full_duplex_bench` (turn-taking/interruption latency) are not implemented yet — both need `nemo_toolkit["asr"]` for word-level timestamps, a large framework whose own pinned `torch`/`torchaudio` versions risk silently conflicting with the ones this project already pins for `moshi` compatibility.

## The demo

A browser-based full-duplex voice interface for ad-hoc testing, edge-case probing, and stakeholder demos — running kyutai-labs/moshi-rag's own, unmodified server stack, not a reimplementation.

It's three separate processes, not one model call:

- **`moshi.server`** — the main front-end model and step loop (VAD, turn-taking, audio streaming)
- **`moshi.server_conditioner`** — the ARC-Encoder, always a separate process, never in-process; this is what turns a retrieved reference into the conditioning embeddings described above
- **Local STT** — synchronous, single-threaded, feeding transcription into the retrieval context

`scripts/instrumented_server.py` wraps `moshi.server` with a thin patch layer: it redirects retrieval so the demo answers `<ret>` triggers through this repo's own `RetrievalBackend` — the same backend, config, and prompt template the eval path uses — instead of moshi's bundled canned templates. That's the one behavioral patch. VAD, the step loop, and turn-taking are left as moshi's own unmodified logic throughout; everything else the wrapper adds (session logs, live latency push to the client) is additive logging on top.

**What we added over upstream:** the original moshi-rag has an unbounded `Channel.input_queue` — any stall (a model load, a slow turn) gets drained afterward faster than real time, flooding the browser, which discards the surplus as a metallic artifact. We bounded it (cap ~2 frames, drop-oldest), which removed 81% of discarded audio in VM testing.

**Retrieval backend: `gemini-3.5-flash-lite`**, the shipped default (`configs/retrieval_backends.yaml`). It was promoted from `gemini-3.5-flash` after a live VM run showed it cuts retrieval latency meaningfully with no quality/grounding regression — comfortably inside the paper's own ≤2 s end-to-end retrieval-delay budget, where the plain `flash` model (a "thinking" model with a slower default reasoning mode) was not.

## Why not just use a hosted voice API?

GPT-Realtime and Gemini Live are stronger general reasoners and someone else keeps them running, but neither is full-duplex in the architectural sense, neither will clone a specific person's voice on open terms, and both retrieve synchronously — the turn waits on the tool call. This project's bet is that an open full-duplex model, a speaker fine-tune, and asynchronous domain retrieval can hold together at once. See [`docs/landscape.md`](docs/landscape.md) for the full comparison — cascade vs. full-duplex, zero-shot vs. SFT voice cloning, and where the alternatives land.

## Status

Actively developed. Most recent completed arc: a turn-onset delay and a metallic audio artifact were both root-caused to self-inflicted regressions and an upstream defect (see above), fixed and VM-validated across six live sessions.

`.env`, model checkpoints, eval results, and demo/session recordings are gitignored — this repo is code and config only.

## Layout

| Path | What it is |
|---|---|
| `core/` | Model adapter (`MoshiRAGAdapter`), GPU device resolution, checkpoint download |
| `evals/` | Eval runner, scoring, latency/retrieval-breakdown instrumentation |
| `configs/` | Eval configs (checkpoint, retrieval backend, dataset subset) |
| `demo/` | Instrumented launch script + forked web client for live manual testing |
| `scripts/` | VM setup, demo launch, diagnostic tooling |
| `tests/` | Unit tests (no GPU required) |
| `specs/` | Authoritative requirements doc |
| `docs/` | Standalone investigation reports and work plans |

## Quick start

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for VM setup, running an eval, running the demo, and tests.
