// MoshiRAG browser demo client — continuous full-duplex audio over WebSocket.
// Vanilla JS, no framework dependencies.

const SAMPLE_RATE = 24000;
// Valid ScriptProcessorNode buffer size (must be a power of two, 256-16384).
// The server treats chunks as opaque bytes, so the exact size doesn't need
// to match any particular model frame rate.
const CHUNK_SAMPLES = 2048;

const CONFIG_WITH_RETRIEVAL = "configs/baseline_with_retrieval.yaml";
const CONFIG_NO_RETRIEVAL = "configs/baseline_no_retrieval.yaml";

let audioContext = null;
let micStream = null;
let micSource = null;
let micProcessor = null;
let ws = null;
let nextPlayTime = 0;
let retrievalEnabled = true;
// The client has no way to know what --config the server was actually
// started with. Only send an explicit ?config= override once the user has
// clicked the toggle themselves — otherwise the first connection silently
// overrides whatever the operator chose at startup (e.g. running the server
// with baseline_no_retrieval.yaml to test without retrieval, only to have
// this client reconnect with retrieval on anyway). The "Retrieval: ON/OFF"
// label is still just this client's assumption until toggled once; there's
// no protocol message announcing the server's actual active config.
let userToggledConfig = false;

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const retrievalToggle = document.getElementById("retrieval-toggle");
const statusEl = document.getElementById("status");

function updateRetrievalLabel() {
  retrievalToggle.textContent = retrievalEnabled ? "Retrieval: ON" : "Retrieval: OFF";
}

function currentConfigPath() {
  return retrievalEnabled ? CONFIG_WITH_RETRIEVAL : CONFIG_NO_RETRIEVAL;
}

function floatTo16BitPCM(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const configParam = userToggledConfig
    ? `?config=${encodeURIComponent(currentConfigPath())}`
    : ""; // let the server use whatever --config it was actually started with
  const url = `${proto}//${location.host}/ws${configParam}`;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    statusEl.textContent = "connected";
  };

  ws.onmessage = (event) => {
    playChunk(event.data);
  };

  ws.onclose = (event) => {
    statusEl.textContent = event.code === 1013 ? "busy - try again" : "disconnected";
  };

  ws.onerror = () => {
    statusEl.textContent = "error";
  };
}

function playChunk(arrayBuffer) {
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 0x8000;
  }

  const buffer = audioContext.createBuffer(1, float32.length, SAMPLE_RATE);
  buffer.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);

  const startAt = Math.max(audioContext.currentTime, nextPlayTime);
  source.start(startAt);
  nextPlayTime = startAt + buffer.duration;
}

async function start() {
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  nextPlayTime = 0;

  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micSource = audioContext.createMediaStreamSource(micStream);

  // ScriptProcessorNode is deprecated in favor of AudioWorkletNode, but
  // remains universally supported and needs no separate worklet module file.
  micProcessor = audioContext.createScriptProcessor(CHUNK_SAMPLES, 1, 1);
  micProcessor.onaudioprocess = (event) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const pcm16 = floatTo16BitPCM(input);
    ws.send(pcm16.buffer);
  };

  micSource.connect(micProcessor);
  // ScriptProcessorNode only fires onaudioprocess when connected to a
  // destination; we never write to outputBuffer, so nothing audible loops back.
  micProcessor.connect(audioContext.destination);

  connect();

  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "connecting...";
}

function stop() {
  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor.onaudioprocess = null;
    micProcessor = null;
  }
  if (micSource) {
    micSource.disconnect();
    micSource = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "stopped";
}

function toggleRetrieval() {
  retrievalEnabled = !retrievalEnabled;
  userToggledConfig = true;
  updateRetrievalLabel();

  // Because Moshi is full-duplex and always listening, there's no client-side
  // turn state to preserve across the switch — a clean reconnect is enough.
  if (ws) {
    ws.close();
    connect();
  }
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
retrievalToggle.addEventListener("click", toggleRetrieval);

updateRetrievalLabel();
