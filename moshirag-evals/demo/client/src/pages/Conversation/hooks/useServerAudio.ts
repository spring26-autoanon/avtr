import { useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { useMediaContext } from "../MediaContext";
import { DecoderWorker } from "../../../decoder/decoderWorker";

export type AudioStats = {
  playedAudioDuration: number;
  missedAudioDuration: number;
  totalAudioMessages: number;
  delay: number;
  minPlaybackDelay: number;
  maxPlaybackDelay: number;
};

type useServerAudioArgs = {
  setGetAudioStats?: (getAudioStats: () => AudioStats) => void;
  setGetAudioDiagLog?: (getAudioDiagLog: () => AudioDiagSample[]) => void;
  setGetSocketOpenPerfMs?: (getSocketOpenPerfMs: () => number | null) => void;
};

type WorkletStats = {
  totalAudioPlayed: number;
  actualAudioPlayed: number;
  delay: number;
  minDelay: number;
  maxDelay: number;
  liveBufferS?: number;
  events?: Record<string, unknown>[];
};

/**
 * One sample of the client-side audio-playback jitter-buffer diagnostic,
 * timed relative to this hook's own socket-open moment (not epoch time —
 * client and server may run on different machines with unsynchronized
 * clocks, but socket-open on the client and Channel construction on the
 * server happen within one WebSocket handshake of each other, so t_rel_s
 * here lines up with server.log/raw_events.jsonl's own t_rel_s closely
 * enough to correlate turn-by-turn).
 *
 * Two different numbers, do not conflate them (a real mistake made
 * building this diagnostic the first time):
 * - `liveBufferS`: audio-processor.ts's currentSamples()/sampleRate at
 *   the moment of this sample — the actual, bounded (~0 to
 *   totalMaxBufferSamples(), a few tens of ms) jitter-buffer occupancy.
 *   This is the real queue-depth signal.
 * - `delay`: the worklet's own mic_duration-minus-played-stream-time
 *   figure. Despite the name, this is NOT buffer occupancy — it only
 *   advances while the model is actively producing audio, so it grows
 *   for the entire duration the model isn't speaking (user's turn,
 *   retrieval wait, silence) and is unbounded over a session length.
 *   Kept for parity with the live "Latency" UI stat, which already reads
 *   this same field — not a queue-depth measure despite resembling one.
 */
export type AudioDiagSample = {
  t_rel_s: number;
  liveBufferS: number;
  delay: number;
  actualAudioPlayed: number;
  totalAudioPlayed: number;
  events?: Record<string, unknown>[];
};

export const useServerAudio = ({setGetAudioStats, setGetAudioDiagLog, setGetSocketOpenPerfMs}: useServerAudioArgs) => {
  const { socket  } = useSocketContext();
  const {startRecording, stopRecording, audioContext, worklet, micDuration, actualAudioPlayed } =
    useMediaContext();
  const analyser = useRef(audioContext.current.createAnalyser());
  worklet.current.connect(analyser.current);
  const startTime = useRef<number | null>(null);
  const decoderWorker = useRef(DecoderWorker);
  const [hasCriticalDelay, setHasCriticalDelay] = useState(false);
  const totalAudioMessages = useRef(0);
  const receivedDuration = useRef(0);
  const workletStats = useRef<WorkletStats>({
    totalAudioPlayed: 0,
    actualAudioPlayed: 0,
    delay: 0,
    minDelay: 0,
    maxDelay: 0,});
  // Jitter-buffer diagnostic (see project_demo_audio_quality_investigation
  // memory / CLAUDE.md "Running the demo" section): socketOpenRef anchors
  // t_rel_s for cross-referencing against server.log/raw_events.jsonl's own
  // t_rel_s; diagLogRef accumulates one sample per worklet message
  // (~every 80ms during active playback). Capped defensively — a demo
  // session is minutes long, not hours, but this avoids unbounded growth
  // if a session runs long.
  const socketOpenRef = useRef<number | null>(null);
  const diagLogRef = useRef<AudioDiagSample[]>([]);
  const AUDIO_DIAG_LOG_CAP = 20000;

  const onDecode = useCallback(
    async (data: Float32Array) => {
      receivedDuration.current += data.length / audioContext.current.sampleRate;
      worklet.current.port.postMessage({frame: data, type: "audio", micDuration: micDuration.current});
    },
    [],
  );

  const onWorkletMessage = useCallback(
    (event: MessageEvent<WorkletStats>) => {
      workletStats.current = event.data;
      actualAudioPlayed.current = workletStats.current.actualAudioPlayed;
      if (socketOpenRef.current !== null) {
        const sample: AudioDiagSample = {
          t_rel_s: (performance.now() - socketOpenRef.current) / 1000,
          liveBufferS: event.data.liveBufferS ?? 0,
          delay: event.data.delay,
          actualAudioPlayed: event.data.actualAudioPlayed,
          totalAudioPlayed: event.data.totalAudioPlayed,
        };
        if (event.data.events && event.data.events.length > 0) {
          sample.events = event.data.events;
        }
        diagLogRef.current.push(sample);
        if (diagLogRef.current.length > AUDIO_DIAG_LOG_CAP) {
          diagLogRef.current.splice(0, diagLogRef.current.length - AUDIO_DIAG_LOG_CAP);
        }
      }
    },
    [],
  );
  worklet.current.port.onmessage = onWorkletMessage;

  const getAudioStats = useCallback(() => {
    return {
      playedAudioDuration: workletStats.current.actualAudioPlayed,
      delay: workletStats.current.delay,
      minPlaybackDelay: workletStats.current.minDelay,
      maxPlaybackDelay: workletStats.current.maxDelay,
      missedAudioDuration: workletStats.current.totalAudioPlayed - workletStats.current.actualAudioPlayed,
      totalAudioMessages: totalAudioMessages.current,
    };
  }, []);

  const getAudioDiagLog = useCallback(() => diagLogRef.current, []);
  // Exposes the same performance.now() reference AudioDiagSample's t_rel_s
  // is anchored to, so other consumers (useRecording.ts's saved-audio
  // filename) can compute their own offset into that same timeline
  // without needing a second, potentially-skewed clock reading.
  const getSocketOpenPerfMs = useCallback(() => socketOpenRef.current, []);

  const onWorkerMessage = useCallback(
    (e: MessageEvent<any>) => {
      if (!e.data) {
        return;
      }
      onDecode(e.data[0]);
    },
    [onDecode],
  );

  let midx = 0;
  const decodeAudio = useCallback((data: Uint8Array) => {
    if (midx < 5) {
      console.log(Date.now() % 1000, "Got NETWORK message", micDuration.current - workletStats.current.actualAudioPlayed, midx++);
    }
    decoderWorker.current.postMessage(
      {
        command: "decode",
        pages: data,
      },
      [data.buffer],
    );
  }, []);

  const onSocketMessage = useCallback(
    (e: MessageEvent) => {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);
      if (message.type === "audio") {
        decodeAudio(message.data);
        //For stats purposes for now
        totalAudioMessages.current++;
      }
    },
    [decodeAudio],
  );

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    worklet.current.port.postMessage({type: "reset"});
    console.log(Date.now() % 1000, "Should start in a bit");
    startRecording();
    currentSocket.addEventListener("message", onSocketMessage);
    totalAudioMessages.current = 0;
    socketOpenRef.current = performance.now();
    diagLogRef.current = [];
    return () => {
      console.log("Stop recording called in unknown function.")
      stopRecording();
      startTime.current = null;
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  useEffect(() => {
    if (setGetAudioStats) {
      console.log("Setting getAudioStats");
      setGetAudioStats(getAudioStats);
    }
  }, [setGetAudioStats, getAudioStats]);

  useEffect(() => {
    if (setGetAudioDiagLog) {
      setGetAudioDiagLog(getAudioDiagLog);
    }
  }, [setGetAudioDiagLog, getAudioDiagLog]);

  useEffect(() => {
    if (setGetSocketOpenPerfMs) {
      setGetSocketOpenPerfMs(getSocketOpenPerfMs);
    }
  }, [setGetSocketOpenPerfMs, getSocketOpenPerfMs]);

  useEffect(() => {
    decoderWorker.current.onmessage = onWorkerMessage;
    // 960 = 24000 / 12.5 / 2
    // The /2 is a bit optional, but won't hurt for recording the mic, and for the
    // the decoding it might help getting some decoded audio out asap.
    decoderWorker.current.postMessage({
      command: "init",
      bufferLength: 960 * audioContext.current.sampleRate / 24000,
      decoderSampleRate: 24000,
      outputBufferSampleRate: audioContext.current.sampleRate,
      resampleQuality: 0,
    });

    return () => {
      console.log("Terminating worker");
    };
  }, [onWorkerMessage]);

  return {
    decodeAudio,
    analyser,
    getAudioStats,
    getAudioDiagLog,
    getSocketOpenPerfMs,
    hasCriticalDelay,
    setHasCriticalDelay,
  };
};
