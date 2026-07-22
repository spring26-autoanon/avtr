import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";

/**
 * Listens for the demo's own instrumentation payloads, pushed by
 * scripts/instrumented_server.py's _push_instrumentation()/_push_session_info()
 * over the same "metadata" WebSocket message type useServerInfo.ts already
 * uses for retrieval-backend capability announcements — but NOT through
 * that hook: useServerInfo validates every "metadata" message against a
 * strict Zod schema (ServersInfoSchema, requiring build_info/
 * text_temperature/etc, fields this payload doesn't have), so it would
 * silently fail .safeParse() and be dropped. This hook listens to the same
 * raw message stream independently and filters on its own "kind"
 * discriminator instead, so it never touches or depends on that schema.
 *
 * Three message kinds, kept in separate state slots rather than merged:
 *   "instrumentation" (ttfat_s only) turn-scoped progress, see
 *                                    InstrumentationTurn
 *   "instrumentation" (retrieval fields) see RetrievalInfo below for why
 *                                    this is tracked separately from the
 *                                    turn-scoped slot above
 *   "instrumentation_session"        session-level retrieval backend, sent
 *                                    once per connection — must not be
 *                                    dropped/overwritten as later per-turn
 *                                    messages arrive
 *
 * Field names intentionally stay snake_case, matching the wire payload
 * (see specs/moshirag-evals-requirements.md's "Demo session log" schema)
 * and this codebase's existing convention for server-payload types (see
 * ServerInfo in useServerInfo.ts).
 */

export interface RetrievalBreakdown {
  asr_wait_s: number;
  api_call_s: number;
  context_injection_s: number;
  total_s: number;
}

/** Turn-scoped progress only (ttfat) — safe to reset the instant a new turn
 * starts, unlike retrieval fields below. */
export interface InstrumentationTurn {
  turn_index: number;
  ttfat_s?: number;
}

/**
 * A retrieval only ever completes 2-4s (real observed api_call_s) after the
 * RAG token that requested it, long enough for a fast back-and-forth
 * exchange to advance the live turn boundary before results arrive (see
 * scripts/instrumented_server.py's _ChannelState.on_ret_triggered docstring
 * for the exact mechanism, confirmed against real moshi-rag source). If
 * this were folded into the same per-turn state as InstrumentationTurn, the
 * next turn's first (ttfat-only) message would wipe rag_triggered back to
 * "no" for the UI the instant it arrived — well before that answer had even
 * finished being spoken, and before the retrieval that grounded it had
 * resolved. Tracked separately here so it persists across a turn boundary
 * instead, tagged with turn_index (the turn that actually asked for it, per
 * on_ret_triggered), not whichever turn is live.
 *
 * Real retrievals are serialized server-side — RAGManager.trigger() cancels
 * any prior in-flight retrieval before starting a new one — so a strictly
 * decreasing turn_index here would mean a genuinely stale/out-of-order
 * message, safe to ignore; anything else (equal or increasing) is real.
 */
export interface RetrievalInfo {
  turn_index: number;
  rag_triggered?: boolean;
  rag_trigger_count?: number;
  retrieval_context?: string;
  retrieved_reference_text?: string;
  retrieval_breakdown_s?: RetrievalBreakdown;
}

export interface RetrievalBackendInfo {
  model: string;
  base_url: string;
}

type InstrumentationFields = Partial<InstrumentationTurn> & Partial<Omit<RetrievalInfo, "turn_index">>;

const RETRIEVAL_FIELD_KEYS: (keyof Omit<RetrievalInfo, "turn_index">)[] = [
  "rag_triggered",
  "rag_trigger_count",
  "retrieval_context",
  "retrieved_reference_text",
  "retrieval_breakdown_s",
];

export const useInstrumentation = () => {
  const [turn, setTurn] = useState<InstrumentationTurn | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalInfo | null>(null);
  const [retrievalBackend, setRetrievalBackend] = useState<RetrievalBackendInfo | null>(null);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type !== "metadata") {
      return;
    }
    const data = message.data as
      | { kind?: string; turn_index?: number; fields?: InstrumentationFields; retrieval_backend?: RetrievalBackendInfo }
      | null;
    if (!data || typeof data.kind !== "string") {
      return;
    }

    if (data.kind === "instrumentation_session") {
      if (data.retrieval_backend) {
        setRetrievalBackend(data.retrieval_backend);
      }
      return;
    }

    if (data.kind !== "instrumentation" || typeof data.turn_index !== "number") {
      return;
    }
    const turnIndex = data.turn_index;
    const fields = data.fields ?? {};

    if (typeof fields.ttfat_s === "number") {
      const ttfatS = fields.ttfat_s;
      setTurn((prev) => (prev && turnIndex < prev.turn_index ? prev : { turn_index: turnIndex, ttfat_s: ttfatS }));
    }

    if (RETRIEVAL_FIELD_KEYS.some((key) => fields[key] !== undefined)) {
      setRetrieval((prev) => {
        if (prev && turnIndex < prev.turn_index) {
          return prev; // genuinely stale — see RetrievalInfo's docstring
        }
        if (prev && prev.turn_index === turnIndex) {
          return { ...prev, ...fields, turn_index: turnIndex };
        }
        return { turn_index: turnIndex, ...fields };
      });
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setTurn(null);
    setRetrieval(null);
    setRetrievalBackend(null);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket, onSocketMessage]);

  return { instrumentation: turn, retrieval, retrievalBackend };
};
