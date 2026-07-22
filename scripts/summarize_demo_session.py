#!/usr/bin/env python3
"""
Summarize a demo session's turns.jsonl + raw_events.jsonl (see
specs/moshirag-evals-requirements.md's "Session instrumentation" and
"Demo session log"/"Demo session summary (console)" sections).

Print the console report:
  uv run scripts/summarize_demo_session.py demo/sessions/2026-07-13T09-32-11Z/

Print one pretty-printed JSON document instead (turns.jsonl itself is
deliberately compact/append-only JSONL, not pretty-printed — see the spec
section above for why):
  uv run scripts/summarize_demo_session.py demo/sessions/2026-07-13T09-32-11Z/ --json

All aggregate stats (mean/p95 per retrieval-breakdown stage, mean/p95
ttfat_s, <ret> trigger rate) are computed on the fly here, not
precomputed/stored anywhere — same "no pre-computation in individual run
files" philosophy already used by evals/runner.py's comparison CLI.

e2ekd_s/keyword_delay_s are always None right now: evals/registry/latency/
e2ekd.py doesn't exist yet (see the spec's Context section on why it's
deliberately deferred), so every turn's e2ekd column reads "—".

turns.jsonl's own `turn` records carry NO retrieval fields at all —
rag_triggered/rag_trigger_count/retrievals are entirely computed here, by
compute_retrievals_from_raw_events(), from raw_events.jsonl's
ret_triggered/retrieval_complete/utterance_end events. This isn't just a
convenience: a `<ret>` trigger's turn_index tag, at the moment
instrumented_server.py writes it, is only ever a live, best-effort value —
the model frequently predicts the RAG token before our own VAD-based turn
boundary has caught up to the user having already finished the *next*
question (see instrumented_server.py's module docstring for the exact
mechanism). Only after the full session is over, with every trigger and
every boundary known, can this be attributed correctly — which is exactly
what this script does and the live process cannot. See
compute_retrievals_from_raw_events()'s own docstring for the precise
reattribution rule, and why it's not as simple as "shift every trigger
forward by one turn."

A turn can have more than one retrieval (e.g. the model re-triggers
mid-monologue with no new user input in between) — `[turns]`'s `<ret>`
column shows "yes ×N" for N>1, and ret.total lists every one's latency
(comma-separated) rather than collapsing them into a single number, so
that behavior stays visible instead of silently averaged/dropped.

The console report ends with a "[raw stream]" section reading
raw_events.jsonl — a chronological, un-turn-scoped interleaving of every
raw user/model text chunk and RAG event, tagged with the same live,
best-effort turn_index instrumented_server.py wrote (deliberately NOT
reattributed there — the raw stream's whole purpose is showing what
actually, literally happened, unbucketed). The "[turns]" table above it
buckets everything into tidy, correctly-reattributed per-turn rows, which
is exactly what would hide *why* the model fired `<ret>` too early, too
late, or unprompted if the raw stream didn't exist separately. Older
sessions recorded before raw_events.jsonl existed just show "n/a" here —
and, since retrieval data now lives there exclusively, every turn shows no
retrieval data at all for such sessions.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

SEP = "─" * 57  # matches evals/runner.py's SEP


def load_records(session_dir: Path) -> tuple[dict, list[dict], dict[int, dict]]:
    """Returns (session_start_record, turn_records, e2ekd_updates_by_turn_index)."""
    turns_path = session_dir / "turns.jsonl"
    if not turns_path.exists():
        print(f"No turns.jsonl found in {session_dir}", file=sys.stderr)
        sys.exit(1)

    session_start: dict | None = None
    turns: list[dict] = []
    e2ekd_updates: dict[int, dict] = {}

    with open(turns_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "session_start":
                session_start = record
            elif record_type == "turn":
                turns.append(record)
            elif record_type == "turn_e2ekd_update":
                e2ekd_updates[record["turn_index"]] = record

    if session_start is None:
        print(f"No session_start record found in {turns_path}", file=sys.stderr)
        sys.exit(1)

    return session_start, turns, e2ekd_updates


def load_raw_events(session_dir: Path) -> list[dict]:
    """raw_events.jsonl is optional — sessions recorded before it existed,
    or a session dir passed by mistake, just get an empty (n/a) raw stream
    rather than a hard failure like load_records' missing-turns.jsonl case."""
    raw_path = session_dir / "raw_events.jsonl"
    if not raw_path.exists():
        return []
    events = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def merge_e2ekd(turns: list[dict], e2ekd_updates: dict[int, dict]) -> list[dict]:
    """Join each turn with its (currently always absent) turn_e2ekd_update, by turn_index."""
    merged = []
    for turn in turns:
        update = e2ekd_updates.get(turn["turn_index"])
        merged_turn = dict(turn)
        if update:
            merged_turn["e2ekd_s"] = update.get("e2ekd_s")
            merged_turn["keyword_delay_s"] = update.get("keyword_delay_s")
        merged.append(merged_turn)
    return merged


def compute_retrievals_from_raw_events(turns: list[dict], raw_events: list[dict]) -> list[dict]:
    """Attributes every retrieval that happened in the session to the turn
    it actually grounds, computed entirely from raw_events.jsonl's
    chronological event order — not from anything baked into turns.jsonl's
    own turn records, which carry no retrieval fields at all (see
    instrumented_server.py's SessionLog docstring for why).

    The RAG token that triggers a retrieval is very often predicted by the
    model *before* our own VAD-based turn boundary (on_utterance_end) has
    caught up to the user having already finished asking the next
    question — see instrumented_server.py's module docstring. Confirmed
    against a real VM session: a turn can trigger `<ret>` *twice* — once
    right before its own boundary flips (a genuine same-turn re-trigger,
    e.g. the model correcting itself mid-monologue with no new user input)
    and once again much later, right before the *next* boundary — and the
    first must stay put while only the second carries forward. So the rule
    isn't "always shift by one": within each group of triggers that share
    an original turn_index tag, only the temporally *last* one is eligible
    to move to turn_index + 1, and only if that next turn's boundary
    actually happened (i.e. the session didn't end mid-response). Every
    earlier trigger in the same group stays exactly where it's tagged.

    Each turn ends up with a `retrievals` list (0, 1, or more entries — a
    turn can legitimately keep several, as above, once same-turn
    re-triggers are excluded from reassignment) instead of a single slot;
    `rag_triggered`/`rag_trigger_count` are derived from its length.
    """
    triggers = [(ev["t_rel_s"], ev["turn_index"]) for ev in raw_events if ev.get("event") == "ret_triggered"]
    completions = [
        (
            ev["t_rel_s"],
            {
                "context": ev.get("retrieval_context"),
                "reference_text": ev.get("retrieved_reference_text"),
                "breakdown": ev.get("retrieval_breakdown_s"),
            },
        )
        for ev in raw_events
        if ev.get("event") == "retrieval_complete"
    ]
    boundary_turns = {ev["turn_index"] for ev in raw_events if ev.get("event") == "utterance_end"}

    # Pair by chronological order — retrievals are serialized server-side
    # (RAGManager.trigger() cancels any prior in-flight one before starting
    # a new one, confirmed against real moshi-rag source), so no true
    # overlaps. If a trigger were ever cancelled before completing (no
    # observed case in any real session so far), it would simply have no
    # completion to pair with here and gets dropped — a known,
    # deliberately-not-engineered-around simplification.
    groups: dict[int, list[dict]] = {}
    for (_, trigger_turn), (_, data) in zip(triggers, completions):
        groups.setdefault(trigger_turn, []).append(data)

    retrievals_by_turn: dict[int, list[dict]] = {}
    for original_turn, group in groups.items():
        for data in group[:-1]:  # same-turn re-triggers — never reassigned
            retrievals_by_turn.setdefault(original_turn, []).append(data)
        last = group[-1]
        target_turn = original_turn + 1 if (original_turn + 1) in boundary_turns else original_turn
        retrievals_by_turn.setdefault(target_turn, []).append(last)

    computed = []
    for turn in turns:
        retrievals = retrievals_by_turn.get(turn["turn_index"], [])
        merged_turn = dict(turn)
        merged_turn["retrievals"] = retrievals
        merged_turn["rag_triggered"] = len(retrievals) > 0
        merged_turn["rag_trigger_count"] = len(retrievals)
        computed.append(merged_turn)
    return computed


def _mean_p95(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 95))


def _fmt_s(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "—"


def _fmt_time(timestamp: str) -> str:
    # now_iso() always emits "%Y-%m-%dT%H:%M:%SZ" — fixed-width, safe to slice.
    return timestamp[11:19] if len(timestamp) >= 19 else timestamp


def _fmt_ret_flag(retrievals: list[dict]) -> str:
    n = len(retrievals)
    if n == 0:
        return "no"
    if n == 1:
        return "yes"
    return f"yes ×{n}"


def _fmt_ret_total(retrievals: list[dict]) -> str:
    """One value for the common single-retrieval case; comma-separated for
    a turn with more than one (see compute_retrievals_from_raw_events) —
    deliberately not summed/averaged, so multiple retrievals stay visible
    as the distinct events they were rather than one blended number."""
    if not retrievals:
        return "—"
    totals = [(r.get("breakdown") or {}).get("total_s") for r in retrievals]
    return ", ".join(_fmt_s(t) for t in totals)


_RAW_EVENT_LABELS = {
    "utterance_end": "· · · turn boundary (utterance end / VAD) · · ·",
    "ret_triggered": "<ret> triggered",
    "retrieval_complete": "retrieval complete",
    "first_audio": "first audio",
}


def _fmt_raw_event(ev: dict) -> str:
    # turn_index here is deliberately the raw, live-tagged value written by
    # instrumented_server.py — NOT the reattributed one [turns]/[full
    # transcript] use (see compute_retrievals_from_raw_events). Showing the
    # raw tag is the point: this section exists to show what actually,
    # literally happened, unbucketed.
    t = f"+{ev.get('t_rel_s', 0.0):>8.3f}s"
    turn = f"turn{ev.get('turn_index', 0)}"
    kind = ev.get("event", "?")
    if kind == "user_text":
        return f"  {t}  {turn:<6}  USER   {ev.get('text', '')}"
    if kind == "model_text":
        return f"  {t}  {turn:<6}  MOSHI  {ev.get('text', '')}"
    if kind == "retrieval_complete":
        breakdown = ev.get("retrieval_breakdown_s") or {}
        return f"  {t}  {turn:<6}  {_RAW_EVENT_LABELS[kind]} ({_fmt_s(breakdown.get('total_s'))})"
    if kind == "first_audio":
        return f"  {t}  {turn:<6}  {_RAW_EVENT_LABELS[kind]} (ttfat {_fmt_s(ev.get('ttfat_s'))})"
    label = _RAW_EVENT_LABELS.get(kind, kind)
    return f"  {t}  {turn:<6}  {label}"


def print_raw_stream_report(raw_events: list[dict]) -> None:
    print("[raw stream]")
    print("  Chronological, un-turn-scoped view of exactly what streamed in from the")
    print("  user and out from the model — use this, not the turn table above, to see")
    print("  <ret> firing too early, too late, or unprompted: turn boundaries here are")
    print("  just one more timestamped event, not an assumed structure on the data.")
    print()
    if not raw_events:
        print("  n/a — no raw_events.jsonl for this session (older session, recorded")
        print("  before this file existed)")
        print()
        return

    for ev in raw_events:
        print(_fmt_raw_event(ev))
    print()


# Retrieval-detail text wraps at this width with a *hanging* indent — the
# continuation of a long line indents deeper (12 spaces) than its first
# line (10 spaces), so a wrapped line still reads as "still inside this
# block" rather than colliding visually with a fresh `#N user:`/`#N moshi:`
# line at column 2. Letting the terminal do the wrapping instead (the
# original approach) loses this distinction entirely, since the terminal
# has no idea these lines are logically indented.
_RETRIEVAL_DETAIL_WIDTH = 96
_RETRIEVAL_DETAIL_INDENT = " " * 10
_RETRIEVAL_DETAIL_CONT_INDENT = " " * 12


def _print_wrapped_retrieval_text(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(
            textwrap.fill(
                line,
                width=_RETRIEVAL_DETAIL_WIDTH,
                initial_indent=_RETRIEVAL_DETAIL_INDENT,
                subsequent_indent=_RETRIEVAL_DETAIL_CONT_INDENT,
            )
        )


def _print_retrieval_detail(turn: dict) -> None:
    """Prints what a retrieving turn actually saw/got back, deeper-indented
    and separated from the `#N user:`/`#N moshi:` lines above it so a reader
    scanning just for the spoken conversation can skip straight past it —
    the user's own framing for why this isn't inline with the dialogue.
    No-op for turns with no retrieval data (the common case). A turn can
    have more than one retrieval (see compute_retrievals_from_raw_events) —
    each one gets its own numbered "retrieval i/N" block rather than being
    collapsed or only showing one."""
    retrievals = turn.get("retrievals", [])
    if not retrievals:
        return
    n = len(retrievals)
    print()
    for i, retrieval in enumerate(retrievals, start=1):
        label = f"retrieval {i}/{n}" if n > 1 else "retrieval"
        context = retrieval.get("context")
        reference = retrieval.get("reference_text")
        if context:
            print(f"        {label} (what the model asked with):")
            _print_wrapped_retrieval_text(context)
        if reference:
            print(f"        {label} (what came back):")
            _print_wrapped_retrieval_text(reference)
        if i < n:
            print()


def print_console_report(session_start: dict, turns: list[dict], raw_events: list[dict] | None = None) -> None:
    print("MoshiRAG Demo Session")
    print(f"session id : {session_start['session_id']}")
    print(f"checkpoint : {session_start['checkpoint']}")
    print(f"git hash   : {session_start['git_hash']}")
    backend = session_start.get("retrieval_backend") or {}
    print(f"retrieval  : {backend.get('model', '?')} (rag_timeout: {session_start.get('rag_timeout_s', '?')}s)")
    print(SEP)
    print()

    print("[turns]")
    # ret.total's field width (10 + one guaranteed literal space, rather
    # than a plain <11) so a multi-retrieval row's comma-separated list —
    # which can easily run past the normal width — still gets at least one
    # space of separation from the ttfat column instead of butting right up
    # against it. Ragged alignment on those rare rows is an accepted
    # tradeoff; a completely missing separator isn't.
    print(f"  {'#':<4}{'time':<10}{'question':<44}{'<ret>':<7}{'ret.total':<10} {'ttfat':<8}{'e2ekd'}")
    for turn in turns:
        question = turn.get("user_question_text") or ""
        if len(question) > 42:
            question = question[:39] + "..."
        retrievals = turn.get("retrievals", [])
        print(
            f"  {turn['turn_index']:<4}{_fmt_time(turn['timestamp']):<10}{question:<44}"
            f"{_fmt_ret_flag(retrievals):<7}{_fmt_ret_total(retrievals):<10} "
            f"{_fmt_s(turn.get('ttfat_s')):<8}{_fmt_s(turn.get('e2ekd_s'))}"
        )
    print()
    print(SEP)

    print("[aggregate]")
    n_turns = len(turns)
    n_triggered = sum(1 for t in turns if t.get("rag_triggered"))
    trigger_rate = (n_triggered / n_turns * 100) if n_turns else 0.0
    print(f"  turns              : {n_turns}")
    print(f"  <ret> trigger rate : {trigger_rate:.1f}%   ({n_triggered}/{n_turns})")

    # Flattened across every individual retrieval, not one-per-turn — a
    # turn with multiple retrievals (see compute_retrievals_from_raw_events)
    # contributes each of them here, so n reflects how many retrievals
    # actually happened, not how many turns triggered at least one.
    breakdowns = [r["breakdown"] for t in turns for r in t.get("retrievals", []) if r.get("breakdown")]
    if breakdowns:
        stages = [
            ("asr_wait", "asr_wait_s"),
            ("api_call", "api_call_s"),
            ("ctx_inj", "context_injection_s"),
            ("total", "total_s"),
        ]
        for i, (label, key) in enumerate(stages):
            mean, p95 = _mean_p95([b[key] for b in breakdowns])
            prefix = "  retrieval breakdown  " if i == 0 else "                       "
            print(f"{prefix}{label:<9} mean {mean:.2f}s   p95 {p95:.2f}s   (n={len(breakdowns)})")
    else:
        print("  retrieval breakdown  n/a — <ret> never fired this session")

    ttfats = [t["ttfat_s"] for t in turns if t.get("ttfat_s") is not None]
    # "ttfat" padded to the same column width the "retrieval breakdown"
    # rows above use for their "  retrieval breakdown  {label:<9} " prefix,
    # so both sections' "mean"/"p95" line up in the same columns.
    if ttfats:
        mean, p95 = _mean_p95(ttfats)
        print(f"  {'ttfat':<30} mean {mean:.2f}s   p95 {p95:.2f}s   (n={len(ttfats)})")
    else:
        print(f"  {'ttfat':<30} n/a — no turns completed")

    print('  ⚠ e2ekd not yet implemented — evals/registry/latency/e2ekd.py doesn\'t')
    print('    exist yet; this column stays "—" until that eval lands and the demo')
    print("    mirrors it")
    print()
    print(SEP)
    print()

    print("[full transcript]")
    for turn in turns:
        question = turn.get("user_question_text") or "—"
        response = turn.get("model_response_text") or "—"
        print(f"  #{turn['turn_index']} user : {question}")
        print(f"  #{turn['turn_index']} moshi: {response}")
        _print_retrieval_detail(turn)
        print()
    print(SEP)
    print()

    print_raw_stream_report(raw_events or [])


def print_json_report(session_start: dict, turns: list[dict], session_dir: Path) -> None:
    doc = {**session_start, "turns": turns}
    print(json.dumps(doc, indent=2))
    print(f"raw logs: {session_dir}/{{server,conditioner}}.log", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=str, help="Path to demo/sessions/<session_id>/")
    parser.add_argument("--json", action="store_true", help="Print one pretty-printed JSON document instead of the console report")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    session_start, raw_turns, e2ekd_updates = load_records(session_dir)
    raw_events = load_raw_events(session_dir)
    turns = compute_retrievals_from_raw_events(merge_e2ekd(raw_turns, e2ekd_updates), raw_events)

    if args.json:
        print_json_report(session_start, turns, session_dir)
    else:
        print_console_report(session_start, turns, raw_events)
        print(f"raw logs: {session_dir}/{{server,conditioner}}.log")


if __name__ == "__main__":
    main()
