#!/usr/bin/env python3
"""Audition scorecard: serve.log + filled session JSONs → numbers + known_good.md.

Replaces single-session judgement calls (which produced the wrong "checkpoint 500
regression" verdict) with the same fixed questions scored the same way every time.
Verdict weighting is the decision of record: stalls + voice/persona first, triggering
second. known_good.md IS the Sunday demo script.

Log patterns come from spec §6 greps. If a real serve.log doesn't match, fix the
constants below — nowhere else.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

TRIGGER_PAT = re.compile(r"model emitted")
REFERENCE_PAT = re.compile(r"Generated reference")
STALL_PAT = re.compile(r"LM buffer empty")
BACKSTOP_PAT = re.compile(r"\[Backstop\] engaged")
STEP_PAT = re.compile(r"batched step.*?([\d.]+)\s*ms")
TS_PAT = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
STALL_MERGE_GAP_SEC = 1.0  # stall lines closer than this are one stall event

QUESTIONS = [
    {"id": "q01", "kind": "greeting"},
    {"id": "q02", "kind": "persona"},
    {"id": "q03", "kind": "persona"},
    {"id": "q04", "kind": "factual"},
    {"id": "q05", "kind": "factual"},
    {"id": "q06", "kind": "factual"},
    {"id": "q07", "kind": "decline"},
    {"id": "q08", "kind": "personal"},
    {"id": "q09", "kind": "persona"},
    {"id": "q10", "kind": "persona"},
    {"id": "q11", "kind": "persona"},
    {"id": "q12", "kind": "persona"},
]
KINDS = {q["id"]: q["kind"] for q in QUESTIONS}


def _ts(line):
    m = TS_PAT.search(line)
    if not m:
        return None
    raw = m.group(1).replace(",", ".").replace("T", " ")
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(raw, fmt)


def parse_log(text):
    triggers, references, stalls, step_ms = [], [], [], []
    n_trig = n_ref = n_stall = n_backstop = 0
    for line in text.splitlines():
        t = _ts(line)
        if TRIGGER_PAT.search(line):
            n_trig += 1
            triggers.append(t)
        if REFERENCE_PAT.search(line):
            n_ref += 1
            references.append(t)
        if STALL_PAT.search(line):
            n_stall += 1
            stalls.append(t)
        if BACKSTOP_PAT.search(line):
            n_backstop += 1
        m = STEP_PAT.search(line)
        if m:
            step_ms.append(float(m.group(1)))

    backstop_denom = n_trig + n_backstop
    backstop_rate = n_backstop / backstop_denom if backstop_denom else None

    longest = None
    ts_stalls = [t for t in stalls if t]
    if ts_stalls:
        longest, start, prev = 0.0, ts_stalls[0], ts_stalls[0]
        for t in ts_stalls[1:]:
            if (t - prev).total_seconds() > STALL_MERGE_GAP_SEC:
                longest = max(longest, (prev - start).total_seconds())
                start = t
            prev = t
        longest = round(max(longest, (prev - start).total_seconds()), 3)

    round_trips = []
    refs = [t for t in references if t]
    ref_idx = 0
    for t in [t for t in triggers if t]:
        while ref_idx < len(refs) and refs[ref_idx] < t:
            ref_idx += 1
        if ref_idx < len(refs):
            round_trips.append(round((refs[ref_idx] - t).total_seconds(), 3))
            ref_idx += 1

    return {"triggers": n_trig, "references": n_ref, "stall_lines": n_stall,
            "longest_stall_sec": longest, "round_trips": round_trips,
            "step_ms": step_ms, "backstops": n_backstop,
            "backstop_rate": backstop_rate}


def score(sessions, log_metrics):
    fired = asked = grounded = spurious = 0
    persona_ok = persona_n = decline_ok = decline_n = 0
    for s in sessions:
        for qid, r in s["results"].items():
            kind = KINDS.get(qid)
            if kind == "factual":
                asked += 1
                if r["fired"]:
                    fired += 1
                    grounded += bool(r["passed"])
            elif kind in ("persona", "personal", "greeting"):
                spurious += bool(r["fired"])
                if kind == "persona":
                    persona_n += 1
                    persona_ok += bool(r["passed"])
            elif kind == "decline":
                decline_n += 1
                decline_ok += bool(r["passed"])
    return {
        "checkpoint": sessions[0]["checkpoint"],
        "scaling": sessions[0]["scaling"],
        "sessions": len(sessions),
        "trigger_rate": round(fired / asked, 3) if asked else None,
        "grounded_rate": round(grounded / fired, 3) if fired else None,
        "spurious_fires": spurious,
        "persona_rate": round(persona_ok / persona_n, 3) if persona_n else None,
        "decline_rate": round(decline_ok / decline_n, 3) if decline_n else None,
        "voice_mean": round(sum(s["voice"] for s in sessions) / len(sessions), 2),
        "interrupt_yields": [s.get("interrupt_yields") for s in sessions],
        **log_metrics,
    }


def known_good(sessions):
    demo, persona, avoid = [], [], []
    need = 2  # of 3 sessions — keeps lucky one-off fires out of the live demo
    for q in QUESTIONS:
        qid, kind = q["id"], q["kind"]
        rs = [s["results"][qid] for s in sessions if qid in s["results"]]
        if kind == "factual":
            ok = sum(1 for r in rs if r["fired"] and r["passed"])
        elif kind == "decline":
            ok = sum(1 for r in rs if r["passed"])
        else:
            ok = sum(1 for r in rs if r["passed"] and not r["fired"])
        bucket = persona if kind == "persona" else demo
        (bucket if ok >= need else avoid).append(qid)
    return {"demo": demo, "persona": persona, "avoid": avoid}


def _render(card, kg):
    lines = [f"# Scorecard — {card['checkpoint']} (scaling {card['scaling']})", ""]
    for k in ("trigger_rate", "grounded_rate", "spurious_fires", "persona_rate",
              "decline_rate", "voice_mean", "longest_stall_sec", "stall_lines",
              "round_trips", "step_ms", "interrupt_yields", "backstops",
              "backstop_rate"):
        lines.append(f"- **{k}**: {card[k]}")
    lines += ["", "## Demo script (fired+grounded or clean in ≥2/3 sessions)",
              ", ".join(kg["demo"]) or "(none)",
              "", "## Persona (in character, no fire, ≥2/3)",
              ", ".join(kg["persona"]) or "(none)",
              "", "## DO NOT ASK IN THE DEMO", ", ".join(kg["avoid"]) or "(none)"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sessions = [json.load(open(p)) for p in a.sessions]
    idents = {(s.get("checkpoint"), s.get("scaling")) for s in sessions}
    if len(idents) > 1:
        raise SystemExit(f"sessions mix checkpoints/scalings {sorted(idents)} — "
                         "one scorecard must come from ONE checkpoint at ONE scaling")
    card = score(sessions, parse_log(Path(a.log).read_text(errors="replace")))
    kg = known_good(sessions)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps({**card, "known_good": kg}, indent=2))
    (out / "scorecard.md").write_text(_render(card, kg))
    print(_render(card, kg))


if __name__ == "__main__":
    main()
