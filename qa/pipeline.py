"""Step 3: rules + judge + merge. python -m qa.pipeline [--limit N --force --no-llm]"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from qa import judge
from qa.rules import load_inputs, run_rules

OUT = Path(os.environ.get("OUT_DIR", "out"))
RUBRIC = yaml.safe_load(open("qa/rubric.yaml"))
W = RUBRIC["score"]["weights"]
# metadata-derived flags about the dialer / workflow, not the agent's conduct on the call
PROCESS_CODES = {"HOLD_OVERRUN", "AFTER_HOURS", "REDUNDANT_CALL"}


def merge(sample, flags, judged):
    rows, flat = [], []
    fl = flags.groupby("call_id")
    for call in sample.to_dict("records"):
        cid = call["call_id"]
        rf = fl.get_group(cid).to_dict("records") if cid in fl.groups else []
        jr = judged.get(cid)
        md = {}
        for f in rf:
            md[f["code"]] = {"code": f["code"], "severity": f["severity"], "source": "rules",
                             "kind": "process" if f["code"] in PROCESS_CODES else "agent",
                             "quote": None, "confidence": 1.0, "reason": f["reason"]}
        for d in (jr or {}).get("defects", []):
            ev = d["evidence"]
            quote = ev.get("quote") if ev["evidence_kind"] == "quote" else f"(absence) {d['rationale']}"
            if d["code"] in md:
                md[d["code"]].update(source="both", quote=quote, confidence=d["confidence"], turn=ev.get("turn"))
            else:
                md[d["code"]] = {"code": d["code"], "severity": d["severity"], "source": "judge", "kind": "agent",
                                 "quote": quote, "confidence": d["confidence"], "reason": d["rationale"], "turn": ev.get("turn")}
        merged = list(md.values())
        n = {s: sum(m["severity"] == s for m in merged) for s in ("critical", "major", "minor")}
        score = max(0, RUBRIC["score"]["base"] - W["critical"] * n["critical"] - W["major"] * n["major"] - W["minor"] * n["minor"])
        n_process = sum(m["kind"] == "process" for m in merged)
        # agent_score = the spec score without process flags (what a coaching conversation should be about)
        agent_score = max(0, RUBRIC["score"]["base"] - sum(W[m["severity"]] for m in merged if m["kind"] == "agent"))
        fc = (jr or {}).get("form_check", [])
        verifiable = [f for f in fc if f["match"] != "unverifiable"]
        form_acc = sum(f["match"] == "match" for f in verifiable) / len(verifiable) if verifiable else None
        auto_fill = sum(1 for f in fc if f.get("transcript_value") not in (None, "", "null"))
        usage = (jr or {}).get("usage") or {}
        review = n["critical"] > 0 or bool((jr or {}).get("needs_human_review"))
        res = {**(jr or {"call_id": cid, "checklist": [], "defects": [], "form_check": [], "coaching_note": "",
                          "needs_human_review": False, "model": None, "prompt_version": None, "evidence_failures": 0}),
               "judged": jr is not None,
               "rule_flags": [{k: f[k] for k in ("code", "severity", "reason")} for f in rf],
               "merged_defects": merged, "score": score, "agent_score": agent_score, "needs_human_review": review,
               "agent_id": call["agent_id"], "call_type": call["call_type"], "duration_seconds": call["duration_seconds"]}
        rows.append(res)
        flat.append({"call_id": cid, "agent_id": call["agent_id"], "call_type": call["call_type"],
                     "destination_name": call["destination_name"], "duration_seconds": call["duration_seconds"],
                     "et_hour": call["et_hour"], "connected": call["connected"], "judged": jr is not None,
                     "score": score, "n_critical": n["critical"], "n_major": n["major"], "n_minor": n["minor"],
                     "needs_human_review": review, "codes": ";".join(sorted(m["code"] for m in merged)),
                     "agent_score": agent_score, "n_process": n_process,
                     "form_accuracy": form_acc, "n_unverifiable": len(fc) - len(verifiable), "auto_fill_fields": auto_fill,
                     "evidence_failures": res["evidence_failures"], "guard_drops": res.get("guard_drops", 0),
                     "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                     "latency_s": res.get("latency_s")})
    return rows, pd.DataFrame(flat)


def write(rows, flat):
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    flat.to_csv(OUT / "scored_calls.csv", index=False)


def rebuild():
    """Merge current rules + cached judge output without calling the API (used by the app after a live re-run)."""
    sample, forms, cases, tx = load_inputs()
    flags = run_rules(sample, forms, cases, tx)
    rows, flat = merge(sample, flags, judge.load_raw())
    write(rows, flat)
    return rows, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    sample, forms, cases, tx = load_inputs()
    flags = run_rules(sample, forms, cases, tx)
    flags.to_csv(OUT / "rule_flags.csv", index=False)
    judged, st = asyncio.run(judge.judge_all(a.limit, a.force, a.no_llm))
    rows, flat = merge(sample, flags, judged)
    write(rows, flat)
    print(f"rules: {len(flags)} flags · judge: {st}")
    print(f"calls: {len(flat)} · judged: {int(flat.judged.sum())} · mean score {flat.score.mean():.1f} · "
          f"with critical: {(flat.n_critical > 0).mean():.0%} · needs review: {int(flat.needs_human_review.sum())}")
    print(f"evidence failures: {int(flat.evidence_failures.sum())} · guard drops: {int(flat.guard_drops.sum())} · "
          f"form accuracy (verifiable fields) {flat.form_accuracy.mean():.2f}")
    if flat.input_tokens.notna().any():
        print(f"judge tokens/call: in {flat.input_tokens.mean():.0f} · out {flat.output_tokens.mean():.0f} · "
              f"latency {flat.latency_s.mean():.1f}s")
    print("-> out/results.jsonl, out/scored_calls.csv")


if __name__ == "__main__":
    main()
