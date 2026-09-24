"""Verify that planted, judge-detectable defects actually manifest in the rendered transcripts.

python -m gen.verify_labels          (also runs at the end of gen.generate)

The renderer is an LLM, so a planted behaviour can fail to appear (e.g. the hint says "agent never asks for the
appeal deadline" but the rep volunteers it). Such labels are noise for the judge, not misses. This adds a
`manifested` column to labels.csv (True/False for judge codes, blank for rule-derived codes) plus a `detail`
column; qa.validate drops rows with manifested == False.
"""
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

SYN = Path(os.environ.get("SYN_DIR", "data/synthetic"))
JUDGE_CODES = {"FAB_CONTACT", "STATUS_MISMATCH", "PHI_DISCLOSURE", "UNPROFESSIONAL", "MISSED_NEXT_STEP", "MISSING_REF"}
REASON_KEYS = {"step therapy": ["step therapy", "topical"], "insufficient": ["documentation"],
               "non-formulary": ["formulary"], "quantity": ["quantity"]}
DOC_KEYS = {"income": ["income"], "consent": ["consent"], "insurance card": ["insurance card", "card"],
            "medical necessity": ["necessity"]}


def date_forms(iso):
    day = dt.date.fromisoformat(str(iso)[:10])
    return {str(iso)[:10], day.strftime("%B %-d").lower(), day.strftime("%b %-d").lower(),
            f"{day.month}/{day.day}", day.strftime("%m/%d"), day.strftime("%m-%d-%Y")}


def present(text, *needles):
    return any(n and str(n).lower() in text for n in needles)


def date_present(text, iso):
    return bool(iso) and any(f in text for f in date_forms(iso))


def _keys(table, value):
    for k, v in table.items():
        if value and k in value.lower():
            return v
    return [value] if value else []


def required_facts(case, status):
    """Facts a competent agent must leave the call with, given what the rep truthfully said."""
    t, ct = case["truth"], case["case_type"]
    if ct == "pa_plan_call":
        if status == "approved":
            return {"auth_number": lambda x: present(x, t.get("auth_number")),
                    "effective_dates": lambda x: date_present(x, t.get("determination_date")) or present(x, "effective")}
        if status == "denied":
            return {"denial_reason": lambda x: present(x, *_keys(REASON_KEYS, t.get("denial_reason"))),
                    "appeal_deadline": lambda x: date_present(x, t.get("appeal_deadline")),
                    "peer_to_peer": lambda x: present(x, "peer")}
        if status == "pending":
            return {"expected_determination": lambda x: (date_present(x, t.get("expected_determination"))
                                                          or present(x, "business day", " days", "week", "by the end of", "turnaround"))}
        return {"how_to_submit": lambda x: present(x, "submit", "fax", "portal", "send", "form")}
    if ct == "patient_access_check":
        if status == "pending_docs":
            return {f"doc:{doc}": (lambda x, doc=doc: present(x, *_keys(DOC_KEYS, doc))) for doc in t.get("missing_docs", [])}
        if status == "enrolled":
            return {"bv_result": lambda x: present(x, "copay", "coverage", "verified"),
                    "bridge": lambda x: present(x, "ship", "bridge")}
        if status == "pending_bv":  # BV has no result yet; the open question is the bridge supply
            return {"bridge": lambda x: present(x, "ship", "bridge")}
        return {"denial_reason": lambda x: present(x, "eligib", "government", "income", "coverage"),
                "appeal_or_alternative": lambda x: present(x, "appeal", "re-appl", "reappl", "alternative", "other program", "reconsider")}
    if status == "not_received":
        return {"resend": lambda x: present(x, "npi", "fax", "re-send", "resend", "resubmit", "send it again", "send over")}
    facts = {"pharmacy_rx_number": lambda x: present(x, t.get("pharmacy_rx_number"))}
    if status != "filled":
        facts["blockers"] = lambda x: (present(x, "back-order", "backorder", "back order", "prior auth", "reject",
                                               "queue", "expected", "eta", "in stock") or date_present(x, t.get("fill_eta")))
    return facts


def manifested(code, case, scen, turns, form):
    text = " ".join(t["text"] for t in turns).lower()
    agent = " ".join(t["text"] for t in turns if t["speaker"] == "AGENT").lower()
    rep = " ".join(t["text"] for t in turns if t["speaker"] == "REP").lower()
    has_rep = any(t["speaker"] == "REP" for t in turns)
    status = scen["rep_truth"]["status"]
    if code == "FAB_CONTACT":
        return (not has_rep) and bool(form["spoke_with_rep"]), "form claims contact, no REP turns" if not has_rep else "REP turns present"
    if code == "STATUS_MISMATCH":
        ok = has_rep and form["outcome_status"] != status
        return ok, f"rep said {status}, form says {form['outcome_status']}"
    if code == "UNPROFESSIONAL":
        ok = present(agent, "sigh", "just check again", "third time")
        return ok, "sigh / 'just check again' in AGENT turns" if ok else "no rude marker in AGENT turns"
    if code == "PHI_DISCLOSURE":
        p = case["patient"]
        street = p.get("address", "").split(",")[0].split(" ", 1)[-1].split(" ")[0]  # "8757 Maple Ave" -> "Maple"
        ok = present(agent, street, case.get("diagnosis"), p.get("member_id"))
        return ok, "address / diagnosis / member ID spoken by AGENT" if ok else "no extra PHI in AGENT turns"
    if code == "MISSING_REF":
        ref = scen.get("reference_number") or scen["rep_truth"].get("pharmacy_rx_number")
        ok = bool(ref) and present(rep, ref) and not form.get("reference_number")
        return ok, f"REP gave {ref}, form empty" if ok else "reference never given by REP"
    if code == "MISSED_NEXT_STEP":
        facts = required_facts(case, status)
        missing = [k for k, fn in facts.items() if not fn(text)]
        return bool(missing), ("missing: " + ", ".join(missing)) if missing else "all required facts were stated"
    return None, ""


def run(syn=SYN):
    labels = pd.read_csv(syn / "labels.csv", dtype={"call_id": str})
    cases = {c["case_ref"]: c for c in map(json.loads, (syn / "cases.jsonl").read_text().splitlines())}
    scen = {s["call_id"]: s for s in map(json.loads, (syn / "scenarios.jsonl").read_text().splitlines())}
    forms = {f["call_id"]: f for f in map(json.loads, (syn / "forms.jsonl").read_text().splitlines())}
    tx = {}
    if (syn / "transcripts.jsonl").exists():
        for line in (syn / "transcripts.jsonl").read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                tx[o["call_id"]] = o["turns"]
    labels = labels[labels.planted_by != "derived-facts"]  # recomputed below; keeps the pass idempotent
    out, detail = [], []
    for r in labels.itertuples():
        if r.code not in JUDGE_CODES or r.call_id not in tx:
            out.append(None)
            detail.append("")
            continue
        ok, why = manifested(r.code, cases[forms[r.call_id]["case_ref"]], scen[r.call_id], tx[r.call_id], forms[r.call_id])
        out.append(ok)
        detail.append(why)
    labels["manifested"] = out
    labels["detail"] = detail
    # Derived MISSED_NEXT_STEP: like OUTCOME_VS_RX, true by construction. Any connected call whose transcript lacks a
    # fact the rubric requires for the rep's stated status is a MISSED_NEXT_STEP, planted or not (the LLM renderer
    # does not always make the "competent" agent ask every applicable question).
    have = {(c, k) for c, k in zip(labels.call_id, labels.code)}
    derived = []
    for cid, turns in tx.items():
        if cid not in forms or (cid, "MISSED_NEXT_STEP") in have or not any(t["speaker"] == "REP" for t in turns):
            continue
        ok, why = manifested("MISSED_NEXT_STEP", cases[forms[cid]["case_ref"]], scen[cid], turns, forms[cid])
        if ok:
            derived.append({"call_id": cid, "code": "MISSED_NEXT_STEP", "severity": "major", "planted_by": "derived-facts",
                            "manifested": True, "detail": why})
    if derived:
        labels = pd.concat([labels, pd.DataFrame(derived)], ignore_index=True)
    labels.to_csv(syn / "labels.csv", index=False)
    print(f"derived MISSED_NEXT_STEP labels (required fact absent on an unplanted call): {len(derived)}")
    for d in derived:
        print(f"  derived  {d['call_id']}  ({d['detail']})")
    j = labels[labels.code.isin(JUDGE_CODES)]
    summary = j.groupby("code").agg(planted=("call_id", "size"),
                                    manifested=("manifested", lambda s: int((s == True).sum())))  # noqa: E712
    summary["dropped"] = summary.planted - summary.manifested
    print(f"label verification ({syn}): {int(summary.dropped.sum())} of {len(j)} judge-code labels did not manifest -> excluded")
    print(summary.to_string())
    for r in labels[labels.manifested == False].itertuples():  # noqa: E712
        print(f"  dropped {r.code:16s} {r.call_id}  ({r.detail})")
    return labels


if __name__ == "__main__":
    run()
