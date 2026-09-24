"""Step 2: deterministic rules layer. python -m qa.rules [--validate]"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

SYN = Path("data/synthetic")
OUT = Path("out")


def _d(s):
    return dt.date.fromisoformat(str(s)[:10]) if s else None


def _on_or_before(s, day):
    return s is not None and _d(s) <= day


# form outcome_status -> predicate(case, call_date) that is True when the form is CONSISTENT with truth.
OUTCOME_MAP = {
    # transfer_confirm_call
    "received":     lambda c, day: _on_or_before(c["truth"]["received_at"], day),
    "not_received": lambda c, day: not _on_or_before(c["truth"]["received_at"], day),
    "filled":       lambda c, day: _on_or_before(c["truth"]["filled_at"], day),
    "needs_pa":     lambda c, day: _on_or_before(c["truth"]["received_at"], day)
                                   and c["truth"]["blocker"] == "needs_pa"
                                   and not _on_or_before(c["truth"]["filled_at"], day),
    # pa_plan_call
    "approved":     lambda c, day: c["truth"]["status"] == "approved" and _on_or_before(c["truth"]["determination_date"], day),
    "denied":       lambda c, day: c["truth"]["status"] == "denied" and _on_or_before(c["truth"]["determination_date"], day),
    "pending":      lambda c, day: c["truth"]["status"] == "pending",
    "not_on_file":  lambda c, day: c["truth"]["status"] == "not_on_file",
    # patient_access_check (denied shared with PA above only when case_type matches; handled in outcome_consistent)
    "enrolled":     lambda c, day: c["truth"]["status"] == "enrolled",
    "pending_docs": lambda c, day: c["truth"]["status"] == "pending_docs",
    "pending_bv":   lambda c, day: c["truth"]["status"] == "pending_bv",
    # any type
    "no_answer":    lambda c, day: True,
}


def outcome_consistent(case, outcome, call_date):
    day = _d(call_date)
    if outcome == "denied" and case["case_type"] == "patient_access_check":
        return case["truth"]["status"] == "denied"
    fn = OUTCOME_MAP.get(outcome)
    return True if fn is None else bool(fn(case, day))


def true_status(case, call_date):
    """What a truthful rep would say on call_date."""
    day, t = _d(call_date), case["truth"]
    if case["case_type"] == "transfer_confirm_call":
        if _on_or_before(t["filled_at"], day):
            return "filled"
        if _on_or_before(t["received_at"], day):
            return "needs_pa" if t["blocker"] == "needs_pa" else "received"
        return "not_received"
    return t["status"]


# ---------------------------------------------------------------- loaders
def load_inputs():
    sample = pd.read_csv(SYN / "calls_sample.csv", dtype={"call_id": str})
    forms = {f["call_id"]: f for f in map(json.loads, (SYN / "forms.jsonl").read_text().splitlines())}
    cases = {c["case_ref"]: c for c in map(json.loads, (SYN / "cases.jsonl").read_text().splitlines())}
    tx = {}
    p = SYN / "transcripts.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                tx[o["call_id"]] = o["turns"]
    return sample, forms, cases, tx


def flag(cid, code, sev, reason):
    return {"call_id": cid, "code": code, "severity": sev, "reason": reason}


# ---------------------------------------------------------------- rules
def r_fab_contact(r, form, case, turns):
    claims = form["spoke_with_rep"] or form.get("rep_name") or form.get("reference_number")
    if not claims:
        return None
    if r.duration_seconds < 30:
        return flag(r.call_id, "FAB_CONTACT", "critical",
                    f"spoke_with_rep={form['spoke_with_rep']}, rep_name={form.get('rep_name')!r} but duration={r.duration_seconds}s")
    if turns is not None and not any(t["speaker"] == "REP" for t in turns):
        return flag(r.call_id, "FAB_CONTACT", "critical", "form claims rep contact but transcript has no REP turns")


def r_outcome_vs_rx(r, form, case, turns):
    if not outcome_consistent(case, form["outcome_status"], r.call_date):
        t = case["truth"]
        facts = ", ".join(f"{k}={v}" for k, v in t.items() if k in ("status", "received_at", "filled_at", "blocker", "determination_date"))
        return flag(r.call_id, "OUTCOME_VS_RX", "critical",
                    f"form outcome={form['outcome_status']} but record on {r.call_date}: {facts}")


def r_missing_ref(r, form, case, turns):
    if r.call_type != "transfer_confirm_call" and r.connected and not form.get("reference_number"):
        return flag(r.call_id, "MISSING_REF", "major", f"connected {r.call_type} ({r.duration_seconds}s) with empty reference_number")


def r_hold_overrun(r, form, case, turns):
    if r.duration_seconds > r.hold_limit + 60:
        return flag(r.call_id, "HOLD_OVERRUN", "minor", f"duration={r.duration_seconds}s > hold limit {r.hold_limit}s + 60")


def r_after_hours(r, form, case, turns):
    if r.et_hour >= 19 or r.et_hour < 8:
        return flag(r.call_id, "AFTER_HOURS", "minor", f"placed at {r.et_hour}:00 ET")


RULES = [r_fab_contact, r_outcome_vs_rx, r_missing_ref, r_hold_overrun, r_after_hours]


def run_rules(sample=None, forms=None, cases=None, tx=None):
    if sample is None:
        sample, forms, cases, tx = load_inputs()
    flags = []
    for r in sample.itertuples():
        form, case = forms[r.call_id], cases[r.case_ref]
        for fn in RULES:
            f = fn(r, form, case, tx.get(r.call_id))
            if f:
                flags.append(f)
    # REDUNDANT_CALL: resolved before call, or 2nd+ call to same case_ref same day
    seen = {}
    s = sample.copy()
    s["_h"] = s["hour_of_day"].astype(str)
    for r in s.sort_values(["call_date", "_h", "call_id"]).itertuples():
        case = cases[r.case_ref]
        key = (r.case_ref, r.call_date)
        if case.get("resolved_before_call"):
            flags.append(flag(r.call_id, "REDUNDANT_CALL", "minor", f"{r.case_ref} already resolved before call (filled {case['truth']['filled_at']})"))
        elif key in seen:
            flags.append(flag(r.call_id, "REDUNDANT_CALL", "minor", f"{r.case_ref} already called earlier on {r.call_date} ({seen[key]})"))
        seen.setdefault(key, r.call_id)
    return pd.DataFrame(flags, columns=["call_id", "code", "severity", "reason"])


def prf(pred, gold):
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    return tp, fp, fn, p, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = run_rules()
    df.to_csv(OUT / "rule_flags.csv", index=False)
    print(f"rule flags: {len(df)} on {df.call_id.nunique()} calls -> out/rule_flags.csv")
    print(df.code.value_counts().to_string())
    if a.validate:
        lab = pd.read_csv(SYN / "labels.csv", dtype={"call_id": str})
        print(f"\n{'code':16s} TP  FP  FN  prec  rec")
        for code in sorted(df.code.unique()):
            tp, fp, fn, p, r = prf(set(df[df.code == code].call_id), set(lab[lab.code == code].call_id))
            print(f"{code:16s} {tp:3d} {fp:3d} {fn:3d}  {p:.2f}  {r:.2f}")


if __name__ == "__main__":
    main()
