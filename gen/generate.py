"""Step 1: synthetic transcripts / forms / cases seeded from the real Aug 2024 call log.

python -m gen.generate [--n-agents 8 --calls-per-agent 10 --seed 42 --limit N --no-llm --force]
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from qa.llm import sampling_kwargs
from qa.rules import outcome_consistent, true_status

load_dotenv()

DATA = Path("data/calls.xlsx")
SYN = Path("data/synthetic")
HOLD = {"transfer_confirm_call": 420, "pa_plan_call": 900, "patient_access_check": 900}
PROFILE_ORDER = ["logs_without_connecting"] * 2 + ["ramping"] * 2 + ["curt", "phi_oversharer"] + ["solid"] * 2
AGENT_NAMES = ["Jo", "Ravi", "Mica", "Anjali", "Paolo", "Rhea", "Carlo", "Nina", "Sam", "Tess"]
REP_NAMES = ["Dana", "Marcus", "Keisha", "Tom", "Priya", "Luis", "Angela", "Brian", "Shonda", "Kevin",
             "Maria", "Derek", "Tanya", "Greg", "Yolanda", "Chris"]
FAKE_REP_NAMES = ["Sarah", "Mike", "Jennifer", "David", "Lisa", "Robert"]
FIRST = ["Maya", "James", "Olivia", "Daniel", "Grace", "Samuel", "Leah", "Victor", "Nora", "Ethan",
         "Ruth", "Isaac", "Clara", "Omar", "Hazel", "Felix", "Ivy", "Marcus", "June", "Theo"]
LAST = ["Okafor", "Bennett", "Nguyen", "Alvarez", "Kowalski", "Hart", "Iyer", "Brennan", "Castillo",
        "Moreau", "Sato", "Whitfield", "Adeyemi", "Lindqvist", "Ramos", "Fischer", "Mensah", "Park"]
STREETS = ["Maple Ave", "Oak St", "Cedar Ln", "Birch Rd", "Elm Ct", "Willow Dr", "Pine St"]
CITIES = ["Columbus, OH", "Tampa, FL", "Mesa, AZ", "Raleigh, NC", "Tulsa, OK", "Fresno, CA"]
PRESCRIBERS = ["Dr. Lena Ruiz", "Dr. Arjun Mehta", "Dr. Karen Holt", "Dr. Samuel Osei", "Dr. Emily Tran"]
SPEC_DRUGS = {"Dupixent 300mg/2mL pen": "atopic dermatitis", "Skyrizi 150mg/mL pen": "plaque psoriasis",
              "Rinvoq 15mg tablet": "rheumatoid arthritis", "Taltz 80mg/mL autoinjector": "psoriatic arthritis",
              "Humira 40mg/0.4mL pen": "Crohn's disease"}
RETAIL_DRUGS = {"atorvastatin 40mg tablet": "hyperlipidemia", "sertraline 50mg tablet": "major depressive disorder",
                "metformin 500mg tablet": "type 2 diabetes", "lisinopril 10mg tablet": "hypertension",
                "albuterol HFA 90mcg inhaler": "asthma"}
DENIAL_REASONS = ["step therapy not met — trial of a topical steroid required",
                  "insufficient clinical documentation", "non-formulary; formulary alternative available",
                  "quantity exceeds plan limit"]
DOCS = ["proof of income", "signed patient consent form", "copy of insurance card (front and back)",
        "prescriber statement of medical necessity"]
POSITIVE = {"transfer_confirm_call": "received", "pa_plan_call": "approved", "patient_access_check": "enrolled"}
HINTS = {
    "PHI_DISCLOSURE": "Early in the call, before the rep asks for any verification, the agent volunteers the "
                      "patient's full date of birth, home address and diagnosis.",
    "UNPROFESSIONAL": "The agent sighs audibly (write it as '*sighs*'), interrupts the rep twice mid-sentence, "
                      "and says 'this is the third time I'm calling, just check again'.",
    "MISSING_REF": "Near the end the rep offers a reference number; the agent says 'okay thanks' and does NOT "
                   "repeat, confirm or write it down.",
}


def missed_step_hint(call_type, status):
    if call_type == "pa_plan_call":
        return {"denied": "The rep says the PA was denied; the agent thanks them and ends the call without asking "
                          "for the denial reason, the appeal deadline or peer-to-peer availability.",
                "approved": "The rep says the PA was approved; the agent ends the call without asking for the "
                            "authorization number or effective dates.",
                "pending": "The rep says the PA is pending; the agent ends the call without asking when a "
                           "determination is expected.",
                }.get(status, "The rep says there is no PA on file; the agent ends the call without asking how or "
                              "where to submit one.")
    if call_type == "patient_access_check":
        return ("The rep gives the enrollment status; the agent ends the call without asking which documents are "
                "missing, the benefits-verification result, or whether a bridge supply can ship.")
    return ("The rep gives the receipt status; the agent ends the call without asking for the pharmacy Rx number "
            "or about any fill blockers (stock, PA, insurance reject).")


# ---------------------------------------------------------------- load + profile
def to_hour(v):
    if isinstance(v, dt.time):
        return v.hour
    if isinstance(v, dt.timedelta):
        return int(v.total_seconds() // 3600)
    return int(str(v).split(":")[0])


def load():
    calls = pd.read_excel(DATA, sheet_name="Call Data")
    rx = pd.read_excel(DATA, sheet_name="Prescriptions Transferred")
    calls.columns = [c.strip() for c in calls.columns]
    rx.columns = [c.strip() for c in rx.columns]
    calls["call_id"] = calls["call_id"].astype(str)
    calls["call_date"] = pd.to_datetime(calls["call_date"]).dt.date
    calls["hour"] = calls["hour_of_day"].map(to_hour)
    calls["et_hour"] = (calls["hour"] - 4) % 24
    calls["duration_seconds"] = pd.to_numeric(calls["duration_seconds"], errors="coerce").fillna(0).astype(int)
    calls["call_type"] = calls["reasons"]
    calls["hold_limit"] = calls["call_type"].map(HOLD).fillna(900).astype(int)
    rx["transferred_on"] = pd.to_datetime(rx["transferred_on"]).dt.date
    rx["ptype"] = rx["pharmacy type"].str.strip().str.lower()
    return calls, rx


def agent_stats(calls):
    g = calls.groupby("email")
    st = pd.DataFrame({
        "volume": g.size(),
        "first_date": g["call_date"].min(),
        "under20": g["duration_seconds"].apply(lambda s: (s < 20).mean()),
        "over_hold": g.apply(lambda d: (d["duration_seconds"] > d["hold_limit"]).mean(), include_groups=False),
    })
    st["tenured"] = st["first_date"] < dt.date(2024, 8, 20)
    return st


def pick_profiles(st):
    chosen = []
    ten = st[st.tenured]
    lwc = ten[ten.volume >= 200].sort_values("under20", ascending=False).index[:2].tolist()
    chosen += lwc
    ramp = st[~st.tenured].sort_values("volume", ascending=False).index[:2].tolist()
    chosen += ramp
    rest = ten.drop(index=lwc).sort_values("volume", ascending=False).index[:4].tolist()
    chosen += rest
    return chosen  # aligned with PROFILE_ORDER


# ---------------------------------------------------------------- sampling
def sample_agent(df, rng, n):
    df = df.sample(frac=1, random_state=int(rng.integers(1e9)))
    picked = []

    def take(mask, k):
        for cid in df[mask & ~df.call_id.isin(picked)].call_id[:k]:
            picked.append(cid)

    take(df.duration_seconds < 20, 2)
    take(df.duration_seconds > df.hold_limit, 2)
    take(df.et_hour >= 19, 1)
    for ct, q in (("pa_plan_call", 3), ("patient_access_check", 2), ("transfer_confirm_call", 5)):
        have = df[df.call_id.isin(picked) & (df.call_type == ct)].shape[0]
        take(df.call_type == ct, max(0, min(q - have, n - len(picked))))
    take(df.call_id.notna(), n - len(picked))
    out = df[df.call_id.isin(picked[:n])].copy()
    order = {c: i for i, c in enumerate(picked)}
    return out.sort_values("call_id", key=lambda s: s.map(order))


# ---------------------------------------------------------------- cases
def fake_patient(rng):
    f, l = rng.choice(FIRST), rng.choice(LAST)
    dob = dt.date(1950, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365 * 50)))
    return {"first": str(f), "last": str(l), "dob": dob.isoformat(), "initials": f"{f[0]}.{l[0]}.",
            "address": f"{int(rng.integers(100, 9999))} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
            "member_id": f"W{int(rng.integers(10**8, 10**9))}"}


def d(x):
    return x.isoformat() if x else None


def make_case(row, rx, rng, used_erx):
    cdate = row.call_date
    ct = row.call_type
    base = {"case_type": ct, "patient": fake_patient(rng), "prescriber": f"{rng.choice(PRESCRIBERS)}, NPI {int(rng.integers(10**9, 2*10**9))}"}
    if ct == "transfer_confirm_call":
        ptype = "specialty" if str(row.related_to_specialty_transfer).strip() in ("1", "1.0") else "retail"
        pool = rx[(rx.ptype == ptype) & (~rx.erx_id.isin(used_erx))]
        win = pool[(pool.transferred_on <= cdate) & (pool.transferred_on >= cdate - dt.timedelta(days=7))]
        r = (win if len(win) else pool).sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        erx = int(r.erx_id)
        used_erx.add(erx)
        t_on = r.transferred_on if len(win) else cdate - dt.timedelta(days=int(rng.integers(0, 8)))
        drugs = SPEC_DRUGS if ptype == "specialty" else RETAIL_DRUGS
        drug = str(rng.choice(list(drugs)))
        resolved = bool(rng.random() < 0.10)
        days_since = (cdate - t_on).days
        state = "filled" if resolved else str(rng.choice(["received", "not_received", "needs_pa", "filled"],
                                                          p=[0.35, 0.3, 0.2, 0.15]))
        recv = filled = None
        if state != "not_received":
            recv = t_on + dt.timedelta(days=int(rng.integers(0, max(1, days_since) + 1)))
            recv = min(recv, cdate)
        else:
            recv = None if rng.random() < 0.6 else cdate + dt.timedelta(days=int(rng.integers(1, 4)))
        if state == "filled":
            filled = recv if resolved else min(cdate, recv + dt.timedelta(days=int(rng.integers(0, 2))))
            if resolved:
                recv = min(recv, cdate - dt.timedelta(days=1))
                filled = recv
        blocker = {"needs_pa": "needs_pa", "received": str(rng.choice(["none", "out_of_stock", "insurance_reject"])),
                   }.get(state)
        if state == "received" and ptype == "specialty" and rng.random() < 0.3:
            blocker = "needs_pa"
        return {"case_ref": f"RX-{erx}", **base, "erx_id": erx, "pharmacy_type": ptype, "transferred_on": d(t_on),
                "target_pharmacy": row.destination_name, "drug": drug, "diagnosis": drugs[drug],
                "resolved_before_call": resolved,
                "truth": {"received_at": d(recv), "filled_at": d(filled), "blocker": blocker,
                          "pharmacy_rx_number": f"RX{int(rng.integers(10**5, 10**6))}" if recv and recv <= cdate else None,
                          "fill_eta": d(cdate + dt.timedelta(days=int(rng.integers(1, 4)))) if state == "received" else None}}
    drugs = SPEC_DRUGS
    drug = str(rng.choice(list(drugs)))
    if ct == "pa_plan_call":
        sub = cdate - dt.timedelta(days=int(rng.integers(3, 15)))
        status = str(rng.choice(["approved", "denied", "pending", "not_on_file"], p=[0.35, 0.25, 0.3, 0.1]))
        det = sub + dt.timedelta(days=int(rng.integers(1, max(2, (cdate - sub).days)))) if status in ("approved", "denied") else None
        return {"case_ref": f"PA-{int(rng.integers(1000, 9999))}", **base, "plan": row.destination_name,
                "drug": drug, "diagnosis": drugs[drug], "submitted_on": d(sub),
                "truth": {"status": status, "determination_date": d(det),
                          "expected_determination": d(cdate + dt.timedelta(days=int(rng.integers(2, 8)))) if status == "pending" else None,
                          "auth_number": f"AUTH{int(rng.integers(10**6, 10**7))}" if status == "approved" else None,
                          "effective": f"{d(det)} to {d(det + dt.timedelta(days=365))}" if status == "approved" else None,
                          "denial_reason": str(rng.choice(DENIAL_REASONS)) if status == "denied" else None,
                          "appeal_deadline": d(det + dt.timedelta(days=60)) if status == "denied" else None,
                          "peer_to_peer": "available within 14 days of denial" if status == "denied" else None}}
    status = str(rng.choice(["enrolled", "pending_docs", "pending_bv", "denied"], p=[0.35, 0.3, 0.2, 0.15]))
    return {"case_ref": f"ENR-{int(rng.integers(1000, 9999))}", **base, "program": row.destination_name,
            "drug": drug, "diagnosis": drugs[drug],
            "truth": {"status": status,
                      "missing_docs": list(rng.choice(DOCS, size=int(rng.integers(1, 3)), replace=False)) if status == "pending_docs" else [],
                      "bv_result": {"enrolled": "commercial coverage verified, $5 copay with copay card",
                                    "pending_bv": "in progress", "denied": "patient not eligible — government insurance",
                                    }.get(status, "not started — waiting on documents"),
                      "bridge_ship_date": d(cdate + dt.timedelta(days=int(rng.integers(1, 5)))) if status in ("enrolled", "pending_bv") else None}}


# ---------------------------------------------------------------- defects
def plant(sample, profiles, cases, rng):
    labels = []  # dicts call_id, code, severity, planted_by
    sev = {"FAB_CONTACT": "critical", "STATUS_MISMATCH": "critical", "OUTCOME_VS_RX": "critical",
           "PHI_DISCLOSURE": "critical", "UNPROFESSIONAL": "major", "MISSED_NEXT_STEP": "major",
           "MISSING_REF": "major", "HOLD_OVERRUN": "minor", "REDUNDANT_CALL": "minor", "AFTER_HOURS": "minor"}

    def add(cid, code, by):
        labels.append({"call_id": cid, "code": code, "severity": sev[code], "planted_by": by})

    for r in sample.itertuples():
        p = profiles[r.agent_id]
        pa_prog = r.call_type != "transfer_confirm_call"
        if p == "logs_without_connecting" and not r.connected and rng.random() < 0.7:
            add(r.call_id, "FAB_CONTACT", "profile")
        if not r.connected:
            continue
        if p == "ramping" and pa_prog:
            if rng.random() < 0.4:
                add(r.call_id, "MISSED_NEXT_STEP", "profile")
            if rng.random() < 0.3:
                add(r.call_id, "MISSING_REF", "profile")
        if p == "curt" and rng.random() < 0.4:
            add(r.call_id, "UNPROFESSIONAL", "profile")
        if p == "phi_oversharer" and rng.random() < 0.4:
            add(r.call_id, "PHI_DISCLOSURE", "profile")
        if p == "solid" and rng.random() < 0.08:
            opts = ["UNPROFESSIONAL", "MISSED_NEXT_STEP"] + (["MISSING_REF"] if pa_prog else [])
            add(r.call_id, str(rng.choice(opts)), "profile")
    # floor: >= 2 labels per profile-driven code so every judge code is measurable (top up on eligible calls)
    floors = [("FAB_CONTACT", "logs_without_connecting", lambda r: not r.connected),
              ("MISSED_NEXT_STEP", "ramping", lambda r: r.connected and r.call_type != "transfer_confirm_call"),
              ("MISSING_REF", "ramping", lambda r: r.connected and r.call_type != "transfer_confirm_call"),
              ("UNPROFESSIONAL", "curt", lambda r: r.connected),
              ("PHI_DISCLOSURE", "phi_oversharer", lambda r: r.connected)]
    for code, prof, ok in floors:
        have = {l["call_id"] for l in labels if l["code"] == code}
        elig = [r.call_id for r in sample.itertuples() if profiles[r.agent_id] == prof and ok(r) and r.call_id not in have]
        rng.shuffle(elig)
        for cid in elig[:max(0, 2 - len(have))]:
            add(cid, code, "profile")
    # global STATUS_MISMATCH on exactly 4 connected calls
    conn = sample[sample.connected]
    w = np.where(conn.agent_id.map(profiles) == "solid", 1.0, 3.0)
    for cid in rng.choice(conn.call_id.values, size=min(4, len(conn)), replace=False, p=w / w.sum()):
        add(cid, "STATUS_MISMATCH", "global")
    # derived (form-independent)
    seen = set()
    for r in sample.sort_values(["call_date", "hour", "call_id"]).itertuples():
        if r.duration_seconds > r.hold_limit + 60:
            add(r.call_id, "HOLD_OVERRUN", "derived")
        if r.et_hour >= 19 or r.et_hour < 8:
            add(r.call_id, "AFTER_HOURS", "derived")
        key = (r.case_ref, r.call_date)
        if cases[r.case_ref].get("resolved_before_call") or key in seen:
            add(r.call_id, "REDUNDANT_CALL", "derived")
        seen.add(key)
    # guarantee >= 25% zero-label calls by dropping profile-planted labels from random calls
    lab = pd.DataFrame(labels)
    need = int(np.ceil(0.25 * len(sample)))
    clean = set(sample.call_id) - set(lab.call_id)
    if len(clean) < need:
        by_call = lab.groupby("call_id")["planted_by"].apply(set)
        droppable = [c for c, s in by_call.items() if s == {"profile"}]
        rng.shuffle(droppable)
        drop = set(droppable[:need - len(clean)])
        lab = lab[~lab.call_id.isin(drop)]
    return lab


# ---------------------------------------------------------------- scenario + form
def budget(dur, connected, rng):
    if not connected:
        return {"ivr_s": dur, "hold_s": 0, "talk_s": 0}
    ivr = int(min(rng.integers(20, 61), dur * 0.3))
    talk = int(min(rng.integers(90, 241), dur - ivr))
    return {"ivr_s": ivr, "hold_s": dur - ivr - talk, "talk_s": talk}


def rep_says(case, cdate):
    t, ct = case["truth"], case["case_type"]
    s = true_status(case, cdate)
    if ct == "transfer_confirm_call":
        if s == "not_received":
            return {"status": s, "say": "No record of this prescription being received. Offer to take the prescriber's NPI and fax number so it can be re-sent."}
        out = {"status": s, "received_on": t["received_at"], "pharmacy_rx_number": t["pharmacy_rx_number"]}
        if s == "filled":
            out["say"] = f"Received on {t['received_at']} and already filled on {t['filled_at']}."
        elif t["blocker"] == "needs_pa":
            out["say"] = f"Received on {t['received_at']} but the claim rejected — needs prior authorization before it can be filled."
        elif t["blocker"] == "out_of_stock":
            out["say"] = f"Received on {t['received_at']}; drug is back-ordered, expected fill {t['fill_eta']}."
        elif t["blocker"] == "insurance_reject":
            out["say"] = f"Received on {t['received_at']}; insurance rejected the claim (refill too soon / plan limitation), working on it, fill ETA {t['fill_eta']}."
        else:
            out["say"] = f"Received on {t['received_at']}, in queue, expected fill {t['fill_eta']}."
        return out
    return {k: v for k, v in t.items() if v not in (None, [], "")} | {"status": s}


def make_form(r, case, scen, codes, rng):
    ct, s = r.call_type, scen["rep_truth"]["status"]
    f = {"call_id": r.call_id, "agent_id": r.agent_id, "case_ref": r.case_ref, "call_type": ct,
         "spoke_with_rep": bool(r.connected), "rep_name": scen.get("rep_name"),
         "reference_number": scen.get("reference_number"), "outcome_status": s if r.connected else "no_answer"}
    if "FAB_CONTACT" in codes:
        f.update(spoke_with_rep=True, rep_name=str(rng.choice(FAKE_REP_NAMES)),
                 reference_number=f"REF-{int(rng.integers(10**5, 10**6))}", outcome_status=POSITIVE[ct])
    elif not r.connected:
        f.update(spoke_with_rep=False, rep_name=None, reference_number=None, outcome_status="no_answer")
    if "STATUS_MISMATCH" in codes:
        opts = {"transfer_confirm_call": ["received", "not_received", "filled", "needs_pa"],
                "pa_plan_call": ["approved", "denied", "pending", "not_on_file"],
                "patient_access_check": ["enrolled", "pending_docs", "pending_bv", "denied"]}[ct]
        # prefer the "optimistic" flip
        flips = [o for o in opts if o != s and not outcome_consistent(case, o, r.call_date)]
        f["outcome_status"] = POSITIVE[ct] if POSITIVE[ct] in flips else str(rng.choice(flips))
    if "MISSING_REF" in codes:
        f["reference_number"] = None
    o = f["outcome_status"]
    nxt = {"no_answer": ("Retry call next business day", 1), "received": ("Confirm fill", 2),
           "not_received": ("Re-send Rx from prescriber office", 1), "filled": ("Close task", None),
           "needs_pa": ("Start PA with plan", 1), "approved": ("Notify pharmacy of approval", 1),
           "denied": ("Prepare appeal with prescriber", 2), "pending": ("Follow up on PA determination", 3),
           "not_on_file": ("Submit PA", 1), "enrolled": ("Confirm bridge shipment", 3),
           "pending_docs": ("Collect missing documents", 2), "pending_bv": ("Follow up on BV", 3)}[o]
    f["next_action"] = nxt[0]
    f["next_follow_up_date"] = (r.call_date + dt.timedelta(days=nxt[1])).isoformat() if nxt[1] else None
    if o == "no_answer":
        f["notes"] = "IVR / no answer"
    else:
        f["notes"] = f"Spoke with {f['rep_name']}. Status: {o.replace('_', ' ')}." + (
            f" Ref {f['reference_number']}." if f["reference_number"] else "")
    return f


def scenario(r, case, codes, agent_name, rng):
    b = budget(r.duration_seconds, r.connected, rng)
    sc = {"call_id": r.call_id, "call_type": r.call_type, "destination_name": r.destination_name,
          "duration_seconds": int(r.duration_seconds), "connected": bool(r.connected), **b,
          "agent_name": agent_name, "case": {k: v for k, v in case.items() if k not in ("truth", "resolved_before_call")},
          "rep_truth": rep_says(case, r.call_date), "planted": sorted(codes), "hints": []}
    if r.connected:
        sc["rep_name"] = str(rng.choice(REP_NAMES))
        if r.call_type != "transfer_confirm_call":
            sc["reference_number"] = f"REF-{int(rng.integers(10**5, 10**6))}"
        for c in ("PHI_DISCLOSURE", "UNPROFESSIONAL", "MISSING_REF"):
            if c in codes:
                sc["hints"].append(HINTS[c])
        if "MISSED_NEXT_STEP" in codes:
            sc["hints"].append(missed_step_hint(r.call_type, sc["rep_truth"]["status"]))
    return sc


# ---------------------------------------------------------------- transcript render
SYSTEM = """You write realistic US healthcare phone-call transcripts for training data.
The AGENT is an offshore BPO caller working for Forus, calling on behalf of a prescriber's office.
The REP is an employee of the insurance plan, manufacturer patient-support hub, or pharmacy being called.
IVR lines must be realistic for the destination (menus, "your call may be recorded", hold messages).
Keep it clean, professional-sounding and non-cartoonish. Roughly one turn per 8-10 seconds of talk time.
Speakers are exactly AGENT, REP or IVR. t_sec is seconds from dial, non-decreasing, and never above the
call duration. Never name, label or allude to any quality problem or defect; just render what happens.
If the call is not connected, produce ONLY IVR / ringing lines (speaker IVR, at most 4 turns, no REP, no AGENT
speech other than possibly nothing) - e.g. ringing, closed-hours message, voicemail box full.
Submit via the submit_transcript tool."""

TOOL = {"name": "submit_transcript", "description": "Submit the rendered transcript.",
        "input_schema": {"type": "object", "required": ["turns"], "properties": {"turns": {"type": "array", "items": {
            "type": "object", "required": ["t_sec", "speaker", "text"],
            "properties": {"t_sec": {"type": "integer"}, "speaker": {"type": "string", "enum": ["AGENT", "REP", "IVR"]},
                           "text": {"type": "string"}}}}}}}


def user_prompt(sc):
    card = {k: v for k, v in sc.items() if k not in ("planted", "call_id")}
    lines = [f"Scenario card:\n{json.dumps(card, indent=1, default=str)}", ""]
    if sc["connected"]:
        rep_t = sc["ivr_s"] + sc["hold_s"]
        lines += [f"Timeline: IVR/menus from 0-{sc['ivr_s']}s, then hold (one IVR hold line), REP picks up at ~{rep_t}s, "
                  f"conversation until ~{sc['duration_seconds']}s (~{max(4, sc['talk_s'] // 9)} turns of talk).",
                  f"The REP introduces themself by first name '{sc['rep_name']}'. The agent introduces themself as "
                  f"{sc['agent_name']} from Forus, calling on behalf of the prescriber's office.",
                  "The REP verifies the patient (asks for name + DOB) before giving details. The REP states the facts in "
                  "rep_truth accurately (use the exact dates / numbers given)."]
        if sc.get("reference_number"):
            lines.append(f"Near the end the REP gives reference number {sc['reference_number']} (say it exactly).")
        lines.append("Agent behaviour notes: " + (" ".join(sc["hints"]) if sc["hints"] else
                     "competent and courteous; verifies, asks the relevant follow-ups, reads the outcome back."))
    else:
        lines.append(f"NOT connected. Total duration {sc['duration_seconds']}s. Only IVR/ringing turns, max 4.")
    return "\n".join(lines)


def validate(sc, turns):
    if not turns:
        return "empty"
    last = -1
    for t in turns:
        if t.get("speaker") not in ("AGENT", "REP", "IVR"):
            return "bad speaker"
        ts = t.get("t_sec")
        if not isinstance(ts, int) or ts < last or ts > sc["duration_seconds"]:
            return f"bad t_sec {ts}"
        last = ts
    if not sc["connected"]:
        if any(t["speaker"] == "REP" for t in turns) or len(turns) > 4:
            return "rep/too many turns on non-connected call"
    else:
        text = " ".join(t["text"] for t in turns if t["speaker"] == "REP")
        if sc["rep_name"] not in text:
            return "rep name missing"
        if sc.get("reference_number") and sc["reference_number"] not in text:
            return "reference number missing"
    return None


def template(sc):
    dur = sc["duration_seconds"]
    if not sc["connected"]:
        turns = [(0, "IVR", "[ringing]")]
        if dur >= 8:
            turns.append((min(dur, 8), "IVR", f"Thank you for calling {sc['destination_name']}. Our office is currently closed. Please call back during normal business hours."))
        return [{"i": i, "t_sec": t, "speaker": s, "text": x} for i, (t, s, x) in enumerate(turns)]
    rt, c, p = sc["rep_truth"], sc["case"], sc["case"]["patient"]
    t0 = sc["ivr_s"] + sc["hold_s"]
    step = max(3, sc["talk_s"] // 12)
    who = sc["agent_name"]
    prescriber = c["prescriber"].split(",")[0]
    hints = " ".join(sc["hints"])
    L = [("IVR", f"Thank you for calling {sc['destination_name']}. This call may be recorded. Please hold for the next available representative.")]
    L.append(("REP", f"Thank you for holding, this is {sc['rep_name']}, how can I help you?"))
    if "full date of birth" in hints:
        L.append(("AGENT", f"Hi, this is {who} from Forus on behalf of {prescriber}'s office. Patient is {p['first']} {p['last']}, date of birth {p['dob']}, lives at {p['address']}, diagnosed with {c['diagnosis']}."))
    else:
        L.append(("AGENT", f"Hi {sc['rep_name']}, this is {who} calling from Forus on behalf of {prescriber}'s office about a patient."))
    L.append(("REP", "Sure, can I have the patient's name and date of birth?"))
    L.append(("AGENT", f"{p['first']} {p['last']}, {p['dob']}."))
    if "sighs" in hints:
        L.append(("AGENT", "*sighs* This is the third time I'm calling, just check again."))
    L.append(("REP", f"Thank you. I see it. {rt.get('say') or ('Status is ' + rt['status'].replace('_', ' ') + '.')} " +
              " ".join(f"{k.replace('_', ' ')}: {v}." for k, v in rt.items() if k not in ('status', 'say'))))
    if "without asking" not in hints:
        L.append(("AGENT", f"Just to confirm, the status is {rt['status'].replace('_', ' ')}. Anything else needed from the office?"))
        L.append(("REP", "No, that's everything."))
    if sc.get("reference_number"):
        L.append(("REP", f"Your reference number for this call is {sc['reference_number']}."))
        L.append(("AGENT", "Okay thanks." if "does NOT" in hints else f"That's {sc['reference_number']}, thank you."))
    L.append(("AGENT", "Thanks for your help, have a good day."))
    out = [{"i": 0, "t_sec": 0, "speaker": "IVR", "text": L[0][1]}]
    for i, (s, x) in enumerate(L[1:], 1):
        out.append({"i": i, "t_sec": min(dur, t0 + (i - 1) * step), "speaker": s, "text": x})
    return out


async def render_one(client, sem, sc, model):
    for attempt in range(2):
        for k in range(3):
            try:
                async with sem:
                    r = await client.messages.create(
                        model=model, max_tokens=4000, **sampling_kwargs(model, 0.8), system=SYSTEM, tools=[TOOL],
                        tool_choice={"type": "tool", "name": "submit_transcript"},
                        messages=[{"role": "user", "content": user_prompt(sc)}])
                break
            except Exception as e:  # noqa
                if k == 2:
                    print(f"  API error {sc['call_id']}: {e}")
                    return template(sc), "template(api)"
                await asyncio.sleep(2 ** (k + 1))
        turns = next(b.input for b in r.content if b.type == "tool_use").get("turns", [])
        err = validate(sc, turns)
        if not err:
            return [{"i": i, **t} for i, t in enumerate(turns)], "llm"
        print(f"  validation fail {sc['call_id']} (try {attempt + 1}): {err}")
    return template(sc), "template(invalid)"


async def render_all(scen, limit, no_llm, force):
    path = SYN / "transcripts.jsonl"
    done = {}
    if path.exists() and not force:
        for line in path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done[o["call_id"]] = o
    todo = [s for s in scen if s["call_id"] not in done]
    if limit:
        todo = todo[:limit]
    src = {"cached": len(done)}
    mode = "w" if force else "a"
    with open(path, mode) as fh:
        if no_llm:
            for s in todo:
                fh.write(json.dumps({"call_id": s["call_id"], "turns": template(s), "source": "template"}) + "\n")
            src["template"] = len(todo)
            return src
        from anthropic import AsyncAnthropic
        client, sem, model = AsyncAnthropic(), asyncio.Semaphore(8), os.environ["GEN_MODEL"]

        async def job(s):
            turns, how = await render_one(client, sem, s, model)
            fh.write(json.dumps({"call_id": s["call_id"], "turns": turns, "source": how}) + "\n")
            fh.flush()
            src[how] = src.get(how, 0) + 1
            print(f"  rendered {s['call_id']} [{how}] {len(turns)} turns", flush=True)

        await asyncio.gather(*(job(s) for s in todo))
    return src


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--calls-per-agent", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    SYN.mkdir(parents=True, exist_ok=True)

    calls, rx = load()
    st = agent_stats(calls)
    emails = pick_profiles(st)[: a.n_agents]
    agent_ids = [f"AGT-{i + 1:02d}" for i in range(len(emails))]
    email2id = dict(zip(emails, agent_ids))
    profiles = dict(zip(agent_ids, PROFILE_ORDER))
    prof_df = pd.DataFrame({"agent_id": agent_ids, "profile": PROFILE_ORDER[: len(emails)],
                            "volume": st.loc[emails, "volume"].values,
                            "first_date": st.loc[emails, "first_date"].values,
                            "under20_share": st.loc[emails, "under20"].round(3).values,
                            "over_hold_share": st.loc[emails, "over_hold"].round(3).values})
    prof_df.to_csv(SYN / "agent_profiles.csv", index=False)

    parts = []
    for e in emails:
        s = sample_agent(calls[calls.email == e], rng, a.calls_per_agent)
        s["agent_id"] = email2id[e]
        parts.append(s)
    sample = pd.concat(parts).drop(columns=["email"]).reset_index(drop=True)
    sample["connected"] = sample.duration_seconds >= 30

    # cases (shared case_ref for same agent + destination + date)
    cases, keymap, used = {}, {}, set()
    refs = []
    for r in sample.itertuples():
        key = (r.agent_id, r.destination_name, r.call_date, r.call_type)
        if key not in keymap:
            c = make_case(r, rx, rng, used)
            while c["case_ref"] in cases:
                c["case_ref"] = c["case_ref"] + "B"
            cases[c["case_ref"]] = c
            keymap[key] = c["case_ref"]
        refs.append(keymap[key])
    sample["case_ref"] = refs

    labels = plant(sample, profiles, cases, rng)
    codes_by = labels.groupby("call_id")["code"].apply(set).to_dict()

    scen, forms = [], []
    name_of = {aid: AGENT_NAMES[i % len(AGENT_NAMES)] for i, aid in enumerate(agent_ids)}
    for r in sample.itertuples():
        codes = codes_by.get(r.call_id, set())
        sc = scenario(r, cases[r.case_ref], codes, name_of[r.agent_id], rng)
        scen.append(sc)
        forms.append(make_form(r, cases[r.case_ref], sc, codes, rng))
    # derived OUTCOME_VS_RX from finished forms
    for f, r in zip(forms, sample.itertuples()):
        if not outcome_consistent(cases[r.case_ref], f["outcome_status"], r.call_date):
            labels = pd.concat([labels, pd.DataFrame([{"call_id": r.call_id, "code": "OUTCOME_VS_RX",
                                                       "severity": "critical", "planted_by": "derived"}])])

    out_cols = [c for c in sample.columns if c not in ("hour",)]
    sample[out_cols].to_csv(SYN / "calls_sample.csv", index=False)
    with open(SYN / "cases.jsonl", "w") as fh:
        for c in cases.values():
            fh.write(json.dumps(c, default=str) + "\n")
    with open(SYN / "scenarios.jsonl", "w") as fh:
        for s in scen:
            fh.write(json.dumps(s, default=str) + "\n")
    with open(SYN / "forms.jsonl", "w") as fh:
        for f in forms:
            fh.write(json.dumps(f, default=str) + "\n")
    labels.to_csv(SYN / "labels.csv", index=False)

    print(f"agents: {dict(zip(agent_ids, PROFILE_ORDER))}")
    print(f"calls sampled: {len(sample)}  connected: {int(sample.connected.sum())}  cases: {len(cases)}  "
          f"types: {sample.call_type.value_counts().to_dict()}")
    print(f"zero-label calls: {len(set(sample.call_id) - set(labels.call_id))} / {len(sample)}")
    print("labels by code:\n" + labels.groupby(["code", "planted_by"]).size().to_string())
    print("labels by agent:\n" + labels.merge(sample[["call_id", "agent_id"]]).pivot_table(
        index="agent_id", columns="code", values="call_id", aggfunc="count", fill_value=0).to_string())

    src = asyncio.run(render_all(scen, a.limit, a.no_llm, a.force))
    print(f"transcripts: {src}")


if __name__ == "__main__":
    main()
