import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SYN, OUT = Path(os.environ.get("SYN_DIR", "data/synthetic")), Path(os.environ.get("OUT_DIR", "out"))
# Held-out set C (seed 11): generated after the prompt was frozen and judged exactly once. Set B (out/holdout, seed 7)
# was used to tune the MISSED_NEXT_STEP rules between prompt v4 and v6, so it is a dev set, not a test set.
HOLD_OUT = Path(os.environ.get("HOLDOUT_DIR", "out/holdout_c" if Path("out/holdout_c/validation_summary.json").exists() else "out/holdout"))
SEV_COLOR = {"critical": "#d62728", "major": "#f0a202", "minor": "#9e9e9e"}
FIELDS = ["spoke_with_rep", "rep_name", "reference_number", "outcome_status", "next_action"]

st.set_page_config(page_title="Call Quality Copilot", layout="wide")


def mtime(p):
    p = Path(p)
    return p.stat().st_mtime if p.exists() else 0


@st.cache_data
def read_jsonl(path, mtime_key):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@st.cache_data
def read_csv(path, mtime_key):
    p = Path(path)
    return pd.read_csv(p, dtype={"call_id": str}) if p.exists() else pd.DataFrame()


def jl(name, base=OUT):
    return read_jsonl(str(base / name), mtime(base / name))


def csv(name, base=OUT):
    return read_csv(str(base / name), mtime(base / name))


results = {r["call_id"]: r for r in jl("results.jsonl")}
scored = csv("scored_calls.csv")
sample = csv("calls_sample.csv", SYN)
forms = {f["call_id"]: f for f in jl("forms.jsonl", SYN)}
cases = {c["case_ref"]: c for c in jl("cases.jsonl", SYN)}
tx = {t["call_id"]: t["turns"] for t in jl("transcripts.jsonl", SYN)}
summary = json.loads((OUT / "validation_summary.json").read_text()) if (OUT / "validation_summary.json").exists() else {}

st.title("Call Quality Copilot")
st.warning("**SYNTHETIC** transcripts and forms, seeded from the real Aug 2024 TaskUs call log (real durations, "
           "destinations, Rx IDs; agents pseudonymised). Agent profiles are illustrative.")

if scored.empty:
    st.error("No scored calls yet — run `make demo`.")
    st.stop()

tab_ov, tab_insp, tab_val = st.tabs(["Overview", "Call Inspector", "Validation"])


def pct(x):
    return "—" if x is None or x != x else f"{x:.0%}"


# ---------------------------------------------------------------- Overview
with tab_ov:
    c = st.columns(5)
    c[0].metric("Calls scored", f"{len(scored)}", help=f"{int(scored.judged.sum())} judged by LLM")
    c[1].metric("% calls with a critical", pct((scored.n_critical > 0).mean()))
    c[2].metric("Mean score", f"{scored.score.mean():.1f}")
    c[3].metric("Form accuracy", pct(scored.form_accuracy.mean()),
                help="share of verifiable form fields the transcript supports; fields the transcript cannot verify "
                     "(e.g. outcome on a no-answer call) are excluded")
    c[4].metric("Evidence validity", pct(summary.get("evidence_validity")), help="judge quotes found verbatim in the transcript")
    md = pd.DataFrame([{**m, "call_id": r["call_id"], "call_type": r["call_type"]}
                       for r in results.values() for m in r["merged_defects"]])
    l, r = st.columns([2, 1])
    if not md.empty:
        cnt = md.groupby(["code", "severity"]).size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(cnt, x="code", y="count", color="severity", color_discrete_map=SEV_COLOR,
                     title="Defects by code", category_orders={"code": cnt.code.tolist()})
        l.plotly_chart(fig, use_container_width=True)
        src = md.groupby(["code", "source"]).size().reset_index(name="n")
    crit = scored.assign(crit=scored.n_critical > 0).groupby("call_type").crit.mean().reset_index()
    fig2 = px.bar(crit, x="call_type", y="crit", title="Critical rate by call type", color_discrete_sequence=["#d62728"])
    fig2.update_yaxes(tickformat=".0%", title=None)
    r.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Layer agreement (rules ∩ judge on FAB_CONTACT / MISSING_REF): {pct(summary.get('layer_agreement'))} · "
               f"calls needing human review: {int(scored.needs_human_review.sum())}")

# ---------------------------------------------------------------- Inspector
with tab_insp:
    s = scored.sort_values(["score", "call_id"])
    labels = {row.call_id: f"{row.score} · {row.agent_id} · {row.call_type} · "
                           f"{row.destination_name if isinstance(row.destination_name, str) else 'Unknown destination'} · "
                           f"{row.duration_seconds}s · {row.codes if isinstance(row.codes, str) else '—'}"
              for row in s.itertuples()}
    ids = list(labels)
    default = ids.index(st.session_state["selected_call"]) if st.session_state.get("selected_call") in ids else 0
    cid = st.selectbox("Call (lowest score first)", ids, index=default, format_func=labels.get)
    res, call = results[cid], sample[sample.call_id == cid].iloc[0]
    form, case = forms[cid], cases[call.case_ref]
    turns = tx.get(cid, [])

    left, right = st.columns([1, 1])
    with left:
        chips = [("agent", call.agent_id), ("type", call.call_type), ("destination", call.destination_name),
                 ("duration", f"{call.duration_seconds}s"), ("ET", f"{int(call.et_hour)}:00"),
                 ("connected", "yes" if call.connected else "no"), ("case", call.case_ref)]
        st.markdown(" ".join(f"<span style='background:#eef;border-radius:10px;padding:2px 8px;margin:2px;"
                             f"display:inline-block;font-size:0.85em'><b>{k}</b> {html.escape(str(v))}</span>"
                             for k, v in chips), unsafe_allow_html=True)
        cited = {}
        for m in res["merged_defects"]:
            if m.get("turn") is not None:
                cited.setdefault(m["turn"], []).append(m)
        for chk in res.get("checklist", []):
            ev = chk.get("evidence") or {}
            if chk["result"] == "fail" and ev.get("turn") is not None and ev["turn"] not in cited:
                cited.setdefault(ev["turn"], [])
        bubbles = []
        for t in turns:
            text = html.escape(t["text"])
            tags = "".join(f"<span style='background:{SEV_COLOR[m['severity']]};color:white;border-radius:4px;"
                           f"padding:0 4px;font-size:0.7em;margin-left:4px'>{m['code']}</span>" for m in cited.get(t["i"], []))
            border = ""
            if t["i"] in cited and cited[t["i"]]:
                border = f"border-left:5px solid {SEV_COLOR[cited[t['i']][0]['severity']]};"
            if t["speaker"] == "IVR":
                style = "background:#f3f3f3;color:#777;font-style:italic;margin-right:20%"
            elif t["speaker"] == "AGENT":
                style = "background:#dcf0ff;margin-left:20%"
            else:
                style = "background:#f0f0f0;margin-right:20%"
            bubbles.append(f"<div style='{style};{border}border-radius:8px;padding:6px 10px;margin:4px 0'>"
                           f"<div style='font-size:0.7em;color:#888'>[{t['i']}] {t['speaker']} · {t['t_sec']}s{tags}</div>{text}</div>")
        st.markdown(f"<div style='max-height:720px;overflow-y:auto'>{''.join(bubbles) or '<i>no transcript</i>'}</div>",
                    unsafe_allow_html=True)

    with right:
        st.subheader("Three-way diff")
        fc = {f["field"]: f for f in res.get("form_check", [])}
        t = case["truth"]
        if case["case_type"] == "transfer_confirm_call":
            from qa.rules import true_status
            truth_status = true_status(case, call.call_date)
            truth_extra = f"received {t['received_at'] or '—'}, filled {t['filled_at'] or '—'}, blocker {t['blocker'] or '—'}"
        else:
            truth_status = t["status"]
            truth_extra = ", ".join(f"{k} {v}" for k, v in t.items() if v and k != "status")
        truth_col = {"spoke_with_rep": "connected" if call.connected else f"not connected ({call.duration_seconds}s)",
                     "rep_name": "—", "reference_number": "—", "outcome_status": truth_status, "next_action": truth_extra}
        rows = []
        for f in FIELDS:
            chk = fc.get(f, {})
            tv = chk.get("transcript_value")
            rows.append({"field": f, "Agent logged (form)": str(form.get(f)),
                         "Transcript supports (auto-draft)": "—" if tv in (None, "", "null") else tv,
                         "System of record (case truth)": truth_col[f], "judge": chk.get("match", "—")})
        diff = pd.DataFrame(rows).set_index("field")
        rule_codes = {m["code"] for m in res["merged_defects"]}

        def hl(row):
            bad = row["judge"] == "mismatch" or (row.name == "outcome_status" and "OUTCOME_VS_RX" in rule_codes) or \
                  (row.name == "spoke_with_rep" and "FAB_CONTACT" in rule_codes)
            return ["background-color:#ffd6d6" if bad else ""] * len(row)
        st.dataframe(diff.style.apply(hl, axis=1), use_container_width=True)
        filled = sum(1 for f in FIELDS if fc.get(f, {}).get("transcript_value") not in (None, "", "null"))
        st.caption(f"Middle column = the judge's extraction from the transcript, effectively an auto-drafted form: "
                   f"{filled}/{len(FIELDS)} fields pre-fillable from this call.")

        st.subheader(f"Defects · score {res['score']}")
        if not res["merged_defects"]:
            st.success("No defects.")
        for m in sorted(res["merged_defects"], key=lambda m: ["critical", "major", "minor"].index(m["severity"])):
            q = f"“{html.escape(m['quote'])}”" if m.get("quote") else html.escape(m.get("reason") or "")
            kind_tag = ("<span style='background:#8a8a8a;color:white;border-radius:4px;padding:1px 6px' "
                        "title='dialer / workflow flag from call metadata, not agent conduct'>process</span> "
                        if m.get("kind") == "process" else "")
            st.markdown(f"<span style='background:{SEV_COLOR[m['severity']]};color:white;border-radius:4px;padding:1px 6px'>"
                        f"{m['severity']}</span> <span style='background:#555;color:white;border-radius:4px;padding:1px 6px'>"
                        f"{m['source']}</span> {kind_tag}**{m['code']}** · conf {m.get('confidence', 1):.2f}<br>"
                        f"<span style='font-size:0.9em'>{q}</span>", unsafe_allow_html=True)
        if res.get("agent_score") is not None and res["agent_score"] != res["score"]:
            st.caption(f"Agent score without process flags: {res['agent_score']}")
        if res.get("checklist"):
            st.subheader("Checklist")
            ck = pd.DataFrame([{"item": c["item_id"], "result": c["result"],
                                "evidence": (c.get("evidence") or {}).get("quote") or ""} for c in res["checklist"]])
            st.dataframe(ck, hide_index=True, use_container_width=True)
        st.subheader("Coaching note")
        st.info(res.get("coaching_note") or "—")
        if st.button("Re-run judge live", help="Calls the judge again on this call and compares with the cached verdict. "
                                                "The demo cache is not modified."):
            with st.spinner("Judging…"):
                try:
                    from qa import judge
                    st.session_state.setdefault("live", {})[cid] = judge.judge_call(cid, force=True, persist=False)
                except Exception as e:  # noqa
                    st.error(f"Judge call failed: {e}")
        live = st.session_state.get("live", {}).get(cid)
        if live:
            cached = sorted(d["code"] for d in res.get("defects", []))
            fresh = sorted(d["code"] for d in live["defects"])
            same = cached == fresh
            (st.success if same else st.warning)(
                f"Live re-run · {live['model']} · prompt {live['prompt_version']} · {live.get('latency_s', '?')}s · "
                f"{live['usage']['input_tokens'] if live.get('usage') else '?'} in / "
                f"{live['usage']['output_tokens'] if live.get('usage') else '?'} out tokens — "
                + ("same judge defects as cached: " + (", ".join(fresh) or "none")
                   if same else f"differs: cached {cached or 'none'} vs live {fresh or 'none'}"))
            for d in live["defects"]:
                ev = d["evidence"]
                st.markdown(f"**{d['code']}** · conf {d['confidence']:.2f} · "
                            f"{html.escape(ev.get('quote') or '(absence) ' + d['rationale'])}")
    with st.expander("Raw JudgeResult JSON"):
        st.json({k: v for k, v in res.items() if k not in ("rule_flags", "merged_defects")})

# ---------------------------------------------------------------- Validation
with tab_val:
    val = csv("validation.csv")
    c = st.columns(4)
    c[0].metric("Critical recall", pct(summary.get("critical_recall")))
    c[1].metric("Clean-call FP rate", pct(summary.get("clean_call_fp_rate")))
    c[2].metric("Evidence validity", pct(summary.get("evidence_validity")))
    rho = summary.get("spearman_rho")
    c[3].metric("Spearman ρ (profile vs score)", "—" if rho is None else f"{rho:.2f}")
    st.caption(f"In-sample: {summary.get('calls_evaluated', '—')} calls · judge prompt "
               f"{', '.join(summary.get('prompt_versions') or ['—'])} · {summary.get('labels_used', '—')} planted labels used, "
               f"{summary.get('labels_dropped_unmanifested', 0)} excluded because the planted behaviour did not appear in the "
               f"rendered transcript (verified by gen.verify_labels).")
    hold_summary = (json.loads((HOLD_OUT / "validation_summary.json").read_text())
                    if (HOLD_OUT / "validation_summary.json").exists() else {})
    hold_val = csv("validation.csv", HOLD_OUT)
    if hold_summary:
        st.markdown(f"**Held-out set** · {hold_summary.get('calls_evaluated')} calls sampled with a different seed after the prompt "
                    f"was frozen, judged once with prompt {', '.join(hold_summary.get('prompt_versions') or ['—'])}; "
                    f"{hold_summary.get('labels_used', '—')} verified labels. A separate dev set was used for tuning (see NOTES.md).")
        h = st.columns(4)
        h[0].metric("Critical recall · held-out", pct(hold_summary.get("critical_recall")))
        h[1].metric("Clean-call FP rate · held-out", pct(hold_summary.get("clean_call_fp_rate")))
        h[2].metric("Evidence validity · held-out", pct(hold_summary.get("evidence_validity")))
        rho_h = hold_summary.get("spearman_rho")
        h[3].metric("Spearman ρ · held-out", "—" if rho_h is None else f"{rho_h:.2f}")
    if not val.empty:
        cols = st.columns(2)
        view = cols[0].radio("Layer", ["merged", "rules", "judge"], horizontal=True)
        which = cols[1].radio("Set", ["in-sample", "held-out"], horizontal=True) if not hold_val.empty else "in-sample"
        src_val = val if which == "in-sample" else hold_val
        v = src_val[src_val.view == view].drop(columns="view")
        st.dataframe(v.style.apply(lambda r: ["background-color:#ffe5e5" if r.severity == "critical" else ""] * len(r), axis=1)
                     .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}, na_rep="—"),
                     hide_index=True, use_container_width=True)
    st.subheader("Agents — hidden ground truth")
    ag = csv("agent_validation.csv")
    if not ag.empty:
        st.dataframe(ag.round(1), hide_index=True, use_container_width=True)
    st.caption("Planted-defect recall is a unit test, not proof: transcripts are cleaner than real ASR and rule-derived "
               "codes are true by construction. Next: 300-call human golden set.")
