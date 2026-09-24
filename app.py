import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

SYN, OUT = Path("data/synthetic"), Path("out")
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

PAGES = ["Overview", "Call Inspector", "Agents", "Review Queue", "Validation"]
st.session_state.setdefault("nav", "Overview")
nav = st.radio("View", PAGES, key="nav", horizontal=True, label_visibility="collapsed")


def open_call(cid):
    st.session_state["selected_call"] = cid
    st.session_state["insp_call"] = cid
    st.session_state["nav"] = "Call Inspector"


def pct(x):
    return "—" if x is None or x != x else f"{x:.0%}"


# ---------------------------------------------------------------- Overview
if nav == "Overview":
    c = st.columns(5)
    c[0].metric("Calls scored", f"{len(scored)}", help=f"{int(scored.judged.sum())} judged by LLM")
    c[1].metric("% calls with a critical", pct((scored.n_critical > 0).mean()))
    c[2].metric("Mean score", f"{scored.score.mean():.1f}")
    c[3].metric("Form accuracy", pct(scored.form_accuracy.mean()), help="share of form fields the transcript supports")
    c[4].metric("Evidence validity", pct(summary.get("evidence_validity")), help="judge quotes found verbatim in the transcript")
    md = pd.DataFrame([{**m, "call_id": r["call_id"], "call_type": r["call_type"]}
                       for r in results.values() for m in r["merged_defects"]])
    l, r = st.columns([2, 1])
    if not md.empty:
        cnt = md.groupby(["code", "severity"]).size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(cnt, x="code", y="count", color="severity", color_discrete_map=SEV_COLOR,
                     title="Defects by code", category_orders={"code": cnt.code.tolist()})
        l.plotly_chart(fig, width="stretch")
        src = md.groupby(["code", "source"]).size().reset_index(name="n")
    crit = scored.assign(crit=scored.n_critical > 0).groupby("call_type").crit.mean().reset_index()
    fig2 = px.bar(crit, x="call_type", y="crit", title="Critical rate by call type", color_discrete_sequence=["#d62728"])
    fig2.update_yaxes(tickformat=".0%", title=None)
    r.plotly_chart(fig2, width="stretch")
    st.caption(f"Layer agreement (rules ∩ judge on FAB_CONTACT / MISSING_REF): {pct(summary.get('layer_agreement'))} · "
               f"calls needing human review: {int(scored.needs_human_review.sum())}")

# ---------------------------------------------------------------- Inspector
if nav == "Call Inspector":
    s = scored.sort_values(["score", "call_id"])
    labels = {row.call_id: f"{row.score} · {row.agent_id} · {row.call_type} · {row.destination_name} · "
                           f"{row.duration_seconds}s · {row.codes if isinstance(row.codes, str) else '—'}"
              for row in s.itertuples()}
    ids = list(labels)
    default = ids.index(st.session_state["selected_call"]) if st.session_state.get("selected_call") in ids else 0
    if st.session_state.get("insp_call") not in ids:
        st.session_state["insp_call"] = ids[default]
    cid = st.selectbox("Call (lowest score first)", ids, key="insp_call", format_func=labels.get)
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
            rows.append({"field": f, "Agent logged (form)": str(form.get(f)),
                         "Transcript supports (auto-draft)": chk.get("transcript_value") or "—",
                         "System of record (case truth)": truth_col[f], "judge": chk.get("match", "—")})
        diff = pd.DataFrame(rows).set_index("field")
        rule_codes = {m["code"] for m in res["merged_defects"]}

        def hl(row):
            bad = row["judge"] == "mismatch" or (row.name == "outcome_status" and "OUTCOME_VS_RX" in rule_codes) or \
                  (row.name == "spoke_with_rep" and "FAB_CONTACT" in rule_codes)
            return ["background-color:#ffd6d6" if bad else ""] * len(row)
        st.dataframe(diff.style.apply(hl, axis=1), width="stretch")
        st.caption("Middle column = the judge's extraction from the transcript — effectively an auto-drafted form.")

        st.subheader(f"Defects · score {res['score']}")
        if not res["merged_defects"]:
            st.success("No defects.")
        for m in sorted(res["merged_defects"], key=lambda m: ["critical", "major", "minor"].index(m["severity"])):
            q = f"“{html.escape(m['quote'])}”" if m.get("quote") else html.escape(m.get("reason") or "")
            st.markdown(f"<span style='background:{SEV_COLOR[m['severity']]};color:white;border-radius:4px;padding:1px 6px'>"
                        f"{m['severity']}</span> <span style='background:#555;color:white;border-radius:4px;padding:1px 6px'>"
                        f"{m['source']}</span> **{m['code']}** · conf {m.get('confidence', 1):.2f}<br>"
                        f"<span style='font-size:0.9em'>{q}</span>", unsafe_allow_html=True)
        if res.get("checklist"):
            st.subheader("Checklist")
            ck = pd.DataFrame([{"item": c["item_id"], "result": c["result"],
                                "evidence": (c.get("evidence") or {}).get("quote") or ""} for c in res["checklist"]])
            st.dataframe(ck, hide_index=True, width="stretch")
        st.subheader("Coaching note")
        st.info(res.get("coaching_note") or "—")
        if st.button("Re-run judge live"):
            with st.spinner("Judging…"):
                try:
                    from qa import judge, pipeline
                    judge.judge_call(cid, force=True)
                    pipeline.rebuild()
                    st.session_state["selected_call"] = cid
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:  # noqa
                    st.error(f"Judge call failed: {e}")
    with st.expander("Raw JudgeResult JSON"):
        st.json({k: v for k, v in res.items() if k not in ("rule_flags", "merged_defects")})

# ---------------------------------------------------------------- Agents
if nav == "Agents":
    rows = []
    for ag, g in scored.groupby("agent_id"):
        codes = pd.Series([c for cs in g.codes.dropna() for c in cs.split(";")])
        rows.append({"agent_id": ag, "calls": len(g), "mean_score": round(g.score.mean(), 1),
                     "criticals_per_100": round(100 * g.n_critical.sum() / len(g), 1),
                     "top_defect": codes.value_counts().index[0] if len(codes) else "—",
                     "pct_needing_review": f"{g.needs_human_review.mean():.0%}",
                     "form_accuracy": f"{g.form_accuracy.mean():.0%}"})
    agt = pd.DataFrame(rows).sort_values(["criticals_per_100", "mean_score"], ascending=[False, True])
    st.dataframe(agt, hide_index=True, width="stretch")
    with st.expander("Ground truth (synthetic)"):
        st.dataframe(csv("agent_profiles.csv", SYN), hide_index=True, width="stretch")

    ag = st.selectbox("Agent", agt.agent_id.tolist())
    g = scored[scored.agent_id == ag]
    l, r = st.columns([1, 1])
    with l:
        mix = pd.DataFrame([{"code": m["code"], "severity": m["severity"]}
                            for c in g.call_id for m in results[c]["merged_defects"]])
        if mix.empty:
            st.success("No defects for this agent.")
        else:
            cnt = mix.groupby(["code", "severity"]).size().reset_index(name="count").sort_values("count", ascending=False)
            st.plotly_chart(px.bar(cnt, x="code", y="count", color="severity", color_discrete_map=SEV_COLOR,
                                   title=f"{ag} defect mix", category_orders={"code": cnt.code.tolist()}),
                            width="stretch")
        st.markdown("**3 worst calls**")
        for row in g.sort_values("score").head(3).itertuples():
            st.button(f"{row.score} · {row.call_type} · {row.destination_name} · {row.codes if isinstance(row.codes, str) else '—'}",
                      key=f"worst_{row.call_id}", on_click=open_call, args=(row.call_id,))
    with r:
        from qa import coach
        card = coach.load_cards().get(ag)
        if st.button("Draft coaching card" if card is None else "Re-draft coaching card"):
            with st.spinner("Drafting…"):
                try:
                    card = coach.draft_card(ag, list(results.values()), force=card is not None)
                except Exception as e:  # noqa
                    st.error(f"Coach call failed: {e}")
        if card:
            st.markdown(f"#### Coaching card · {ag}")
            st.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in card["strengths"]))
            st.markdown("**Fix next**")
            for f in card["fix"]:
                st.markdown(f"- **{f['behaviour']}**  \n  > {f['quote']}  \n  _{f['why_it_matters']}_")
            st.info(f"Practice line: {card['practice_line']}")

# ---------------------------------------------------------------- Review queue
if nav == "Review Queue":
    import datetime as _dt
    HL = OUT / "human_labels.csv"
    done = csv("human_labels.csv")
    reviewed = set(zip(done.call_id, done.code)) if not done.empty else set()
    reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer", "qa-lead"), key="reviewer")
    all_codes = list(yaml.safe_load(open("qa/rubric.yaml"))["codes"])
    queue = []
    for cid in scored[scored.needs_human_review].call_id:
        ms = results[cid]["merged_defects"]
        queue.append((-sum(m["severity"] == "critical" for m in ms), min([m.get("confidence", 1) for m in ms] or [1]), cid))
    queue.sort()
    pending = sum((cid, m["code"]) not in reviewed for _, _, cid in queue for m in results[cid]["merged_defects"]
                  if m["severity"] in ("critical", "major"))
    st.caption(f"{len(queue)} calls need review · {pending} critical/major defects not yet reviewed · "
               f"{len(reviewed)} verdicts recorded → out/human_labels.csv")

    def record(cid, code, verdict, new_code=None):
        row = pd.DataFrame([{"call_id": cid, "code": code, "verdict": verdict, "new_code": new_code,
                             "reviewer": st.session_state.get("reviewer", ""), "ts": _dt.datetime.now().isoformat(timespec="seconds")}])
        row.to_csv(HL, mode="a", header=not HL.exists(), index=False)

    for _, _, cid in queue:
        r_ = results[cid]
        with st.container(border=True):
            st.markdown(f"**{r_['score']}** · {r_['agent_id']} · {r_['call_type']} · {r_['duration_seconds']}s")
            st.button("Open in Inspector", key=f"open_{cid}", on_click=open_call, args=(cid,))
            for m in sorted(r_["merged_defects"], key=lambda m: ["critical", "major", "minor"].index(m["severity"])):
                if m["severity"] == "minor":
                    continue
                k = f"{cid}|{m['code']}"
                cols = st.columns([4, 1, 1, 2])
                q = m.get("quote") or m.get("reason") or ""
                cols[0].markdown(f"<span style='background:{SEV_COLOR[m['severity']]};color:white;border-radius:4px;padding:1px 6px'>"
                                 f"{m['code']}</span> <small>{m['source']} · conf {m.get('confidence', 1):.2f}</small><br>"
                                 f"<small>{html.escape(q)}</small>", unsafe_allow_html=True)
                if (cid, m["code"]) in reviewed:
                    v = done[(done.call_id == cid) & (done.code == m["code"])].iloc[-1]
                    cols[1].markdown(f"✅ {v.verdict}" + (f" → {v.new_code}" if isinstance(v.new_code, str) else ""))
                    continue
                cols[1].button("Confirm", key=f"c_{k}", on_click=record, args=(cid, m["code"], "confirm"))
                cols[2].button("Reject", key=f"r_{k}", on_click=record, args=(cid, m["code"], "reject"))
                nc = cols[3].selectbox("Change code", ["—"] + [c for c in all_codes if c != m["code"]], key=f"s_{k}",
                                       label_visibility="collapsed")
                if nc != "—":
                    cols[3].button(f"Change → {nc}", key=f"x_{k}", on_click=record, args=(cid, m["code"], "change", nc))

# ---------------------------------------------------------------- Validation
if nav == "Validation":
    val = csv("validation.csv")
    c = st.columns(4)
    c[0].metric("Critical recall", pct(summary.get("critical_recall")))
    c[1].metric("Clean-call FP rate", pct(summary.get("clean_call_fp_rate")))
    c[2].metric("Evidence validity", pct(summary.get("evidence_validity")))
    rho = summary.get("spearman_rho")
    c[3].metric("Spearman ρ (profile vs score)", "—" if rho is None else f"{rho:.2f}")
    if not val.empty:
        view = st.radio("Layer", ["merged", "rules", "judge"], horizontal=True)
        v = val[val.view == view].drop(columns="view")
        st.dataframe(v.style.apply(lambda r: ["background-color:#ffe5e5" if r.severity == "critical" else ""] * len(r), axis=1)
                     .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}, na_rep="—"),
                     hide_index=True, width="stretch")
    st.subheader("Agents — hidden ground truth")
    ag = csv("agent_validation.csv")
    if not ag.empty:
        st.dataframe(ag.round(1), hide_index=True, width="stretch")
    hl = csv("human_labels.csv")
    if not hl.empty:
        st.subheader("Judge vs human reviewers")
        hl = hl.drop_duplicates(["call_id", "code"], keep="last")
        src = {(r["call_id"], m["code"]): m["source"] for r in results.values() for m in r["merged_defects"]}
        hl["source"] = [src.get((c, k), "?") for c, k in zip(hl.call_id, hl.code)]
        agree = hl.groupby("code").agg(reviewed=("verdict", "size"), confirmed=("verdict", lambda v: (v == "confirm").sum()),
                                       rejected=("verdict", lambda v: (v == "reject").sum()),
                                       recoded=("verdict", lambda v: (v == "change").sum())).reset_index()
        agree["agreement"] = (agree.confirmed / agree.reviewed).map("{:.0%}".format)
        st.dataframe(agree, hide_index=True, width="stretch")
        st.caption(f"{len(hl)} verdicts · overall agreement {(hl.verdict == 'confirm').mean():.0%}. "
                   "Confirmed verdicts become the golden set.")
    st.caption("Planted-defect recall is a unit test, not proof: transcripts are cleaner than real ASR and rule-derived "
               "codes are true by construction. Next: 300-call human golden set.")
