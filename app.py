import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

SYN, OUT = Path(os.environ.get("SYN_DIR", "data/synthetic")), Path(os.environ.get("OUT_DIR", "out"))
# Held-out set C (seed 11): generated after the prompt was frozen and judged exactly once. Set B (out/holdout, seed 7)
# was used to tune the MISSED_NEXT_STEP rules between prompt v4 and v6, so it is a dev set, not a test set.
HOLD_OUT = Path(os.environ.get("HOLDOUT_DIR", "out/holdout_c" if Path("out/holdout_c/validation_summary.json").exists() else "out/holdout"))
SEV_COLOR = {"critical": "#d62728", "major": "#f0a202", "minor": "#9e9e9e"}
SONNET_IN_PER_M, SONNET_OUT_PER_M = 2.0, 10.0  # claude-sonnet-5 list price per million tokens, checked 2026-09-24
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
hold_summary = (json.loads((HOLD_OUT / "validation_summary.json").read_text())
                if (HOLD_OUT / "validation_summary.json").exists() else {})
ops = json.loads((OUT / "ops_summary.json").read_text()) if (OUT / "ops_summary.json").exists() else {}

st.title("Call Quality Copilot")
st.warning("**SYNTHETIC** transcripts and forms, seeded from the real Aug 2024 TaskUs call log (real durations, "
           "destinations, Rx IDs; agents pseudonymised). Agent profiles are illustrative.")

if scored.empty:
    st.error("No scored calls yet — run `make demo`.")
    st.stop()

PAGES = ["Overview", "Call Inspector", "Agents", "Review Queue", "Operations", "Validation"]
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
    def n_calls_with(*codes):
        return int(sum(any(m["code"] in codes for m in r["merged_defects"]) for r in results.values()))

    review = int(scored.needs_human_review.sum())
    c = st.columns(5)
    c[0].metric("Calls audited", f"{len(scored)}", help=f"rules ran on all; {int(scored.judged.sum())} judged by the LLM")
    c[1].metric("Fabricated contacts", n_calls_with("FAB_CONTACT"),
                help="form claims a rep conversation on a call that never got past the IVR")
    c[2].metric("Status contradictions", n_calls_with("STATUS_MISMATCH", "OUTCOME_VS_RX"),
                help="form outcome contradicts what the rep said (judge) or the system of record (rules)")
    c[3].metric("PHI over-disclosures", n_calls_with("PHI_DISCLOSURE"),
                help="agent volunteered DOB / address / diagnosis / member ID before the rep asked")
    c[4].metric("Calls needing a human", f"{review} of {len(scored)}", delta=f"-{1 - review / len(scored):.0%} listening",
                delta_color="inverse", help="every call with a critical flag; the rest are auditable from the flags alone")
    q = st.columns(5)
    q[0].metric("Mean score", f"{scored.score.mean():.1f}", help="spec score: 100 − 40·critical − 15·major − 5·minor")
    q[1].metric("Form accuracy", pct(scored.form_accuracy.mean()),
                help="share of verifiable form fields the transcript supports; fields the transcript cannot verify "
                     "(e.g. outcome on a no-answer call) are excluded")
    q[2].metric("Evidence validity", pct(summary.get("evidence_validity")), help="judge quotes found verbatim in the transcript")
    q[3].metric("Critical recall · held-out", pct(hold_summary.get("critical_recall")),
                help="planted criticals caught on calls never used for prompt tuning (Validation page)")
    q[4].metric("Clean-call FP rate · held-out", pct(hold_summary.get("clean_call_fp_rate")),
                help="zero-label held-out calls that received a major or critical flag")

    md = pd.DataFrame([{**m, "call_id": r["call_id"], "call_type": r["call_type"]}
                       for r in results.values() for m in r["merged_defects"]])
    if not md.empty and "kind" not in md:
        md["kind"] = "agent"
    l, mid, r = st.columns([2, 1, 1])
    if not md.empty:
        ag = md[md.kind == "agent"]
        cnt = ag.groupby(["code", "severity"]).size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(cnt, x="code", y="count", color="severity", color_discrete_map=SEV_COLOR,
                     title="Agent-conduct defects by code", category_orders={"code": cnt.code.tolist()})
        l.plotly_chart(fig, width="stretch")
        pr = md[md.kind == "process"].groupby("code").size().reset_index(name="count").sort_values("count", ascending=False)
        if not pr.empty:
            figp = px.bar(pr, x="code", y="count", title="Process flags (dialer / workflow)", color_discrete_sequence=["#9e9e9e"])
            mid.plotly_chart(figp, width="stretch")
    crit = scored.assign(crit=scored.n_critical > 0).groupby("call_type").crit.mean().reset_index()
    fig2 = px.bar(crit, x="call_type", y="crit", title="Critical rate by call type", color_discrete_sequence=["#d62728"])
    fig2.update_yaxes(tickformat=".0%", title=None)
    r.plotly_chart(fig2, width="stretch")
    st.caption(f"Process flags (hold overrun, after-hours dial, repeat call) come from call metadata and are not agent conduct; "
               f"they count in the spec score but are tagged separately. Layer agreement (rules ∩ judge on FAB_CONTACT / "
               f"MISSING_REF): {pct(summary.get('layer_agreement'))}.")

    st.subheader("Cost & scale")
    tin, tout, lat = scored.input_tokens.mean(), scored.output_tokens.mean(), scored.latency_s.mean()
    monthly = int(ops.get("calls", 8294))
    if tin == tin:
        per_call = tin / 1e6 * SONNET_IN_PER_M + tout / 1e6 * SONNET_OUT_PER_M
        k = st.columns(5)
        k[0].metric("Judge tokens / call", f"{tin / 1000:.1f}k in · {tout / 1000:.1f}k out", help="measured from API usage on the judged calls")
        k[1].metric("Judge cost / call", f"${per_call:.3f}",
                    help=f"claude-sonnet-5 list price ${SONNET_IN_PER_M:.0f} / ${SONNET_OUT_PER_M:.0f} per M tokens; before prompt caching")
        k[2].metric(f"Per month at {monthly:,} calls", f"${per_call * monthly:,.0f}", help="August 2024 volume from the real call log")
        k[3].metric("Latency / call", f"{lat:.0f} s", help="wall-clock per judge call at 8 concurrent requests")
        conn = scored[scored.connected.astype(str) == "True"]
        k[4].metric("Form fields auto-fillable", f"{conn.auto_fill_fields.mean():.1f} / 5",
                    help="on connected calls: fields the judge extracted from the transcript, i.e. a pre-filled form")
        st.caption(f"Reviewing only flagged calls cuts listening by {1 - review / len(scored):.0%}. The debrief measured about 5 minutes of "
                   f"form entry per call, ≈ {monthly * 5 / 60:,.0f} agent-hours a month at this volume that the auto-draft can pre-fill.")
    else:
        st.caption("Token usage appears after a judge run with prompt v4 or later.")

# ---------------------------------------------------------------- Inspector
if nav == "Call Inspector":
    s = scored.sort_values(["score", "call_id"])
    labels = {row.call_id: f"{row.score} · {row.agent_id} · {row.call_type} · "
                           f"{row.destination_name if isinstance(row.destination_name, str) else 'Unknown destination'} · "
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
        st.dataframe(diff.style.apply(hl, axis=1), width="stretch")
        filled = sum(1 for f in FIELDS if fc.get(f, {}).get("transcript_value") not in (None, "", "null"))
        st.caption(f"Middle column = the judge's extraction from the transcript, effectively an auto-drafted form: "
                   f"{filled}/{len(FIELDS)} fields pre-fillable from this call.")
        with st.expander("Auto-drafted form (what the transcript supports)"):
            dc = st.columns(2)
            for i, f in enumerate(FIELDS):
                tv = fc.get(f, {}).get("transcript_value")
                dc[i % 2].text_input(f, value="" if tv in (None, "", "null") else str(tv), disabled=True, key=f"draft_{cid}_{f}")
            st.caption("In production this draft pre-fills the agent's form; the agent confirms or corrects instead of typing. "
                       "Empty = the transcript did not establish the value.")

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
            st.dataframe(ck, hide_index=True, width="stretch")
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

# ---------------------------------------------------------------- Operations (real log, rules only)
if nav == "Operations":
    oa, oh, ot = csv("ops_agents.csv"), csv("ops_hours.csv"), csv("ops_types.csv")
    if not ops:
        st.info("Run `python -m qa.ops` with data/calls.xlsx present to build the real-log aggregates.")
    else:
        st.markdown(f"**Layer 1 on the real August 2024 log** · {ops['calls']:,} calls · {ops['agents']} agents (pseudonymised) · "
                    f"{ops['destinations']} destinations · {ops['hours_on_calls']:,.0f} hours on calls. Rules only, no transcripts: "
                    "these run today on the dialer export.")
        k = st.columns(6)
        k[0].metric("After-hours dials", f"{ops['after_hours']:,}", f"{ops['after_hours_share']:.1%} of calls", delta_color="off",
                    help="placed at or after 7 pm or before 8 am ET")
        k[1].metric("Hold overruns", f"{ops['overrun']:,}", f"{ops['overrun_share']:.1%} · {ops['excess_hold_hours']:.0f} h excess",
                    delta_color="off", help="duration > hold limit + 60 s (420 s transfer, 900 s PA / program)")
        k[2].metric("Calls under 20 s", f"{ops['under20']:,}", f"{ops['under20_share']:.1%} · {ops['zero_s']} at 0 s", delta_color="off",
                    help="never reached a rep; the sampling signal behind FAB_CONTACT")
        k[3].metric("Repeat dials", f"{ops['repeat_dials']:,}", f"{ops['repeat_dial_share']:.0%} · upper bound", delta_color="off",
                    help="2nd+ call by the same agent to the same number on the same day; the log has no case reference, "
                         "so a shared pharmacy line counts as a repeat")
        k[4].metric("Transfer calls per Rx", f"{ops['transfer_calls_per_rx']:.2f}",
                    help=f"{ops['transfer_calls']:,} transfer-confirm calls for {ops['rx_transferred']:,} transferred prescriptions")
        k[5].metric("New cohort over hold limit", f"{ops['new_cohort_over_limit_share']:.0%}",
                    f"vs {ops['tenured_over_limit_share_same_dates']:.1%} tenured, same dates", delta_color="off",
                    help=f"{ops['new_cohort_agents']} agents whose first call is on or after Aug 20")
        l, r = st.columns([1, 1])
        if not oh.empty:
            oh = oh.copy()
            oh["window"] = oh.et_hour.map(lambda h: "after hours" if h >= 19 or h < 8 else "business hours")
            fig = px.bar(oh, x="et_hour", y="under20_share", color="window", hover_data=["calls"],
                         color_discrete_map={"after hours": "#d62728", "business hours": "#9e9e9e"},
                         title="Share of calls under 20 s by hour (ET)")
            fig.update_yaxes(tickformat=".0%", title=None)
            fig.update_xaxes(title="hour of day, ET")
            l.plotly_chart(fig, width="stretch")
        if not oa.empty:
            fig3 = px.scatter(oa, x="over_limit_share", y="under20_share", size="calls", color="cohort", text="agent",
                              hover_data=["calls", "after_hours_share", "repeat_dial_share"],
                              color_discrete_map={"tenured": "#4c72b0", "new (from Aug 20)": "#f0a202"},
                              title="Agents: hold-limit breaches vs short calls (bubble = volume)")
            fig3.update_traces(textposition="top center", textfont_size=9)
            fig3.update_xaxes(tickformat=".0%", title="share of calls over the hold limit")
            fig3.update_yaxes(tickformat=".0%", title="share of calls under 20 s")
            r.plotly_chart(fig3, width="stretch")
        st.caption(f"Under-20 s share is {ops['under20_share_9_17']:.1%} during 9–17 ET and {ops['under20_share_19_20']:.1%} at 19–20 ET: "
                   "evening dials mostly reach closed destinations. AGT-01..08 are the agents sampled for the judge demo. "
                   "Short calls are a sampling signal, not proof of fabrication: the judge confirms on transcripts.")
        if not oa.empty:
            show = oa[["agent", "cohort", "calls", "first_call", "under20_share", "zero_s_calls", "over_limit_share", "overrun_share",
                       "excess_hold_min", "after_hours_share", "repeat_dial_share", "in_judge_sample"]].copy()
            for col in ["under20_share", "over_limit_share", "overrun_share", "after_hours_share", "repeat_dial_share"]:
                show[col] = show[col].map("{:.1%}".format)
            show["excess_hold_min"] = show.excess_hold_min.round(0).astype(int)
            st.dataframe(show, hide_index=True, width="stretch")
        if not ot.empty:
            ot2 = ot.copy()
            for col in ["under20_share", "overrun_share", "after_hours_share"]:
                ot2[col] = ot2[col].map("{:.1%}".format)
            ot2["mean_duration_s"] = ot2.mean_duration_s.round(0).astype(int)
            st.dataframe(ot2, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Validation
if nav == "Validation":
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
                     hide_index=True, width="stretch")
    for fname in ("stability_focus.json", "stability.json"):
        stab = json.loads((OUT / fname).read_text()) if (OUT / fname).exists() else {}
        if stab:
            st.markdown(f"**Stability** · {stab['calls']} calls ({stab.get('selection', 'random')}) × {stab['runs']} judge runs, "
                        f"prompt {stab.get('prompt_version')}: {stab['calls_with_identical_defect_sets']:.0%} of calls got identical "
                        f"defect sets · {stab['identical_code_decisions']:.0%} of {stab['judge_code_decisions']} (call, code) decisions "
                        f"identical · checklist items identical {stab['checklist_items_identical']:.0%} of {stab['checklist_items']} · "
                        f"form-field verdicts identical {stab['form_fields_identical']:.0%} of {stab['form_fields']} · "
                        f"score std-dev mean {stab['mean_score_std']:.1f} (max {stab['max_score_std']:.1f}).")
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
