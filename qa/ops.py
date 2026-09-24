"""Layer 1 on the real call log: rule flags that need no transcript. python -m qa.ops

Reads data/calls.xlsx (gitignored: it holds real agent emails, which never leave this module) and writes
pseudonymised aggregates the app can show offline:
  out/ops_summary.json   month totals + cohort comparison
  out/ops_agents.csv     one row per agent (AGT-01..08 = the agents sampled for the judge demo, others agent-09..)
  out/ops_hours.csv      calls and under-20 s share by ET hour
  out/ops_types.csv      calls, short-call and hold-overrun shares by call type
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from gen.generate import HOLD, agent_stats, load, pick_profiles

OUT = Path("out")
NEW_COHORT_FROM = dt.date(2024, 8, 20)


def run():
    calls, rx = load()
    calls["after_hours"] = (calls.et_hour >= 19) | (calls.et_hour < 8)
    calls["under20"] = calls.duration_seconds < 20
    calls["zero"] = calls.duration_seconds == 0
    calls["over_limit"] = calls.duration_seconds > calls.hold_limit            # spec §3 signal
    calls["overrun"] = calls.duration_seconds > calls.hold_limit + 60          # HOLD_OVERRUN rule
    calls["excess_min"] = (calls.duration_seconds - calls.hold_limit - 60).clip(lower=0) / 60
    order = calls.sort_values(["call_date", "hour", "call_id"])
    calls["repeat_dial"] = order.groupby(["email", "destination", "call_date"]).cumcount().reindex(calls.index) > 0
    calls["connected"] = calls.duration_seconds >= 30

    # pseudonyms: the 8 demo agents keep their AGT ids (same deterministic pick as the generator), others by volume rank
    st = agent_stats(calls)
    demo = dict(zip(pick_profiles(st)[:8], [f"AGT-{i + 1:02d}" for i in range(8)]))
    others = st.drop(index=list(demo)).sort_values("volume", ascending=False).index
    alias = {**demo, **{e: f"agent-{i + 9:02d}" for i, e in enumerate(others)}}
    calls["agent"] = calls.email.map(alias)

    g = calls.groupby("agent")
    agents = pd.DataFrame({
        "calls": g.size(), "first_call": g.call_date.min(),
        "cohort": g.call_date.min().map(lambda d: "new (from Aug 20)" if d >= NEW_COHORT_FROM else "tenured"),
        "under20_share": g.under20.mean(), "zero_s_calls": g.zero.sum(),
        "over_limit_share": g.over_limit.mean(), "overrun_share": g.overrun.mean(),
        "excess_hold_min": g.excess_min.sum(), "after_hours_share": g.after_hours.mean(),
        "repeat_dial_share": g.repeat_dial.mean(), "connected_share": g.connected.mean(),
        "median_duration_s": g.duration_seconds.median(), "hours_on_calls": g.duration_seconds.sum() / 3600,
    }).reset_index()
    agents["in_judge_sample"] = agents.agent.str.startswith("AGT-")
    agents = agents.sort_values("calls", ascending=False)

    hours = calls.groupby("et_hour").agg(calls=("call_id", "size"), under20_share=("under20", "mean"),
                                         connected_share=("connected", "mean")).reset_index()
    types = calls.groupby("call_type").agg(calls=("call_id", "size"), under20_share=("under20", "mean"),
                                           overrun_share=("overrun", "mean"), mean_duration_s=("duration_seconds", "mean"),
                                           after_hours_share=("after_hours", "mean")).reset_index()
    types["hold_limit_s"] = types.call_type.map(HOLD)

    n = len(calls)
    ten = calls[calls.email.isin(st[st.first_date < NEW_COHORT_FROM].index)]
    new = calls[~calls.email.isin(st[st.first_date < NEW_COHORT_FROM].index)]
    same_dates = new.call_date.min()
    summary = {
        "calls": n, "agents": int(calls.email.nunique()), "destinations": int(calls.destination_name.nunique()),
        "date_from": str(calls.call_date.min()), "date_to": str(calls.call_date.max()),
        "hours_on_calls": round(calls.duration_seconds.sum() / 3600, 1),
        "after_hours": int(calls.after_hours.sum()), "after_hours_share": calls.after_hours.mean(),
        "overrun": int(calls.overrun.sum()), "overrun_share": calls.overrun.mean(),
        "excess_hold_hours": round(calls.excess_min.sum() / 60, 1),
        "under20": int(calls.under20.sum()), "under20_share": calls.under20.mean(), "zero_s": int(calls.zero.sum()),
        "under20_share_19_20": calls[calls.et_hour.isin([19, 20])].under20.mean(),
        "under20_share_9_17": calls[calls.et_hour.between(9, 17)].under20.mean(),
        "repeat_dials": int(calls.repeat_dial.sum()), "repeat_dial_share": calls.repeat_dial.mean(),
        "transfer_calls": int((calls.call_type == "transfer_confirm_call").sum()), "rx_transferred": int(len(rx)),
        "transfer_calls_per_rx": (calls.call_type == "transfer_confirm_call").sum() / len(rx),
        "new_cohort_agents": int(new.email.nunique()),
        "new_cohort_over_limit_share": new.over_limit.mean(),
        "tenured_over_limit_share_same_dates": ten[ten.call_date >= same_dates].over_limit.mean(),
        "agent_under20_median": float(agents.under20_share.median()),
        "agent_under20_max": float(agents.under20_share.max()),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "ops_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    agents.round(4).to_csv(OUT / "ops_agents.csv", index=False)
    hours.round(4).to_csv(OUT / "ops_hours.csv", index=False)
    types.round(4).to_csv(OUT / "ops_types.csv", index=False)
    print(f"real log: {n} calls · {summary['agents']} agents · {summary['destinations']} destinations · "
          f"{summary['date_from']}..{summary['date_to']} · {summary['hours_on_calls']} h on calls")
    print(f"after-hours {summary['after_hours']} ({summary['after_hours_share']:.1%}) · hold overruns {summary['overrun']} "
          f"({summary['overrun_share']:.1%}, {summary['excess_hold_hours']} h excess) · <20 s {summary['under20']} "
          f"({summary['under20_share']:.1%}, {summary['zero_s']} at 0 s) · repeat dials {summary['repeat_dials']} "
          f"({summary['repeat_dial_share']:.1%}, upper bound) · {summary['transfer_calls_per_rx']:.2f} transfer calls / Rx")
    print(f"<20 s share 9-17h {summary['under20_share_9_17']:.1%} vs 19-20h {summary['under20_share_19_20']:.1%} · "
          f"new cohort over hold limit {summary['new_cohort_over_limit_share']:.1%} vs tenured same dates "
          f"{summary['tenured_over_limit_share_same_dates']:.1%}")
    print("-> out/ops_summary.json, out/ops_agents.csv, out/ops_hours.csv, out/ops_types.csv")
    return summary, agents


if __name__ == "__main__":
    run()
