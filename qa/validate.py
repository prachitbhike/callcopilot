"""Step 4: precision/recall vs planted labels. python -m qa.validate"""
import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import pandas as pd
import yaml
from scipy.stats import spearmanr

load_dotenv()
load_dotenv(".env.local")
SYN = Path("data/synthetic")
OUT = Path("out")
RUBRIC = yaml.safe_load(open("qa/rubric.yaml"))
SEV = {c: v["severity"] for c, v in RUBRIC["codes"].items()}
BADNESS = {"solid": 0, "ramping": 1, "curt": 2, "phi_oversharer": 2, "logs_without_connecting": 3}


def load_results():
    return [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines() if l.strip()]


def pred_pairs(results, view):
    out = set()
    for r in results:
        for m in r["merged_defects"]:
            if view == "merged" or m["source"] in (view, "both"):
                out.add((r["call_id"], m["code"]))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else np.nan
    return p, r, f


def compute():
    results = load_results()
    labels = pd.read_csv(SYN / "labels.csv", dtype={"call_id": str})
    judged = {r["call_id"] for r in results if r.get("judged")}
    universe = judged or {r["call_id"] for r in results}
    results = [r for r in results if r["call_id"] in universe]
    gold = {(c, k) for c, k in zip(labels.call_id, labels.code) if c in universe}
    rows = []
    for view in ("rules", "judge", "merged"):
        pred = pred_pairs(results, view)
        for code in RUBRIC["codes"]:
            det = RUBRIC["codes"][code]["detector"]
            if view != "merged" and view not in det:
                continue
            p_c = {x for x in pred if x[1] == code}
            g_c = {x for x in gold if x[1] == code}
            tp, fp, fn = len(p_c & g_c), len(p_c - g_c), len(g_c - p_c)
            p, r, f = prf(tp, fp, fn)
            rows.append({"view": view, "code": code, "severity": SEV[code], "support": len(g_c),
                         "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f})
    val = pd.DataFrame(rows)

    merged = pred_pairs(results, "merged")
    crit_gold = {x for x in gold if SEV[x[1]] == "critical"}
    critical_recall = len(crit_gold & merged) / len(crit_gold) if crit_gold else np.nan
    judge_crit_gold = {x for x in crit_gold if "judge" in RUBRIC["codes"][x[1]]["detector"]}
    judge_crit_recall = (len(judge_crit_gold & pred_pairs(results, "judge")) / len(judge_crit_gold)
                         if judge_crit_gold else np.nan)
    labelled = {c for c, _ in gold}
    clean = [r for r in results if r["call_id"] not in labelled]
    clean_fp = (np.mean([any(m["severity"] in ("critical", "major") for m in r["merged_defects"]) for r in clean])
                if clean else np.nan)
    ev_fail = sum(r.get("evidence_failures", 0) for r in results if r.get("judged"))
    ev_total = ev_fail + sum(len(r.get("defects", [])) for r in results if r.get("judged"))
    evidence_validity = 1 - ev_fail / ev_total if ev_total else np.nan
    both_codes = [c for c, v in RUBRIC["codes"].items() if set(v["detector"]) >= {"rules", "judge"}]
    md = [m for r in results for m in r["merged_defects"] if m["code"] in both_codes]
    agreement = np.mean([m["source"] == "both" for m in md]) if md else np.nan

    prof = pd.read_csv(SYN / "agent_profiles.csv")
    sc = pd.DataFrame([{"agent_id": r["agent_id"], "score": r["score"],
                        "crit": sum(m["severity"] == "critical" for m in r["merged_defects"])} for r in results])
    ag = sc.groupby("agent_id").agg(calls=("score", "size"), mean_score=("score", "mean"), crit=("crit", "sum")).reset_index()
    ag["criticals_per_100"] = 100 * ag.crit / ag.calls
    ag = prof[["agent_id", "profile"]].merge(ag.drop(columns="crit"), on="agent_id")
    ag["badness"] = ag.profile.map(BADNESS)
    rho = spearmanr(ag.badness, ag.mean_score).statistic if len(ag) > 2 else np.nan

    headline = {"calls_evaluated": len(universe), "critical_recall": critical_recall,
                "judge_critical_recall": judge_crit_recall, "clean_call_fp_rate": clean_fp,
                "clean_calls": len(clean), "evidence_validity": evidence_validity, "evidence_failures": ev_fail,
                "judge_defects_total": ev_total, "layer_agreement": agreement, "spearman_rho": rho}

    hum = OUT / "human_labels.csv"
    human = None
    if hum.exists():
        h = pd.read_csv(hum, dtype={"call_id": str})
        if len(h):
            h = h.drop_duplicates(["call_id", "code"], keep="last")
            human = h.groupby("code").agg(reviewed=("verdict", "size"),
                                          agree=("verdict", lambda s: (s == "confirm").mean())).reset_index()
    return val, headline, ag, human


def stability(n, runs, seed=7):
    """Re-judge n random calls `runs` times (no cache writes); report decision agreement + score std-dev."""
    import asyncio
    import os
    import random
    from anthropic import AsyncAnthropic
    from qa import judge
    from qa.llm import client_kwargs
    from qa.pipeline import merge
    from qa.rules import load_inputs, run_rules

    sample, forms, cases, tx = load_inputs()
    flags = run_rules(sample, forms, cases, tx)
    calls = {r["call_id"]: r for r in sample.to_dict("records")}
    ids = random.Random(seed).sample(sorted(calls), n)
    model = os.environ["JUDGE_MODEL"]

    async def go():
        client, sem = AsyncAnthropic(**client_kwargs()), asyncio.Semaphore(8)

        async def one(cid):
            c = calls[cid]
            r = await judge.judge_one(client, sem, c, tx[cid], forms[cid], cases[c["case_ref"]], model)
            return cid, r.model_dump()
        return [dict(await asyncio.gather(*(one(c) for c in ids))) for _ in range(runs)]

    per_run = asyncio.run(go())
    sub = sample[sample.call_id.isin(ids)]
    scores, decisions = {c: [] for c in ids}, {c: [] for c in ids}
    for jr in per_run:
        rows, _ = merge(sub, flags[flags.call_id.isin(ids)], jr)
        for r in rows:
            scores[r["call_id"]].append(r["score"])
            decisions[r["call_id"]].append({d["code"] for d in r["defects"]})
    def agree(key):  # share of per-call items whose value is identical in every run
        vals = {}
        for jr in per_run:
            for cid, r in jr.items():
                for it in r[key]:
                    vals.setdefault((cid, it["item_id" if key == "checklist" else "field"]), []).append(
                        it["result" if key == "checklist" else "match"])
        return float(np.mean([len(set(v)) == 1 for v in vals.values()])), len(vals)
    ck_agree, ck_n = agree("checklist")
    fc_agree, fc_n = agree("form_check")
    pairs = [(c, k) for c in ids for k in set().union(*decisions[c])]
    same = [all(k in d for d in decisions[c]) for c, k in pairs]
    identical_calls = np.mean([all(d == decisions[c][0] for d in decisions[c]) for c in ids])
    sd = np.mean([np.std(v) for v in scores.values()])
    out = {"calls": n, "runs": runs, "judge_code_decisions": len(pairs),
           "identical_code_decisions": float(np.mean(same)) if same else 1.0,
           "calls_with_identical_defect_sets": float(identical_calls), "mean_score_std": float(sd),
           "max_score_std": float(max(np.std(v) for v in scores.values())), "model": model,
           "prompt_version": judge.PROMPT_VERSION,
           "checklist_items_identical": ck_agree, "checklist_items": ck_n,
           "form_fields_identical": fc_agree, "form_fields": fc_n}
    (OUT / "stability.json").write_text(json.dumps(out, indent=1))
    print(f"Stability ({model}, {judge.PROMPT_VERSION}): {n} calls x {runs} runs -> "
          f"{out['identical_code_decisions']:.0%} of {len(pairs)} (call, code) judge decisions identical across runs; "
          f"{identical_calls:.0%} of calls had identical defect sets; score std-dev mean {sd:.1f} (max {out['max_score_std']:.1f}); "
          f"checklist results identical {ck_agree:.0%} of {ck_n}; form_check verdicts identical {fc_agree:.0%} of {fc_n}.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", type=int, help="(Step 9) re-judge N random calls")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    if a.stability:
        stability(a.stability, a.runs)
        return
    val, h, ag, human = compute()
    val.to_csv(OUT / "validation.csv", index=False)
    ag.to_csv(OUT / "agent_validation.csv", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps({k: (None if v != v else v) for k, v in h.items()}, indent=1, default=float))
    pd.set_option("display.width", 160)
    print(val[val.view == "merged"].round(2).to_string(index=False))
    print(f"\ncalls evaluated {h['calls_evaluated']} · critical recall {h['critical_recall']:.2f} "
          f"(judge-only criticals {h['judge_critical_recall']:.2f}) · clean-call FP rate {h['clean_call_fp_rate']:.2f} "
          f"({h['clean_calls']} clean) · evidence validity {h['evidence_validity']:.2f} "
          f"({h['evidence_failures']}/{h['judge_defects_total']}) · layer agreement {h['layer_agreement']:.2f}")
    print(ag.round(1).to_string(index=False))
    print(f"Spearman rho(profile badness, mean score) = {h['spearman_rho']:.2f}")
    if human is not None:
        print("judge-vs-human agreement:\n" + human.to_string(index=False))


if __name__ == "__main__":
    main()
