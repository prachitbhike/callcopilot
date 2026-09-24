"""Step 3: LLM judge. python -m qa.judge [--limit N --force --no-llm]"""
import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from qa.llm import sampling_kwargs
from qa.rules import load_inputs
from qa.schemas import AuditSubmission, JudgeResult

load_dotenv()
load_dotenv(".env.local")
PROMPT_VERSION = "v1"
OUT = Path("out")
RAW = OUT / "judge_raw.jsonl"
RUBRIC = yaml.safe_load(open("qa/rubric.yaml"))
FORM_FIELDS = ["spoke_with_rep", "rep_name", "reference_number", "outcome_status", "next_action"]

SYSTEM = f"""You are a QA auditor for outbound pharmacy, insurance-plan and manufacturer-program calls placed on behalf of Forus. Judge only on transcript evidence. For every fail and every defect cite the turn index and a verbatim quote from that turn; if the evidence is an absence (nothing was said / no rep reached) use evidence_kind "absence" with turn null. Do not penalise the agent for the counterparty's behaviour, IVR outages or hold time. Use only these defect codes: {", ".join(RUBRIC["judge_codes"])}. Compare the agent's form to the transcript field by field ({", ".join(FORM_FIELDS)}) and report match / mismatch / unverifiable. Write a two-sentence coaching note addressed to the agent.

Defect definitions and severities:
""" + "\n".join(f"- {c} ({RUBRIC['codes'][c]['severity']}): {RUBRIC['codes'][c]['definition']}" for c in RUBRIC["judge_codes"]) + """

Quotes must be copied exactly from a single turn (a short contiguous span is best). Keep rationale to 25 words or fewer.
Report a defect only when the transcript/form evidence supports it; a clean call should have no defects.
Set needs_human_review true if any critical defect is reported or you are unsure. Submit via the submit_audit tool."""


def _inline(schema):
    defs = schema.pop("$defs", {})

    def walk(o):
        if isinstance(o, dict):
            if "$ref" in o:
                return walk(copy.deepcopy(defs[o["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in o.items() if k != "title"}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(schema)


TOOL = {"name": "submit_audit", "description": "Submit the QA audit for this call.",
        "input_schema": _inline(AuditSubmission.model_json_schema())}


def case_public(call, case):
    return {"call_type": call["call_type"], "destination_name": call["destination_name"],
            "patient_initials": case["patient"]["initials"], "drug": case.get("drug"), "prescriber": case.get("prescriber")}


def build_prompt(call, transcript, form, case_pub, rubric=RUBRIC):
    items = {**rubric["checklists"]["all"], **rubric["checklists"][call["call_type"]]}
    checklist = "\n".join(f"- {k}: {v}" for k, v in items.items())
    tx = "\n".join(f"[{t['i']}] (t={t['t_sec']}s) {t['speaker']}: {t['text']}" for t in transcript)
    form_pub = {k: v for k, v in form.items() if k not in ("agent_id",)}
    return (f"Call type: {call['call_type']} · duration {call['duration_seconds']}s\n\n"
            f"Checklist (report every item as pass / fail / na):\n{checklist}\n\n"
            f"Transcript (numbered turns):\n{tx}\n\n"
            f"Agent form:\n{json.dumps(form_pub, indent=1)}\n\n"
            f"Case (public facts):\n{json.dumps(case_pub, indent=1)}")


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[\"“”‘’'…]|\.\.\.", "", s or "")).strip().lower()


def verify_quote(ev, turns):
    """True if quote is found; fixes the turn index when found in another turn."""
    q = _norm(ev.quote)
    if not q:
        return False
    by_i = {t["i"]: t for t in turns}
    if ev.turn in by_i and q in _norm(by_i[ev.turn]["text"]):
        return True
    for t in turns:
        if q in _norm(t["text"]):
            ev.turn = t["i"]
            return True
    return False


def postprocess(sub: AuditSubmission, call_id, turns, model):
    fails = 0
    keep = []
    for d in sub.defects:
        if d.code not in RUBRIC["judge_codes"]:
            continue
        d.severity = RUBRIC["codes"][d.code]["severity"]
        if d.evidence.evidence_kind == "quote" and not verify_quote(d.evidence, turns):
            fails += 1
            continue
        keep.append(d)
    for c in sub.checklist:
        if c.result == "fail" and c.evidence and c.evidence.evidence_kind == "quote" and not verify_quote(c.evidence, turns):
            c.evidence = None
    res = JudgeResult(**sub.model_dump(exclude={"defects"}), defects=keep, call_id=call_id,
                      model=model, prompt_version=PROMPT_VERSION, evidence_failures=fails)
    if any(d.severity == "critical" for d in keep):
        res.needs_human_review = True
    return res


def stub_result(call, form):
    items = {**RUBRIC["checklists"]["all"], **RUBRIC["checklists"][call["call_type"]]}
    return JudgeResult(call_id=call["call_id"], checklist=[{"item_id": k, "result": "na"} for k in items], defects=[],
                       form_check=[{"field": f, "form_value": None if form.get(f) is None else str(form.get(f)),
                                    "transcript_value": None, "match": "unverifiable"} for f in FORM_FIELDS],
                       coaching_note="(no-llm stub)", needs_human_review=False, model="stub", prompt_version=PROMPT_VERSION)


async def judge_one(client, sem, call, turns, form, case, model):
    prompt = build_prompt(call, turns, form, case_public(call, case))
    last = None
    for k in range(3):
        try:
            async with sem:
                r = await client.messages.create(
                    model=model, max_tokens=8000, **sampling_kwargs(model, 0.0), system=SYSTEM, tools=[TOOL],
                    tool_choice={"type": "tool", "name": "submit_audit"},
                    messages=[{"role": "user", "content": prompt}])
            block = next(b for b in r.content if b.type == "tool_use")
            sub = AuditSubmission.model_validate(block.input)
            return postprocess(sub, call["call_id"], turns, model)
        except Exception as e:  # noqa - retry API + validation errors
            last = e
            await asyncio.sleep(2 ** (k + 1))
    raise RuntimeError(f"judge failed for {call['call_id']}: {last}")


def load_raw():
    out = {}
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                out[o["call_id"]] = o
    return out


def _context():
    sample, forms, cases, tx = load_inputs()
    calls = {r["call_id"]: r for r in sample.to_dict("records")}
    return calls, forms, cases, tx


async def judge_all(limit=None, force=False, no_llm=False, call_ids=None):
    OUT.mkdir(exist_ok=True)
    calls, forms, cases, tx = _context()
    done = load_raw()
    todo = [c for c in (call_ids or calls) if c in tx and (force or c not in done)]
    if limit:
        todo = todo[:limit]
    # append-only cache: load_raw keeps the last line per call_id, so a failed forced re-run keeps the old result
    stats = {"cached": len(done), "judged": 0, "failed": 0}
    model = os.environ.get("JUDGE_MODEL", "")
    if not no_llm:
        from anthropic import AsyncAnthropic
        client, sem = AsyncAnthropic(), asyncio.Semaphore(8)
    results = dict(done)

    async def job(cid):
        call, form = calls[cid], forms[cid]
        try:
            res = stub_result(call, form) if no_llm else await judge_one(
                client, sem, call, tx[cid], form, cases[call["case_ref"]], model)
        except Exception as e:  # noqa
            print(f"  {e}", flush=True)
            stats["failed"] += 1
            return
        results[cid] = res.model_dump()
        with open(RAW, "a") as fh:
            fh.write(json.dumps(results[cid]) + "\n")
        stats["judged"] += 1
        print(f"  judged {cid}: {len(res.defects)} defects, {res.evidence_failures} evidence failures", flush=True)

    await asyncio.gather(*(job(c) for c in todo))
    return results, stats


def judge_call(call_id, force=True):
    """Sync single-call entry point for the app. Raises on failure."""
    res, st = asyncio.run(judge_all(force=force, call_ids=[call_id]))
    if st["failed"]:
        raise RuntimeError(f"judge failed for {call_id} (see console)")
    return res[call_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    res, st = asyncio.run(judge_all(a.limit, a.force, a.no_llm))
    print(f"judge: {st}; total results {len(res)} -> {RAW}")


if __name__ == "__main__":
    main()
