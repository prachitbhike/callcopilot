"""Step 7: per-agent coaching card. python -m qa.coach [--agent AGT-01 --force --no-llm]"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from qa.llm import client_kwargs, sampling_kwargs

load_dotenv()
load_dotenv(".env.local")
OUT = Path("out")
CARDS = OUT / "coaching_cards.jsonl"

SYSTEM = """You are a call-quality coach for an offshore BPO team placing outbound calls to insurance plans,
manufacturer programs and pharmacies on behalf of Forus. From one agent's audited defects (with verbatim
quotes), draft a short, specific, respectful coaching card addressed to the agent. Use only the evidence given;
quote it exactly. If the agent has fewer than two defects, the remaining "fix" items should be forward-looking habits
grounded in the checklist items they passed least often, not invented incidents. Submit via the submit_card tool."""

TOOL = {"name": "submit_card", "description": "Submit the coaching card.",
        "input_schema": {"type": "object", "required": ["strengths", "fix", "practice_line"], "properties": {
            "strengths": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"},
                          "description": "two specific things the agent does well"},
            "fix": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
                "type": "object", "required": ["behaviour", "quote", "why_it_matters"],
                "properties": {"behaviour": {"type": "string", "description": "<= 20 words"},
                               "quote": {"type": "string", "description": "one verbatim quote from the evidence"},
                               "why_it_matters": {"type": "string", "description": "<= 25 words"}}}},
            "practice_line": {"type": "string", "description": "one sentence the agent can practise saying"}}}}


def load_cards():
    out = {}
    if CARDS.exists():
        for line in CARDS.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                out[o["agent_id"]] = o
    return out


def evidence_for(agent_id, results):
    rows, passes = [], []
    for r in results:
        if r["agent_id"] != agent_id:
            continue
        for m in r["merged_defects"]:
            if m.get("kind") == "process":
                continue  # hold overruns / after-hours dials / repeat calls are dialer & workflow issues, not coaching material
            rows.append({"call_type": r["call_type"], "code": m["code"], "severity": m["severity"],
                         "source": m["source"], "quote": m.get("quote"), "reason": m.get("reason")})
        passes += [c["item_id"] for c in r.get("checklist", []) if c["result"] == "pass"]
    return rows, passes


def draft_card(agent_id, results, force=False, no_llm=False):
    cards = load_cards()
    if agent_id in cards and not force:
        return cards[agent_id]
    defects, passes = evidence_for(agent_id, results)
    if no_llm:
        card = {"strengths": ["(stub)", "(stub)"], "fix": [], "practice_line": "(no-llm stub)"}
    else:
        import anthropic
        from collections import Counter
        model = os.environ["JUDGE_MODEL"]
        prompt = (f"Agent {agent_id}, {sum(r['agent_id'] == agent_id for r in results)} audited calls.\n"
                  f"Checklist items passed (counts): {dict(Counter(passes).most_common(8))}\n"
                  f"Defects:\n{json.dumps(defects, indent=1)}")
        from qa.judge import _repair
        client = anthropic.Anthropic(**client_kwargs())
        for attempt in range(3):
            r = client.messages.create(model=model, max_tokens=2000, **sampling_kwargs(model, 0.0), system=SYSTEM,
                                       tools=[TOOL], tool_choice={"type": "tool", "name": "submit_card"},
                                       messages=[{"role": "user", "content": prompt}])
            card = _repair(next(b.input for b in r.content if b.type == "tool_use"))
            card["fix"] = [_repair({"x": f})["x"] if isinstance(f, str) else f for f in card.get("fix", [])]
            if (len(card.get("strengths", [])) == 2 and len(card["fix"]) == 2 and card.get("practice_line")
                    and all(isinstance(f, dict) and {"behaviour", "quote", "why_it_matters"} <= f.keys() for f in card["fix"])):
                break
        else:
            raise RuntimeError(f"coaching card for {agent_id} failed schema check 3x")
    card = {"agent_id": agent_id, **card, "n_defects": len(defects)}
    OUT.mkdir(exist_ok=True)
    with open(CARDS, "a") as fh:
        fh.write(json.dumps(card) + "\n")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    results = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines() if l.strip()]
    agents = [a.agent] if a.agent else sorted({r["agent_id"] for r in results})
    for ag in agents:
        c = draft_card(ag, results, a.force, a.no_llm)
        print(f"{ag}: {c['n_defects']} defects · fix: {[f['behaviour'][:50] for f in c['fix']]}")


if __name__ == "__main__":
    main()
