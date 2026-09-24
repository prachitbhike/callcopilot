# NOTES

## Choices made (build log)
- Python 3.11 venv at `.venv/` (system python is 3.13). Makefile targets use `.venv/bin/python`.
- `PLAN.md` copied to `CLAUDE.md` per the spec's instructions.
- Git remote: `origin = github.com/prachitbhike/callcopilot` (private). `.env`, `.venv/`, and raw `data/calls.xlsx`
  (contains real agent emails) are gitignored. Everything else incl. `out/` and `data/synthetic/` is committed so
  the app demos offline.
- **Model params:** Sonnet 5-era models reject `temperature` and think by default; forced tool use requires thinking
  off. `qa/llm.py::sampling_kwargs` sends `temperature` to Haiku 4.5 (generator, 0.8) and
  `thinking: {type: disabled}` to Sonnet 5 (judge; no temperature available → "temperature 0" is not settable).
- `OUTCOME_MAP` / `true_status` live in `qa/rules.py`; the generator imports them to derive `OUTCOME_VS_RX` labels,
  so that code is rule-true by construction (as the spec anticipates).
- Rules `MISSING_REF` only fires on connected PA/program calls; planted `MISSING_REF` is restricted to PA/program
  calls so rule and label definitions coincide. Solid agents' random defect ∈ {UNPROFESSIONAL, MISSED_NEXT_STEP,
  MISSING_REF (PA/program only)} — minor codes are derived-only, so not randomly planted.
- Generator writes forms/cases/labels first (deterministic from seed), then renders transcripts, appending to
  `transcripts.jsonl` as each finishes, so rules run immediately and re-runs skip rendered calls.
- Transcript validation also requires the REP's name and (PA/program) the reference number to appear verbatim —
  otherwise the form's rep_name/ref would be unverifiable. 1 retry, then template fallback (logged per call).
- Judge cache `out/judge_raw.jsonl` is append-only; last line per call_id wins, so a failed live re-run keeps the old result.
- Quote verification strips quotes/ellipses, normalises whitespace+case; a quote found in a different turn is
  accepted and its turn index corrected. Failed checklist quotes have evidence nulled (not counted);
  `evidence_failures` counts dropped defects only.
- Validation universe = judged calls (so partial runs are comparable across layers). Extra outputs:
  `out/validation_summary.json`, `out/agent_validation.csv`.
- REDUNDANT_CALL ordering within a day uses `hour_of_day` then `call_id` (no minute-level timestamps exist).
