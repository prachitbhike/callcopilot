# NOTES

## Status (Step 6 freeze)
**Works end to end on real data + API:** real Aug 2024 log → 8 profiled agents / 80 sampled calls → Haiku
transcripts (78 LLM, 2 template fallback) → deterministic forms/cases/labels → rules (100% of calls) →
Sonnet 5 judge (80/80, prompt v3) → merge/score → validation → Streamlit app (Overview / Inspector / Validation,
live re-judge verified).
**Stretch (Steps 7–9) done:** Agents page (table sorted by criticals, defect mix, 3 worst calls → Inspector,
Sonnet coaching cards cached in `out/coaching_cards.jsonl` for all 8 agents); Review Queue (criticals first then
lowest confidence; Confirm / Reject / Change code → `out/human_labels.csv`; Validation page shows judge-vs-human
agreement); stability (`make stability`, results in `out/stability.json`).
**Not done / stubbed:** no human verdicts are shipped (the queue starts empty; test clicks were deleted).

**Planted labels (80 calls, 45 clean):** AFTER_HOURS 13 · HOLD_OVERRUN 11 · OUTCOME_VS_RX 6 · PHI_DISCLOSURE 5 ·
MISSED_NEXT_STEP 5 · FAB_CONTACT 4 · STATUS_MISMATCH 4 · REDUNDANT_CALL 4 · MISSING_REF 2 · UNPROFESSIONAL 2.

**Validation (merged):** critical recall 1.00 · clean-call FP rate 0.00 · evidence validity 1.00 (0/21 quotes
failed) · layer agreement 1.00 · Spearman ρ(profile badness, score) = −0.63. Only miss: MISSED_NEXT_STEP recall
0.60 (3/5). **Caveat:** prompt v3 + the n/a-defect guard were tuned after seeing v2 validation FPs on this same
set, so v3 numbers are in-sample. v2 (pre-tuning) headline: critical recall 1.00, clean-call FP 0.22, evidence
validity 1.00 (0/33), ρ −0.59 (`out/validation_summary_v2.json`, `out/validation_v2.csv`). v2 FPs were mostly
spec ambiguity (transfer "received" with back-order/insurance-reject blockers; conditional checklist items).

**Stability (Step 9, 15 calls × 3 Sonnet re-runs, prompt v3):** run 1: 100% identical defect sets (only 3 judge
defect decisions in the sample); run 2: 13/15 calls identical, 50% of 4 (call, code) decisions identical, score
std-dev mean 0.9 (max 7.1 — one MISSED_NEXT_STEP flipping), checklist results 99% identical (144 items),
form_check verdicts 88% identical (75 fields). Debrief line: judge is stable on criticals; residual variance sits
in the judgement-heavy major code MISSED_NEXT_STEP and in "unverifiable vs mismatch" form-field calls.

Launch: `make app` (= `.venv/bin/streamlit run app.py`).

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
- **Label floor:** seed 42 planted 0 UNPROFESSIONAL and 1 MISSING_REF at the spec's probabilities, leaving those
  judge codes unmeasurable. The generator now tops up each profile-driven code to >= 2 labels on eligible calls
  of the matching profile (planted_by=profile).
- Real data check: `call_id` embeds a precise UTC timestamp, so same-day REDUNDANT_CALL ordering is exact.
  Picked agents match §3: AGT-01/02 under-20s share 14.5% / 12.6%; ramping AGT-03/04 over-hold ≈ 25%.
- **Models:** `GEN_MODEL=claude-haiku-4-5-20251001`, `JUDGE_MODEL=claude-sonnet-5` (from `models.list()`).
  anthropic SDK 1.x removed `temperature` from `messages.create()`; Haiku gets it via `extra_body` (0.8).
  Sonnet 5 rejects non-default temperature, so the judge runs with `thinking: disabled` (required for forced
  tool use) at default sampling.
- Clean-call PHI leakage: the scenario card only includes address / member ID / diagnosis when PHI_DISCLOSURE
  is planted (otherwise Haiku volunteered them on clean calls → judge false positives).
- Transfer forms log the pharmacy Rx number as `reference_number` (the natural "reference" for a pharmacy
  call); judge prompt defines reference_number per call type.
- Judge prompt v2 (after a 3-call check): no STATUS_MISMATCH / MISSING_REF on no-rep calls (FAB_CONTACT
  subsumes them); STATUS_MISMATCH must quote a REP turn (enforced in code, not just prompt).
- Generation: 78/80 transcripts from Haiku; 2 fell back to templates after two validation failures
  (t_sec out of range / reference number not spoken).
- App nav uses a horizontal `st.radio` (not `st.tabs`) so buttons can switch to the Inspector programmatically.
- Fixed: Streamlit `cache_data` ignores `_`-prefixed args, so the file-mtime cache key never invalidated; renamed.
- Coaching card: forced `submit_card` tool; JSON-in-string repair + shape check with up to 3 attempts.
