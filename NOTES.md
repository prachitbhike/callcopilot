# NOTES

## Status (Phase 1 demo hardening, 2026-09-24)
**Works end to end on real data + API:** real Aug 2024 log → 8 profiled agents / 80 sampled calls → Haiku transcripts →
deterministic forms / cases / labels → label verification (`gen.verify_labels`) → rules (100% of calls) → Sonnet 5 judge
(prompt **v6**, strict tool schema) → merge / score → validation on three call sets → Streamlit app
(Overview / Inspector / Validation; live re-judge verified, cache untouched).

**Not on this branch:** stretch Steps 7–9 (Agents page + coaching cards, Review Queue, `--stability`) were built in a
parallel session on `build/steps-0-5` (commits 1f22cb9, 84ce7be) and still need to be merged with this branch
(both touch `app.py`, `qa/validate.py`, `Makefile`, this file).

### Call sets
| set | dir | seed | role |
| --- | --- | --- | --- |
| A | `data/synthetic` / `out` | 42 | in-sample: every prompt version was tuned on it |
| B | `data/holdout` / `out/holdout` | 7 | dev set: used once to diagnose MISSED_NEXT_STEP false positives (v4 → v6); disjoint from A |
| C | `data/holdout_c` / `out/holdout_c` | 11 | **held-out**: generated after v6 was frozen, judged exactly once; disjoint from A and B |

Each set = 80 calls, same 8 pseudonymous agents, sampled from the real log with the in-sample call_ids excluded.

### Headline numbers (merged rules + judge, verified labels)
| | A in-sample (v6) | C held-out (v6, judged once) |
| --- | --- | --- |
| critical recall (FAB_CONTACT, STATUS_MISMATCH, OUTCOME_VS_RX, PHI_DISCLOSURE) | 1.00 (19/19) | 1.00 (16/16) |
| critical precision | 1.00 | 1.00 |
| clean-call false-positive rate (major or critical flag on a zero-label call) | 0.00 (45 clean) | 0.03 (1 of 30 clean) |
| evidence validity (judge quotes found verbatim) | 1.00 (22/22) | 1.00 (27/27) |
| UNPROFESSIONAL · MISSING_REF · MISSED_NEXT_STEP recall | 1.00 · 1.00 · 1.00 | 1.00 · 1.00 · 1.00 |
| MISSING_REF · MISSED_NEXT_STEP precision | 1.00 · 1.00 | 0.80 · 0.40 |
| Spearman ρ (profile badness vs mean agent score) | −0.63 | −0.78 |
| labels used (dropped as not manifested) | 56 (1) | 60 (3) |

C disagreements, all major-severity: 2 × TX_RESEND flagged when the agent gave the NPI but had no fax number
(strict but arguable), 1 × PA_NOT_ON_FILE (agent never asked how to submit), 1 × MISSING_REF where the rep spoke
the Rx number in a different format than the record. No critical disagreement on any set.

Dev set B history (why v4 → v6): v4 on B: critical recall 1.00, clean-call FP 0.10, MISSED_NEXT_STEP 0/3 + 5 FP
(readback / self-identification mislabelled as MISSED_NEXT_STEP), 1 STATUS_MISMATCH FP contradicting the judge's own
form_check, 1 MISSING_REF FP from a form Rx number the rep never spoke. v5 on B: MISSED_NEXT_STEP 4/4 + 7 FP
(new granular items applied to non-applicable statuses). v6 fixes are listed below.

**Judge cost (measured from API usage):** ~4.9k input + ~1.2k output tokens per call on `claude-sonnet-5` ≈ **$0.021 / call**,
~9 s latency at 8 concurrent → 80 calls ≈ $1.70; 8,294 calls / month ≈ **$175 / month** before prompt caching.
Today's build spent ≈ $13 (≈ 600 judge calls across prompt versions and sets, 160 Haiku renders).

Launch: `make app` (= `.venv/bin/streamlit run app.py`). Held-out set: `make holdout` (writes `data/holdout`, `out/holdout`).

## What changed in Phase 1 (from the demo snapshot a2069d8)
- **Prompt v3 → v6** (`qa/judge.py`): strict tool schema (`strict: true`, every property required, no defaults) after v4
  returned partial tool inputs on 4 calls; checklist items split into single facts (`qa/rubric.yaml`: PA_AUTH_NUMBER /
  PA_AUTH_DATES, PA_DENIAL_REASON / PA_APPEAL_DEADLINE / PA_PEER_TO_PEER, PA_NOT_ON_FILE, ENR_DENIAL_INFO, TX_RX_NUMBER);
  MISSED_NEXT_STEP restricted to call-type items with an explicit applicability table; form_check `transcript_value`
  must be form-ready (enum / name / number / null) so the middle column of the three-way diff is a real auto-draft.
- **Post-processing guards:** confidence floor 0.5 (placeholder defects arrive at ~0.0–0.3); STATUS_MISMATCH must cite a
  REP turn *and* agree with the judge's own form_check; MISSING_REF must quote the REP turn; counted in `guard_drops`.
- **Label verification** (`gen/verify_labels.py`, run by the generator): each planted judge-code label is checked against
  the rendered transcript (did the rep really withhold the appeal deadline? did the agent really sigh?). Labels that did
  not manifest are excluded from validation (`manifested` column). MISSED_NEXT_STEP is additionally **derived** for any
  connected call whose transcript lacks a rubric-required fact (`planted_by=derived-facts`), the same way OUTCOME_VS_RX
  is derived. Before this, 2 of 5 in-sample MISSED_NEXT_STEP "misses" were label noise (Haiku had the rep volunteer the
  info) and 2 were real judge misses (auth number / denial reason never stated, judge passed the bundled item).
- **Generator consistency:** transfer reps must state the pharmacy Rx number unless MISSED_NEXT_STEP is planted; a
  post-render pass nulls a form Rx number the rep never spoke; dates and DOBs are spoken naturally (no ISO); 3 render
  attempts before template fallback; `--rerender ID,ID`, `--exclude CSV` (disjoint held-out sampling), `SYN_DIR`/`OUT_DIR`
  env vars so every module runs on another set; blank destination names → "Unknown destination".
- **Metrics:** form accuracy counts verifiable fields only (unverifiable fields on no-answer calls were dragging 92% down
  to 81%); `agent_score` and `kind: process|agent` on merged defects (HOLD_OVERRUN, AFTER_HOURS, REDUNDANT_CALL are dialer /
  workflow flags, tagged "process" in the Inspector); token usage + latency captured per judge call.
- **App:** live re-run compares against the cached verdict without writing to `out/` (a re-run during the demo can no
  longer change headline numbers); Validation shows in-sample and held-out side by side plus labels-used / dropped;
  auto-draft caption counts pre-fillable fields; Streamlit `cache_data` key fix (underscore args are ignored).
- Re-rendered the robotic template transcript (939 s Dupixent MyWay call) and the ISO-date CVS call.

## Honest caveats for the debrief
- Set A numbers are in-sample by construction; quote **set C**. Set B was consumed by tuning.
- MISSED_NEXT_STEP is the judgement-heavy code: recall is solid, precision on C is 0.40 (3 FPs, all defensible strictness).
  Rule-derived codes (OUTCOME_VS_RX, HOLD_OVERRUN, AFTER_HOURS, REDUNDANT_CALL) are true by construction.
- Label verification and derived labels use keyword / date matching; a rep phrasing a fact unusually can mis-verify.
- Transcripts are cleaner than real ASR; 0-second calls fall back to a one-line "[ringing]" template (7 across A/B/C).
- Stability (Step 9) exists on `build/steps-0-5` for prompt v3 only; re-run on v6 after the merge.

## Choices made (build log)
- Python 3.11 venv at `.venv/` (system python is 3.13). Makefile targets use `.venv/bin/python`. Worktrees symlink `.venv`,
  copy `.env` / `.env.local` and symlink `data/calls.xlsx` from the main checkout (all gitignored).
- `PLAN.md` copied to `CLAUDE.md` per the spec's instructions.
- Git remote: `origin = github.com/prachitbhike/callcopilot` (private). `.env`, `.venv`, and raw `data/calls.xlsx`
  (contains real agent emails) are gitignored. Everything else incl. `out/` and `data/*` is committed so the app demos offline.
- **Model params:** Sonnet 5 rejects `temperature` and thinks by default; forced tool use requires thinking off.
  `qa/llm.py::sampling_kwargs` sends `temperature` (0.8) to Haiku 4.5 via `extra_body` and `thinking: disabled` to Sonnet 5.
- `OUTCOME_MAP` / `true_status` live in `qa/rules.py`; the generator imports them to derive `OUTCOME_VS_RX` labels.
- Rules `MISSING_REF` only fires on connected PA/program calls; planted `MISSING_REF` is restricted to PA/program calls.
  Solid agents' random defect ∈ {UNPROFESSIONAL, MISSED_NEXT_STEP, MISSING_REF (PA/program only)}.
- Generator writes forms/cases/labels first (deterministic from seed), then renders transcripts, appending as each finishes.
- Transcript validation requires the REP's name, the reference number (PA/program) and the pharmacy Rx number (transfer,
  unless MISSED_NEXT_STEP is planted) to appear verbatim.
- Judge cache `out/judge_raw.jsonl` is append-only; last line per call_id wins. It keeps v1–v6 history (useful for
  prompt-version comparisons); `--force` re-judges everything.
- Quote verification strips quotes/ellipses, normalises whitespace+case; a quote found in another turn is accepted and
  its turn index corrected. `evidence_failures` counts dropped defects only.
- Validation universe = judged calls. Extra outputs: `validation_summary.json`, `agent_validation.csv` per set.
- REDUNDANT_CALL ordering within a day uses `hour_of_day` then `call_id` (call_id embeds a precise UTC timestamp).
- Label floor: each profile-driven code is topped up to ≥ 2 planted labels on eligible calls of the matching profile.
- Clean-call PHI leakage: the scenario card only includes address / member ID / diagnosis when PHI_DISCLOSURE is planted.
- Transfer forms log the pharmacy Rx number as `reference_number`; the judge prompt defines reference_number per call type.
- The spec's `score` formula is unchanged; `agent_score` (without process flags) is an additional column.
- ENR_BV_RESULT applies only when enrolled (pending BV has no result to obtain); ENR_BRIDGE when enrolled or pending BV.
