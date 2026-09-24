# Call Quality Copilot

A prototype QA system for outbound BPO calls to insurance plans, manufacturer programs and
pharmacies placed on behalf of Forus. It scores every call by checking three records against
each other: the **transcript** (what was said), the **agent form** (what the agent logged) and
the **case / Rx record** (what actually happened).

The full build spec is in [PLAN.md](PLAN.md). This README covers the design decisions behind it
and the tradeoffs we accepted.

> **Status:** the repo has the build plan only. The decisions below describe the intended design.

---

## Core thesis: look for contradictions between records

Today, quality problems only come to light when a counterparty escalates. We bet that the most
damaging defects are **disagreements between records**. For example, a form says "spoke with
rep, ref #4471" on a 12-second ring-out, or a form says "approved" while the rep said "pending".
None of the three records shows these defects by itself.

**Tradeoff:** this puts integrity problems (fabricated contact, wrong status) ahead of softer
quality signals like tone or empathy. Those still get scored, but they aren't the focus.

---

## Three-layer architecture

| Layer | Runs on | Cost | Catches |
| --- | --- | --- | --- |
| 1. Rules | 100% of calls | ~$0 | Record contradictions and policy breaches (duration vs claimed contact, form vs case truth, hold limits, after-hours, redundant calls) |
| 2. LLM judge | Every transcript | ~cents per call | Conversation behaviour: PHI oversharing, unprofessional conduct, missed next steps, form-vs-transcript mismatches |
| 3. Humans | Every critical flag | Reviewer time | Confirm or reject; their verdicts become the golden set |

**Decision: each defect goes to the cheapest layer that catches it reliably.** `HOLD_OVERRUN`
is arithmetic, so an LLM shouldn't handle it. `PHI_DISCLOSURE` needs the conversation read, so a
regex shouldn't handle it. Two codes (`FAB_CONTACT` and `MISSING_REF`) are deliberately
detected by **both** layers, which lets us measure layer agreement.

**Tradeoff:** two detection paths means a merge step and possible double-counting. We merge on
`(call_id, code)` and record `source = rules | judge | both` so overlap is visible, not hidden.

### The judge never sees the case truth

The judge gets the transcript, the form and a *public* slice of the case (call type,
destination, patient initials, drug, prescriber). It does **not** get the truth timeline.

- **Why:** this keeps the two layers independent. If the judge could see the truth, it could
  just repeat the rules layer, and "layer agreement" would mean nothing.
- **Cost:** the judge can't catch `OUTCOME_VS_RX` (form vs system of record) by itself. That code
  is rules-only. Humans still see the truth in the Call Inspector.

---

## LLM judge design

- **Evidence or it didn't happen.** Every defect and failed checklist item must cite a turn index
  and a verbatim quote. For absences ("no rep was ever reached"), it must set
  `evidence_kind: "absence"` explicitly.
- **Quotes are checked in code.** After the model responds, we normalise each quote and confirm
  it is a substring of the cited turn. Defects whose quote fails the check are **dropped** and
  counted in `evidence_failures`, which is reported as *evidence validity*.
  - *Tradeoff:* a real defect with a slightly paraphrased quote gets thrown away. We accept lower
    recall in exchange for never showing a reviewer a made-up quote.
- **Structured output through forced tool use.** The schema comes from pydantic models in
  `qa/schemas.py`, shared by the judge, the pipeline and the app. We chose forced tool use over
  free-text JSON parsing to avoid parse failures.
- **Temperature 0 for the judge** and a pinned `PROMPT_VERSION`, so reruns are comparable. The
  stretch Step 9 measures how stable the output actually is instead of assuming it.
- **Fixed defect vocabulary.** The judge can only emit six named codes, so it can't invent new
  categories that the scoring and validation can't handle.
- **Don't blame the agent for the counterparty.** The prompt tells the judge to ignore IVR
  outages, hold time and rep behaviour. Hold time is handled by rules against policy limits.
- **Per-call-type checklists** (PA, program, transfer) plus a shared baseline. A single generic
  rubric would miss type-specific next steps, such as asking for the appeal deadline on a denial.

### Side benefit: the auto-drafted form

The judge fills in a `transcript_value` for each form field so it can compare against the form.
That output is effectively an **auto-drafted form**, and the UI labels it that way ("Transcript
supports (auto-draft)"). Form entry takes about 5 minutes per call, roughly as long as the call
itself, so the same extraction that audits the form could also pre-fill it.

---

## Scoring

```
score = max(0, 100 − 40·criticals − 15·majors − 5·minors)
any critical ⇒ needs_human_review
```

- **Simple and legible on purpose.** A supervisor can reproduce any score by hand. We gave up
  calibrated weights for transparency.
- **Criticals dominate.** One critical defect costs more than two majors plus a minor.
  Fabricating contact should never be offset by good manners.
- **Agent scorecards show mean score *and* criticals per 100 calls side by side.** A mean score
  can hide a rare but serious pattern, and a critical rate can hide broad sloppiness. Neither
  number is enough alone.

---

## Synthetic data, grounded in real data

We don't have real transcripts or forms, so we generate them. The generator is **seeded from the
real Aug 2024 call log**: real durations, destinations, Rx IDs, call-type mix and time-of-day
patterns. Agents are pseudonymised (`AGT-01..08`) and patients are fake.

- **Agent profiles come from real statistics, not names.** For example, the "logs without
  connecting" profile goes to the tenured agents with the highest share of calls under 20
  seconds. The "ramping" profile goes to the new cohort, whose hold-overrun rate is about 23%
  versus 12.5% for tenured agents. The planted behaviours match patterns that exist in the data.
- **Every planted defect is logged as ground truth** in `labels.csv`, so we can measure the
  judge's precision and recall before trusting it.
- **Transcripts get behavioural hints, never defect names.** The generator is told "agent
  volunteers the patient's full DOB before being asked", never "plant PHI_DISCLOSURE". This
  keeps the judge from pattern-matching on vocabulary.
- **Forms are built deterministically** from truth plus planted defects, not by the LLM. Form
  defects are then exact and reproducible.
- **At least 25% of calls are clean**, so the clean-call false-positive rate is measurable.

**Tradeoffs we're open about:**

- Synthetic transcripts are cleaner than real speech-to-text output (no crosstalk, no
  mis-hearings). The judge will look better here than in production.
- Rule-derived codes (`HOLD_OVERRUN`, `AFTER_HOURS`, `REDUNDANT_CALL`, `OUTCOME_VS_RX`) are
  labelled by the same logic that detects them, so their recall is **true by construction**.
- The app says so directly: *planted-defect recall is a unit test, not proof.* The real next
  step is a 300-call human golden set.
- Generation uses the cheap model (Haiku) at temperature 0.8 for variety. Judging uses the
  stronger model (Sonnet) at temperature 0. Spending goes where accuracy matters.

---

## Engineering choices

| Decision | Why | Given up |
| --- | --- | --- |
| Flat files (CSV / JSONL), no database | Zero setup; files are easy to diff and inspect | Concurrency, querying, scale |
| Streamlit + Plotly | Fastest path to a three-tab interactive demo | UI polish, multi-user state |
| `python -m` modules + Makefile | Each stage runs and checks on its own | No orchestration framework |
| Idempotent, cached LLM steps (skip call_ids already done; `--force` to override) | Reruns cost almost nothing; a crash doesn't restart from zero | Must remember `--force` after prompt changes |
| `--no-llm` on every LLM script | The pipeline never blocks on API or Wi-Fi problems | Template output is only good enough to exercise the plumbing |
| `asyncio`, semaphore of 8, 3 retries with backoff | 80 calls finish in minutes without hitting rate limits | Some added complexity |
| Commit `out/` in the demo snapshot | The demo runs fully offline | Generated files end up in git |
| Model IDs from env, checked against `models.list()` | No hardcoded model IDs going stale | One extra setup step |
| Budget < $5 for the whole build | 80 calls is enough to show the concept | Small sample; per-code metrics are noisy |

### Deliberately out of scope

Auth, deployment, test suites beyond per-step checks, CSS polish and docstrings. The build had a
90-minute window, and we chose a working end-to-end demo over code quality everywhere. The
fallback rules in [PLAN.md §9](PLAN.md) make these cuts ahead of time so they don't need to be
debated mid-build.

---

## Privacy

- Real agent emails never appear in any synthetic or output file. They are dropped right after
  profile assignment.
- All patients are fake, and the app shows a **SYNTHETIC** banner.
- The PHI checklist item (`MIN_PHI`: give only the identifiers the rep asks for) comes from the
  same idea: the product should model least-disclosure, not just check for it.

---

## What we'd change for production

- Swap planted-label validation for a **human-labelled golden set**, and track judge-vs-human
  agreement per code (the stretch review queue starts this).
- Run on real speech-to-text transcripts and re-check quote verification against transcription
  noise (fuzzy matching instead of exact substring).
- Calibrate score weights and confidence thresholds against reviewer outcomes.
- Move from flat files to a real store once multiple reviewers or daily volume (~8k calls/month)
  are involved.
- Measure run-to-run stability continuously, not as a one-off.
