# Call Quality Copilot — build spec for Claude Code

> **How to use this file.** Save it as `CLAUDE.md` at the root of an empty folder. Copy the
> exercise spreadsheet to `data/calls.xlsx`. Put your key in `.env`. Then paste the
> **Kick-off prompt** (bottom of this file) into Claude Code. Budget: 90 minutes to demo.
> Must-haves (Steps 0–6) fit in ~65 minutes. Stretch (Steps 7–9) uses the rest.

---

## 1. What we are building and why (read first)

Forus uses a BPO (TaskUs) to place outbound calls to insurance plans, manufacturer programs and
pharmacies. Quality problems surface only when a counterparty escalates. We now have (in this
prototype, synthetically) three records per call:

1. **Transcript** — what was actually said
2. **Agent form** — what the agent logged after the call
3. **Case / Rx record** — what actually happened (system of record)

**Thesis:** the most damaging defects are *contradictions between these three records* (e.g. form
says "spoke with rep, ref #4471" on a 12-second ring-out; form says "approved" while the rep
said "pending"). No single source reveals them. So:

- **Layer 1 — Rules** run on 100% of calls at ~$0 and catch record contradictions and policy breaches.
- **Layer 2 — LLM judge** reads every transcript against a per-call-type checklist, compares the
  transcript to the form field by field, and cites a verbatim quote for every finding.
- **Layer 3 — Humans** confirm every critical flag; their verdicts become the golden set.

Synthetic transcripts/forms/cases are **seeded from the real Aug 2024 call log** (real agents
pseudonymised, real destinations, real durations, real Rx IDs). Every planted defect is logged
as ground truth so we can measure the judge's precision/recall before trusting it.

**The demo must show:** (a) an Overview of defects across all scored calls, (b) a Call Inspector
with the three-way diff (transcript vs form vs record) and highlighted evidence, (c) a Validation
tab with precision/recall per defect vs planted labels. Stretch: agent scorecards + auto-drafted
coaching cards, and a human review queue.

**Debrief bridge (label things this way in the UI):** the judge's per-field
`transcript_value` is effectively an *auto-drafted form*. Call the middle column of the three-way
diff **"Transcript supports (auto-draft)"**. The morning analysis found form entry takes ~5 min
per call (≈ call length), so the same extraction that audits the form can pre-fill it.

---

## 2. Ground rules for the build

- **Stack:** Python 3.11, pandas, openpyxl, streamlit, plotly, anthropic, pydantic v2, pyyaml,
  python-dotenv, scipy (for Spearman only). Flat files (CSV/JSONL). No database, no Docker.
- **Time discipline:** working > pretty. No refactors, no unit-test suites, no type-check passes.
  Each step ends with a **Check** command; run it and fix before moving on. Don't ask me
  questions unless truly blocked — make a reasonable choice and log it in `NOTES.md`.
- **Long runs go in the background** (`nohup python -m ... > out/gen.log 2>&1 &`) and you continue
  to the next step; poll the log before you need the output.
- **Idempotent + cached:** every LLM step skips call_ids already present in its output file, so
  re-runs are cheap. Add `--force` to override and `--limit N` for smoke tests.
- **Every LLM script also has `--no-llm`** which produces templated output so the pipeline never
  blocks on the API.
- **Models from env:** `GEN_MODEL` (cheapest current Haiku) and `JUDGE_MODEL` (current Sonnet).
  In Step 0 verify IDs with `client.models.list()` and write the exact IDs into `.env`.
- **Structured output = forced tool use** (`tool_choice={"type":"tool","name":...}`) with a JSON
  schema derived from a pydantic model. Judge temperature 0; generator temperature 0.8.
- **Concurrency:** `asyncio` + `AsyncAnthropic`, semaphore of 8, 3 retries with exponential backoff.
- **Privacy:** never write real agent emails into any synthetic or output file. Use `AGT-01..`.
  All patients are fake. Say "SYNTHETIC" in the app banner.
- **Every script runs as `python -m <module>`** and prints a 5–10 line summary.
- **Makefile:** `make data`, `make rules`, `make judge`, `make validate`, `make app`, `make demo`
  (= data → rules → judge → validate, then prints how to launch the app).
- **Budget:** whole build should cost < $5 in API calls (80 calls generated + judged).

---

## 3. Real data facts (`data/calls.xlsx`)

Sheet **Call Data** (8,294 rows, Aug 1–31 2024, UTC):

| column | notes |
| --- | --- |
| `call_id` | unique string |
| `email` | agent (25 distinct) — pseudonymise, never output |
| `reasons` | `pa_plan_call` (963) · `patient_access_check` (1,046) · `transfer_confirm_call` (6,285) |
| `call_date` | date, UTC |
| `hour_of_day` | time-like string `HH:00:00`, UTC. ET = UTC−4 in August |
| `duration_seconds` | includes IVR + hold; some calls never answered. 572 calls < 20 s, 122 at 0 s |
| `destination`, `destination_name` | phone + entity name (446 distinct names) |
| `related_to_specialty_transfer` | 1 / 0 / blank (474 blank) |

Sheet **Prescriptions Transferred** (4,770 rows): `transferred_on`, `erx_id`, `pharmacy type`
(`retail` 2,476 / `specialty` 2,294). Overall ≈ 1.32 transfer calls per transferred Rx.

**Hold-limit policy:** 420 s for `transfer_confirm_call`, 900 s for the other two.

**Known signals to seed profiles from (compute, don't hardcode names):**
- Agents whose first call is on/after 2024-08-20 (9 of them) = **new cohort**: 23% of their calls
  exceed the hold limit vs 12.5% for tenured agents over the same dates.
- Two tenured high-volume agents have ~13–15% of calls under 20 s vs a 5% agent median.
- Under-20 s share jumps to 11–18% at 7–8 pm ET (dialing closed destinations).

---

## 4. Repo layout

```
CLAUDE.md  NOTES.md  Makefile  requirements.txt  .env  .env.example
data/calls.xlsx
data/synthetic/
  calls_sample.csv        80 sampled real calls + agent_id, case_ref, et_hour, hold_limit
  agent_profiles.csv      agent_id, profile (hidden ground truth)
  cases.jsonl             one per case_ref: facts + true status timeline
  scenarios.jsonl         per call: the plan the generator rendered (facts + defects to plant)
  transcripts.jsonl       {call_id, turns:[{i,t_sec,speaker,text}]}
  forms.jsonl             agent submission per call
  labels.csv              planted defects: call_id, code, severity
gen/generate.py           Step 1
qa/rubric.yaml            codes, severities, weights, checklists
qa/schemas.py             pydantic models shared by judge + app
qa/rules.py               Step 2
qa/judge.py  qa/pipeline.py   Step 3
qa/validate.py            Step 4
qa/coach.py               Step 7 (stretch)
app.py                    Step 5 (+7, 8)
out/rule_flags.csv  out/results.jsonl  out/scored_calls.csv  out/validation.csv
out/human_labels.csv  out/*.log
```

---

## 5. Schemas

### 5.1 `calls_sample.csv`
Real columns from Call Data **minus `email`**, plus: `agent_id` (AGT-01..AGT-08), `call_type`
(= reasons), `case_ref`, `et_hour` (int), `hold_limit` (420/900), `connected` (bool, see §6).

### 5.2 `cases.jsonl` — one per `case_ref`
```json
{"case_ref":"RX-32827","case_type":"transfer_confirm_call","erx_id":32827,
 "pharmacy_type":"specialty","transferred_on":"2024-08-03",
 "target_pharmacy":"Optum Specialty Pharmacy","drug":"Dupixent 300mg/2mL pen",
 "patient":{"first":"Maya","last":"Okafor","dob":"1979-04-12","initials":"M.O."},
 "prescriber":"Dr. Lena Ruiz, NPI 1234567890",
 "truth":{"received_at":"2024-08-04","filled_at":null,"blocker":"needs_pa","pharmacy_rx_number":"RX88213"}}
```
PA cases: `case_type` `pa_plan_call`, fields `plan`, `submitted_on`, `truth:{status: approved|denied|pending|not_on_file, determination_date, auth_number, denial_reason, appeal_deadline}`.
Program cases: `case_type` `patient_access_check`, fields `program`, `truth:{status: enrolled|pending_docs|pending_bv|denied, missing_docs:[...], bv_result, bridge_ship_date}`.
Add `"resolved_before_call": true|false` on transfer cases (10% true → REDUNDANT_CALL).

### 5.3 `transcripts.jsonl`
```json
{"call_id":"...","turns":[{"i":0,"t_sec":0,"speaker":"IVR","text":"Thank you for calling Optum Specialty..."},
                          {"i":1,"t_sec":190,"speaker":"REP","text":"Pharmacy, this is Dana."},
                          {"i":2,"t_sec":194,"speaker":"AGENT","text":"Hi Dana, this is Jo calling from Forus on behalf of Dr. Ruiz's office..."}]}
```
Speakers: `AGENT | REP | IVR`. `t_sec` non-decreasing and ≤ duration_seconds. If `connected` is
false: **no REP turns**, ≤ 4 turns total, IVR/ring only.

### 5.4 `forms.jsonl`
```json
{"call_id":"...","agent_id":"AGT-03","case_ref":"RX-32827","call_type":"transfer_confirm_call",
 "spoke_with_rep":true,"rep_name":"Dana","reference_number":null,
 "outcome_status":"received","next_action":"Follow up on PA","next_follow_up_date":"2024-08-06",
 "notes":"Rx received 8/4, PA needed before fill."}
```
`outcome_status` enums by call type:
- `pa_plan_call`: `approved | denied | pending | not_on_file | no_answer`
- `patient_access_check`: `enrolled | pending_docs | pending_bv | denied | no_answer`
- `transfer_confirm_call`: `received | not_received | filled | needs_pa | no_answer`

### 5.5 `labels.csv`
`call_id, code, severity, planted_by` where `planted_by ∈ {profile, global, derived}`.

### 5.6 Judge output (`qa/schemas.py`, pydantic)
```python
class Evidence(BaseModel):
    turn: int | None          # None only when evidence_kind == "absence"
    quote: str | None         # verbatim from that turn
    evidence_kind: Literal["quote","absence"]  # absence = "no REP turns", "never asked X"
class ChecklistItem(BaseModel):
    item_id: str; result: Literal["pass","fail","na"]; evidence: Evidence | None
class Defect(BaseModel):
    code: str; severity: Literal["critical","major","minor"]
    evidence: Evidence; confidence: float; rationale: str   # rationale ≤ 25 words
class FormCheck(BaseModel):
    field: str; form_value: str | None; transcript_value: str | None
    match: Literal["match","mismatch","unverifiable"]
class JudgeResult(BaseModel):
    call_id: str; checklist: list[ChecklistItem]; defects: list[Defect]
    form_check: list[FormCheck]; coaching_note: str; needs_human_review: bool
    model: str; prompt_version: str; evidence_failures: int = 0
```

### 5.7 `out/results.jsonl` (merged, one per call)
JudgeResult fields + `rule_flags:[{code,severity,reason}]` + `merged_defects:[{code,severity,source: rules|judge|both, quote, confidence}]` + `score` + `agent_id` + `call_type` + `duration_seconds`.
`out/scored_calls.csv` = flat version (one row per call) for the app.

---

## 6. Defect taxonomy, scoring, checklists (→ `qa/rubric.yaml`)

Ten codes in the prototype. Detector = the cheapest layer that catches it reliably.

| code | severity | detector | rule / judge definition |
| --- | --- | --- | --- |
| `FAB_CONTACT` | critical | rules + judge | form `spoke_with_rep` true (or has rep_name/reference_number) but `duration < 30` or transcript has no REP turns |
| `STATUS_MISMATCH` | critical | judge | form `outcome_status` contradicts what the REP said (form_check on outcome_status = mismatch) |
| `OUTCOME_VS_RX` | critical | rules | form `outcome_status` contradicts case `truth` as of call_date (explicit mapping table, e.g. form `received` but `received_at` null or after call_date; form `approved` but truth `denied/pending`) |
| `PHI_DISCLOSURE` | critical | judge | agent volunteers full DOB / address / diagnosis / member ID before the rep asks to verify, or shares another patient's details |
| `UNPROFESSIONAL` | major | judge | rude, sarcastic, interrupts, sighs audibly ("just check again"), hangs up on rep |
| `MISSED_NEXT_STEP` | major | judge | a required checklist item was available and not asked (e.g. denial → no appeal deadline asked) |
| `MISSING_REF` | major | rules + judge | rules: connected PA/program call, `reference_number` empty. judge: REP offered a reference # / name and agent did not capture it |
| `HOLD_OVERRUN` | minor | rules | `duration_seconds > hold_limit + 60` |
| `REDUNDANT_CALL` | minor | rules | case `resolved_before_call` true, or same `case_ref` already called earlier that day |
| `AFTER_HOURS` | minor | rules | `et_hour >= 19 or et_hour < 8` |

**Score** = `max(0, 100 − 40·criticals − 15·majors − 5·minors)`. Any critical ⇒ `needs_human_review`.
Agent scorecards show **mean score** *and* **criticals per 100 calls** side by side.

**Checklists (judge, pass/fail/na with evidence):**
- `all`: `ID_SELF` identifies self + Forus + on whose behalf · `MIN_PHI` gives only identifiers the rep asks for · `RIGHT_PATIENT` confirms patient · `CAPTURE_REF` gets rep name + reference # · `READBACK` reads outcome back · `PROFESSIONAL`
- `pa_plan_call`: `PA_STATUS` · `PA_PENDING_DATE` (if pending) · `PA_DENIAL_INFO` reason + appeal deadline + peer-to-peer (if denied) · `PA_AUTH` auth # + effective dates (if approved)
- `patient_access_check`: `ENR_STATUS` · `ENR_MISSING_DOCS` · `ENR_BV_RESULT` · `ENR_BRIDGE`
- `transfer_confirm_call`: `TX_RECEIVED` + pharmacy Rx # · `TX_BLOCKERS` stock / fill ETA / PA / insurance reject · `TX_RESEND` if not received: confirm NPI/fax, request re-send

---

## 7. Steps — must-haves (~65 min)

### Step 0 — Scaffold (5 min)
- Create layout in §4, `requirements.txt`, `.env.example` (`ANTHROPIC_API_KEY`, `GEN_MODEL`,
  `JUDGE_MODEL`), `Makefile`, empty `NOTES.md`, `qa/rubric.yaml` encoding §6 verbatim,
  `qa/schemas.py` from §5.6.
- Install requirements. Load `.env`, call `client.models.list()`, pick the current Haiku for
  `GEN_MODEL` and current Sonnet for `JUDGE_MODEL`, write exact IDs into `.env`.
- **Check:** `python -c "import yaml;r=yaml.safe_load(open('qa/rubric.yaml'));print(len(r['codes']),'codes');print(list(r['checklists']))"` and a 1-token API call succeeds.

### Step 1 — Synthetic generator `gen/generate.py` (15 min; start full run by minute 20)
Args: `--n-agents 8 --calls-per-agent 10 --seed 42 --limit N --no-llm --force`.

1. **Load** both sheets. Compute per-agent: volume, first call date, share < 20 s, share > hold
   limit. `et_hour = (hour − 4) mod 24`.
2. **Pick 8 agents → hidden profiles** (`agent_profiles.csv`), by stats not names:
   - 2 × `logs_without_connecting`: tenured (first call < Aug 20), volume ≥ 200, highest under-20 s share
   - 2 × `ramping`: first call ≥ Aug 20, highest volume
   - 1 × `curt`, 1 × `phi_oversharer`: next two tenured by volume
   - 2 × `solid`: next two tenured by volume
   Map to `AGT-01..08` in that order. Drop `email` everywhere after this.
3. **Sample 10 calls per agent** (80 total). Within each agent prefer: ≥ 2 calls < 20 s (if any),
   ≥ 2 calls > hold limit, ≥ 1 call at et_hour ≥ 19, type mix ~ 5 transfer / 3 PA / 2 program
   where the agent has them (fill from transfer otherwise). Fill the rest randomly.
   `connected = duration_seconds >= 30` (a 5% random exception: connected but very short = wrong number / immediate "closed" message → still connected=false; keep it simple: connected = duration ≥ 30).
4. **Cases** (`cases.jsonl`): transfer calls → pick a real `erx_id` of matching pharmacy type
   (`related_to_specialty_transfer==1` → specialty else retail), `transferred_on` within 0–7 days
   before `call_date`; `target_pharmacy = destination_name`; drug by type (specialty: Dupixent,
   Skyrizi, Rinvoq, Taltz, Humira; retail: atorvastatin, sertraline, metformin, lisinopril,
   albuterol). Fake patient via a small name list + random DOB 1950–2000. Truth timeline
   consistent with call_date. PA / program calls → synthetic `PA-xxxx` / `ENR-xxxx` with
   `plan`/`program = destination_name`. 10% of transfer cases `resolved_before_call = true`.
   If two sampled calls share agent + destination + date, give them the **same** `case_ref`
   (this creates natural REDUNDANT_CALL cases).
5. **Plant defects → `labels.csv`** (`rng` seeded):
   - `logs_without_connecting`: on its **non-connected** calls, `FAB_CONTACT` with p = 0.7
     (form claims spoke_with_rep, invented rep_name + reference_number, plausible positive status)
   - `ramping`: on connected PA/program calls, `MISSED_NEXT_STEP` p = 0.4, `MISSING_REF` p = 0.3
   - `curt`: connected calls, `UNPROFESSIONAL` p = 0.4
   - `phi_oversharer`: connected calls, `PHI_DISCLOSURE` p = 0.4
   - `solid`: any connected call, one random major/minor with p = 0.08
   - **global**: `STATUS_MISMATCH` on exactly 4 connected calls (weighted toward non-solid agents)
   - **derived** (compute, don't randomise): `HOLD_OVERRUN`, `AFTER_HOURS`, `REDUNDANT_CALL`
     (resolved_before_call or 2nd+ call to same case_ref same day), and `OUTCOME_VS_RX` whenever
     the finished form's `outcome_status` contradicts truth as of call_date (this will co-occur with
     FAB_CONTACT / STATUS_MISMATCH — label both codes).
   - Guarantee **≥ 25% of calls have zero labels** (drop planted defects from random calls if needed).
   - Print a count table by code and by agent.
6. **Scenario card per call** (`scenarios.jsonl`): call_type, destination_name, duration budget
   split (IVR s / hold s / talk s — for connected calls IVR 20–60, hold = remainder − talk, talk
   90–240 clipped to duration), case facts, `connected`, what the REP truthfully says (from `truth`),
   and for each judge-detectable planted defect a **behavioural hint** (never the code name):
   - PHI: "agent volunteers the patient's full DOB, home address and diagnosis before being asked to verify"
   - UNPROFESSIONAL: "agent sighs audibly, interrupts the rep twice, says 'this is the third time I'm calling, just check again'"
   - MISSED_NEXT_STEP: "rep says the PA was denied; agent thanks them and ends the call without asking for the denial reason or appeal deadline"
   - MISSING_REF: "rep offers a reference number near the end; agent says 'okay thanks' and does not repeat or confirm it"
   - STATUS_MISMATCH / OUTCOME_VS_RX: transcript is truthful (rep states the true status); the *form* will disagree — no transcript hint needed
   - FAB_CONTACT: connected=false → IVR/ring-only transcript
7. **Render transcript** with `GEN_MODEL` (temp 0.8) from the scenario card via forced tool
   `submit_transcript` returning `turns`. System prompt: realistic US healthcare phone call;
   agent is an offshore BPO caller for Forus calling on behalf of a prescriber's office; rep is a
   plan / hub / pharmacy employee; IVR lines realistic for the destination; numbers of turns
   proportional to talk seconds (~1 turn per 8–10 s); clean, non-cartoonish; **never name or
   allude to a quality defect**; if `connected` is false produce only IVR/ring lines.
   Validate: speakers ∈ set, `t_sec` monotone ≤ duration, no REP turns when not connected. Retry
   once on validation failure, then fall back to template. `--no-llm` = templates only
   (a few canned lines per phase; good enough to run the pipeline).
8. **Build form deterministically** from truth + planted defects: default = truthful form
   (outcome = truth status as the rep would state it, rep_name from transcript REP intro if any,
   reference_number `REF-######` for connected PA/program calls). Then apply: FAB_CONTACT →
   spoke_with_rep true + invented rep + ref + positive status; STATUS_MISMATCH → outcome flipped
   to a *different* plausible status; MISSING_REF → reference_number null; not connected and
   no FAB → `spoke_with_rep false, outcome no_answer, notes "IVR / no answer"`.
9. Write all files. Print summary.

- **Check:** `python -m gen.generate --limit 6` → read two transcripts by eye (one connected, one
  not) and the label table. Then start the full run in background:
  `nohup python -m gen.generate > out/gen.log 2>&1 &` → continue to Step 2.

### Step 2 — Rules engine `qa/rules.py` (8 min)
- One function per rules-detectable code in §6, each returns `{call_id, code, severity, reason}`
  with a human-readable reason (e.g. `"spoke_with_rep=True but duration=11s"`).
- `OUTCOME_VS_RX` uses an explicit mapping table `form_outcome → predicate(truth, call_date)`; put
  it in a dict at module top so it's inspectable.
- Writes `out/rule_flags.csv`. `--validate` flag prints precision/recall per code vs `labels.csv`.
- **Check:** `python -m qa.rules --validate` (works on whatever the generator has finished — it
  reads forms/cases/sample; rerun after gen completes).

### Step 3 — LLM judge + pipeline (14 min)
`qa/judge.py`:
- `build_prompt(call, transcript, form, case_public, rubric)` where `case_public` = call_type,
  destination_name, patient initials, drug, prescriber — **NOT** the truth timeline (keeps the
  judge independent of the rules layer; the app shows truth to humans).
- System prompt (put `PROMPT_VERSION = "v1"` in code): *You are a QA auditor for outbound
  pharmacy, insurance-plan and manufacturer-program calls placed on behalf of Forus. Judge only
  on transcript evidence. For every fail and every defect cite the turn index and a verbatim quote
  from that turn; if the evidence is an absence (nothing was said / no rep reached) use
  evidence_kind "absence" with turn null. Do not penalise the agent for the counterparty's
  behaviour, IVR outages or hold time. Use only these defect codes: FAB_CONTACT,
  STATUS_MISMATCH, PHI_DISCLOSURE, UNPROFESSIONAL, MISSED_NEXT_STEP, MISSING_REF. Compare the
  agent's form to the transcript field by field (spoke_with_rep, rep_name, reference_number,
  outcome_status, next_action) and report match / mismatch / unverifiable. Write a two-sentence
  coaching note addressed to the agent.* Then the call-type checklist from rubric.yaml, the
  numbered transcript, the form JSON, and case_public.
- Forced tool `submit_audit` with `JudgeResult` schema (minus model/prompt_version, which you fill).
  Temperature 0.
- **Quote verification:** for each defect / failed checklist item with `evidence_kind == "quote"`,
  normalise whitespace + case and check the quote is a substring of the cited turn (fallback: any
  turn). Failures are dropped from `defects` and counted in `evidence_failures`.
- Async, semaphore 8, retries, cache by call_id in `out/judge_raw.jsonl`. `--limit`, `--force`,
  `--no-llm` (emits an empty-defect result so the pipeline still runs).

`qa/pipeline.py`:
- Run rules (import, don't shell out), run judge, **merge** by `(call_id, code)`: `source` =
  `rules | judge | both`; quote/confidence from judge when present. Compute `score`,
  `needs_human_review`. Write `out/results.jsonl` and `out/scored_calls.csv` (one row per call:
  call_id, agent_id, call_type, destination_name, duration_seconds, et_hour, score,
  n_critical, n_major, n_minor, needs_human_review, codes (semicolon-joined), form_accuracy =
  share of form_check fields == match).
- **Check:** wait for gen to finish (`tail out/gen.log`), then
  `python -m qa.pipeline --limit 3` and pretty-print one result; confirm quotes are real and the
  form_check makes sense. Then `nohup python -m qa.pipeline > out/judge.log 2>&1 &` → Step 4.

### Step 4 — Validation `qa/validate.py` (6 min)
- Join `merged_defects` to `labels.csv` on `(call_id, code)`. Per code, separately for `rules`,
  `judge`, `merged`: TP, FP, FN, precision, recall, F1 (handle zero denominators).
- Headline metrics: **critical recall** (all critical codes pooled), **clean-call false-positive
  rate** (share of zero-label calls with ≥ 1 major/critical flag), **evidence validity** =
  1 − evidence_failures / total judge defects, **layer agreement** = share of merged defects with
  source `both` among codes both layers detect.
- Agent level: table `agent_id, profile, calls, mean_score, criticals_per_100` + Spearman rho
  between a profile badness rank (`solid`=0 < `ramping`=1 < `curt`=`phi_oversharer`=2 <
  `logs_without_connecting`=3) and mean score.
- Optional: if `out/human_labels.csv` exists, judge-vs-human agreement per code.
- Writes `out/validation.csv`; prints a table. Runs on partial results (whatever the judge has done).
- **Check:** `python -m qa.validate` prints sensible numbers on the calls judged so far.

### Step 5 — Streamlit app `app.py` (14 min) — three tabs
Wide layout, `st.cache_data` loaders keyed on file mtime, banner:
*"SYNTHETIC transcripts and forms, seeded from the real Aug 2024 TaskUs call log (real
durations, destinations, Rx IDs; agents pseudonymised). Agent profiles are illustrative."*

**Overview**
- `st.metric` row: calls scored · % calls with a critical · mean score · form accuracy · evidence validity.
- Plotly bar: defects by code, coloured by severity (critical red, major amber, minor grey),
  sorted by count. Second small bar: critical rate by call type.
- One line under the charts: layer agreement %.

**Call Inspector**
- `st.selectbox` of calls sorted by score ascending, label like
  `72 · AGT-03 · transfer_confirm_call · CVS Specialty Pharmacy · 412s · STATUS_MISMATCH`.
- Left column: metadata chips (agent, type, destination, duration, ET time, connected), then the
  transcript as chat bubbles (AGENT right-aligned, REP left, IVR grey italic). Turns cited by any
  defect get a coloured left border + the code as a small tag (red critical / amber major).
- Right column:
  1. **Three-way diff table** with columns **"Agent logged (form)" · "Transcript supports
     (auto-draft)" · "System of record (case truth)"**, one row per form field; mismatches
     highlighted. The third column comes from `cases.jsonl` truth (humans may see it).
  2. Defect list: code, severity badge, source badge (rules / judge / both), quote, confidence.
  3. Checklist results (pass/fail/na) as a compact table.
  4. Coaching note.
  5. Button **"Re-run judge live"** → calls `judge.judge_call(..., force=True)` and re-renders
     (show a spinner; catch and display API errors).
- Expander at bottom: raw JudgeResult JSON.

**Validation**
- Table from `out/validation.csv` (merged view by default, toggle rules/judge), critical codes
  highlighted; `st.metric` row for critical recall, clean-call FP rate, evidence validity, Spearman rho.
- Agent-level table (profile revealed here, labelled "hidden ground truth").
- One caption: *"Planted-defect recall is a unit test, not proof: transcripts are cleaner than real
  ASR and rule-derived codes are true by construction. Next: 300-call human golden set."*

- **Check:** `streamlit run app.py`, click all three tabs, open the lowest-scoring call, press
  Re-run judge live once.

### Step 6 — Freeze the demo (3 min)
- Confirm `out/results.jsonl` covers all 80 calls; rerun `python -m qa.validate`.
- `git init && git add -A && git commit -m "demo snapshot"` **including `out/`** so the demo works
  with no API / Wi-Fi.
- Write `NOTES.md`: what works, what's stubbed, choices made, counts by code, validation headline.
- **STOP HERE and report** (see kick-off prompt). Do not start stretch until told.

---

## 8. Stretch (only after go-ahead; ~25 min)

### Step 7 — Agents tab + coaching cards (10 min)
- Table per `agent_id`: calls, mean score, criticals / 100 calls, top defect, % needing review,
  form accuracy; sorted by criticals desc. Profile hidden by default (expander "ground truth
  (synthetic)").
- Select an agent → defect mix bar, 3 worst calls (button sets `st.session_state.selected_call`
  and switches to Inspector), and **"Draft coaching card"** button → `qa/coach.py`: one Sonnet
  call with the agent's merged defects + quotes → tool `submit_card` returning
  `{strengths:[2], fix:[{behaviour, quote, why_it_matters}×2], practice_line}`. Cache per agent.

### Step 8 — Review queue (8 min)
- Tab listing calls with `needs_human_review`, criticals first then lowest confidence.
- Per defect: quote + buttons **Confirm / Reject / Change code** (selectbox) → append row to
  `out/human_labels.csv` (`call_id, code, verdict, new_code, reviewer, ts`). Validation tab reads it
  and shows judge-vs-human agreement.

### Step 9 — Stability (7 min)
- `python -m qa.validate --stability 15 --runs 3`: re-judge 15 random calls 3× with `force`,
  report share of identical `(call, code)` decisions and score std-dev. Print one line for the debrief.

---

## 9. Fallbacks (apply without asking)
- Generator API trouble by minute 25 → `--no-llm` for the remaining calls, note it in NOTES.md.
- Judge not finished by minute 55 → build the app on partial results; validation runs on what exists.
- Anything in Step 5 taking > 15 min → ship Overview + Inspector; Validation can be a `st.dataframe`
  of `validation.csv` with no styling.
- Never spend time on: auth, deployment, tests beyond the Checks, CSS polish, docstrings.

---

## 10. Kick-off prompt (paste this into Claude Code)

```
Read CLAUDE.md in full before doing anything. Execute Steps 0–6 in order. After each step run
its Check and fix failures before moving on. Start the long runs (Step 1 full generation,
Step 3 full judge run) in the background with nohup and keep building the next step while they
run; poll the logs before you need their output. Follow §9 fallbacks on your own. Don't ask me
questions unless you are truly blocked — make a reasonable choice and record it in NOTES.md.
When Step 6 is done, STOP and give me a status in ≤ 12 lines: what works end to end, what is
stubbed, planted-defect counts by code, validation headline numbers (critical recall,
clean-call FP rate, evidence validity), and the exact command to launch the app. Wait for my
go-ahead before starting Steps 7–9.
```