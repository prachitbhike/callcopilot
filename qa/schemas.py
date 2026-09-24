from typing import Literal
from pydantic import BaseModel


class Evidence(BaseModel):
    turn: int | None = None  # None only when evidence_kind == "absence"
    quote: str | None = None  # verbatim from that turn
    evidence_kind: Literal["quote", "absence"]


class ChecklistItem(BaseModel):
    item_id: str
    result: Literal["pass", "fail", "na"]
    evidence: Evidence | None = None


class Defect(BaseModel):
    code: str
    severity: Literal["critical", "major", "minor"]
    evidence: Evidence
    confidence: float
    rationale: str  # <= 25 words


class FormCheck(BaseModel):
    field: str
    form_value: str | None = None
    transcript_value: str | None = None
    match: Literal["match", "mismatch", "unverifiable"]


class AuditSubmission(BaseModel):
    """What the model returns via the submit_audit tool."""
    checklist: list[ChecklistItem]
    defects: list[Defect]
    form_check: list[FormCheck]
    coaching_note: str
    needs_human_review: bool


class JudgeResult(AuditSubmission):
    call_id: str
    model: str
    prompt_version: str
    evidence_failures: int = 0
    guard_drops: int = 0            # placeholder / low-confidence defects removed in postprocess
    usage: dict | None = None       # {input_tokens, output_tokens} from the API response
    latency_s: float | None = None
