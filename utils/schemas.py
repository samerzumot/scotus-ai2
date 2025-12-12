from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from utils.scotus import BENCH_ORDER, JUSTICE_NAMES


VoteLabel = Literal["PETITIONER", "RESPONDENT", "UNCERTAIN"]
UploaderSide = Literal["PETITIONER", "RESPONDENT", "AMICUS", "UNKNOWN"]


class JusticeVote(BaseModel):
    justice_id: str
    justice_name: str = ""
    vote: VoteLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("justice_id")
    @classmethod
    def _known_justice(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in JUSTICE_NAMES:
            raise ValueError(f"Unknown justice_id: {v}")
        return v

    @model_validator(mode="after")
    def _fill_name(self):
        if not self.justice_name:
            self.justice_name = JUSTICE_NAMES.get(self.justice_id, self.justice_id)
        return self


class JusticeQuestion(BaseModel):
    justice_id: str
    justice_name: str = ""
    question: str
    what_it_tests: str = ""

    @field_validator("justice_id")
    @classmethod
    def _known_justice(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in JUSTICE_NAMES:
            raise ValueError(f"Unknown justice_id: {v}")
        return v

    @model_validator(mode="after")
    def _fill_name(self):
        if not self.justice_name:
            self.justice_name = JUSTICE_NAMES.get(self.justice_id, self.justice_id)
        return self


class RetrievedCaseRef(BaseModel):
    case_id: str
    case_name: str
    term: Optional[int] = None
    tags: List[str] = []
    outcome: Optional[str] = None


class OverallPrediction(BaseModel):
    predicted_winner: VoteLabel
    confidence: float = Field(ge=0.0, le=1.0)
    why: str = ""
    swing_justice: Optional[str] = None


class ModelInfo(BaseModel):
    provider: Literal["google", "huggingface", "fallback"]
    predict_model: str
    embed_model: Optional[str] = None
    retrieval_top_k: int = 0


class VoteQuestionPrediction(BaseModel):
    uploader_side: UploaderSide = "UNKNOWN"
    overall: OverallPrediction
    votes: List[JusticeVote]
    questions: List[JusticeQuestion]
    retrieved_cases: List[RetrievedCaseRef] = []
    model: ModelInfo

    @model_validator(mode="after")
    def _validate_coverage(self):
        vote_ids = [v.justice_id for v in self.votes]
        if set(vote_ids) != set(BENCH_ORDER):
            raise ValueError("votes must contain exactly 9 justices (all bench members).")
        q_ids = [q.justice_id for q in self.questions]
        if set(q_ids) != set(BENCH_ORDER):
            raise ValueError("questions must contain exactly 9 justices (all bench members).")
        return self


class QuestionBacktestMatch(BaseModel):
    predicted: str
    best_actual: str
    similarity: float = Field(ge=0.0, le=1.0)


class BacktestResult(BaseModel):
    transcript_url: str = ""
    transcript_found: bool = False
    transcript_auto_detected: bool = False
    questions_score_pct: int = Field(ge=0, le=100)
    matches: List[QuestionBacktestMatch] = []
    explanation: str = ""


