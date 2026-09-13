from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing response sub-models (unchanged — keep /upload compatible)
# ---------------------------------------------------------------------------

class ToneModel(BaseModel):
    score: float = Field(..., description="Sentiment score -1 (negative) to 1 (positive)")
    appropriateness: str


class GrammarIssue(BaseModel):
    issue: str
    correction: str


class PronunciationIssue(BaseModel):
    word: str
    correction: str
    phonetic: str = ""


class AnswerStatus(str, Enum):
    VALID = "VALID"
    NO_SPEECH = "NO_SPEECH"
    INSUFFICIENT_RESPONSE = "INSUFFICIENT_RESPONSE"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"


class AnswerValidity(BaseModel):
    status: AnswerStatus
    speech_detected: bool
    speech_duration: float = 0.0
    transcript_available: bool
    transcript_word_count: int
    meaningful_answer: bool
    reason: str | None = None


class ContentAnalysisModel(BaseModel):
    word_count: int
    clarity: str
    engagement: str
    structure: str
    grammar: list[GrammarIssue] = Field(default_factory=list)
    tone: ToneModel
    relevance: str
    answer_quality: str = ""
    suggestions: str


class ConfidenceAnalysisModel(BaseModel):
    confidence_score: float = Field(..., description="Combined 0-100 confidence score")
    frames_analyzed: int
    total_frames: int
    blinks_per_minute: float = 0.0
    blink_rate_category: str = ""
    emotion_percentages: dict[str, float] = Field(default_factory=dict)
    frame_details: list[dict[str, Any]] = Field(default_factory=list)



class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Session / Interview models
# ---------------------------------------------------------------------------

class QuestionItemSchema(BaseModel):
    """A single question as returned to the frontend."""
    question_id: str
    question_number: int
    question_text: str
    category: str = ""
    dimension: str = ""


class CreateSessionResponse(BaseModel):
    """Response from POST /interviews."""
    session_id: str
    status: str
    questions: list[QuestionItemSchema]


class SessionStatusResponse(BaseModel):
    """Lightweight status response."""
    session_id: str
    status: str
    questions_answered: int
    total_questions: int


# ---------------------------------------------------------------------------
# Per-question result schema
# ---------------------------------------------------------------------------

class QuestionScoreSchema(BaseModel):
    confidence_score: float
    body_language_score: float
    speech_score: float
    content_score: float
    question_score: float


class QuestionResultResponse(BaseModel):
    """Response from POST /interviews/{sid}/questions/{qid}/response."""
    session_id: str
    question_id: str
    question_number: int
    question_text: str

    score: QuestionScoreSchema

    transcription: str
    confidence_analysis: ConfidenceAnalysisModel
    speech_analysis: list[PronunciationIssue]
    content_analysis: ContentAnalysisModel
    answer_validity: AnswerValidity | None = None

    # Convenience flag so the frontend knows what to do next
    is_last_question: bool
    next_question: QuestionItemSchema | None = None


class QuestionProcessingStatusResponse(BaseModel):
    """Response returned immediately when the video is queued for background processing."""
    status: str = Field(default="processing")
    message: str = Field(default="Video uploaded successfully and is being processed.")


class QuestionPollResponse(BaseModel):
    """Response returned when polling for the result of a specific question."""
    status: str  # "processing", "completed", "failed"
    result: QuestionResultResponse | None = None


# ---------------------------------------------------------------------------
# Overall interview report schema
# ---------------------------------------------------------------------------

class TrendItemSchema(BaseModel):
    metric: str
    observation: str


class OverallAnalysisSchema(BaseModel):
    overall_score: float

    communication_score: float
    confidence_score: float
    speech_score: float
    body_language_score: float
    content_score: float

    strengths: list[str] = Field(default_factory=list)
    areas_to_improve: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    trends: list[TrendItemSchema] = Field(default_factory=list)

    # Per-question breakdown for sparkline / bar charts
    question_scores: list[float] = Field(default_factory=list)
    confidence_trend: list[float] = Field(default_factory=list)
    speech_trend: list[float] = Field(default_factory=list)
    content_trend: list[float] = Field(default_factory=list)


class QuestionDetailSchema(BaseModel):
    """Full detail for one question in the final report."""
    question_id: str
    question_number: int
    question_text: str

    score: float
    confidence_score: float
    body_language_score: float
    speech_score: float
    content_score: float

    transcription: str
    confidence_analysis: ConfidenceAnalysisModel
    speech_analysis: list[PronunciationIssue]
    content_analysis: ContentAnalysisModel
    answer_validity: AnswerValidity | None = None


class InterviewReportResponse(BaseModel):
    """Response from GET /interviews/{session_id}."""
    session_id: str
    status: str

    overall_analysis: OverallAnalysisSchema | None = None
    questions: list[QuestionDetailSchema] = Field(default_factory=list)
