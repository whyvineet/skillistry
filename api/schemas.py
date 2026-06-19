from typing import Any
from pydantic import BaseModel, Field


# Response sub-models

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


class ContentAnalysisModel(BaseModel):
    word_count: int
    clarity: str
    engagement: str
    structure: str
    grammar: list[GrammarIssue] = []
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
    emotion_percentages: dict[str, float] = {}
    frame_details: list[dict[str, Any]] = []


# Top-level response

class AnalysisResponse(BaseModel):
    video_id: str
    transcription: str
    confidence_analysis: ConfidenceAnalysisModel
    speech_analysis: list[PronunciationIssue]
    content_analysis: ContentAnalysisModel


class ErrorResponse(BaseModel):
    error: str