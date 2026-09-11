from typing import Any
from pydantic import BaseModel, Field


# Response sub-models

class ToneModel(BaseModel):
    score: float = Field(default=0.0, description="Sentiment score -1 (negative) to 1 (positive)")
    appropriateness: str = "Neutral"


class GrammarIssue(BaseModel):
    issue: str = ""
    correction: str = ""


class PronunciationIssue(BaseModel):
    word: str = ""
    correction: str = ""
    phonetic: str = ""


class ContentAnalysisModel(BaseModel):
    word_count: int = 0
    relevance_score: float = Field(default=0.0, description="0-100 score on how directly and thoroughly the answer addressed the specific question asked")
    content_score: float = Field(default=0.0, description="0-100 overall score for content quality, structure, clarity, and depth")
    clarity: str = ""
    engagement: str = ""
    structure: str = ""
    grammar: list[GrammarIssue] = []
    tone: ToneModel = Field(default_factory=ToneModel)
    relevance: str = ""
    answer_quality: str = ""
    suggestions: str = ""


class ConfidenceAnalysisModel(BaseModel):
    confidence_score: float = Field(default=0.0, description="Combined 0-100 confidence score")
    frames_analyzed: int = 0
    total_frames: int = 0
    blinks_per_minute: float = 0.0
    blink_rate_category: str = ""
    speaking_rate_wpm: float = 0.0
    pacing_category: str = "optimal"
    emotion_percentages: dict[str, float] = {}
    frame_details: list[dict[str, Any]] = []


# Top-level response

class AnalysisResponse(BaseModel):
    video_id: str
    transcription: str = ""
    confidence_analysis: ConfidenceAnalysisModel = Field(default_factory=ConfidenceAnalysisModel)
    speech_analysis: list[PronunciationIssue] = []
    content_analysis: ContentAnalysisModel = Field(default_factory=ContentAnalysisModel)


class ErrorResponse(BaseModel):
    error: str