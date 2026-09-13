"""
Centralized scoring logic for the interview system.

All numerical scores are derived from measurable analysis outputs —
not from LLM text generation.  The LLM is used only for qualitative
synthesis in the interview-level analyzer.

Scoring Methodology
-------------------
Each question response produces five dimension scores (0-100):

  Confidence (25% of question_score)
    Source: video_analysis.confidence_score (already 0-100 from VideoAnalyzer).
    The VideoAnalyzer combines emotion weights and blink-rate contribution.

  Body Language (15% of question_score)
    Source: video_analysis.blink_rate_category + video_analysis.emotion_percentages.
    - Blink rate contributes a bonus/penalty:
        very_low / low / normal   → +0 penalty (neutral)
        high                      → -5
        very_high                 → -15
    - Positive emotion ratio (happy + neutral + surprise) contributes
      proportionally up to +20 bonus.
    - Base: 60; clamped 0-100.

  Speech (20% of question_score)
    Source: speech_analysis.pronunciation_issues count.
    - Base score: 100
    - -8 per pronunciation / disfluency issue (floors at 20).
    - Clamped 0-100.

  Content (40% of question_score)
    Source: content_analysis.tone.score + qualitative field presence.
    - Tone score is on -1 to +1; mapped to 0-100 as ((score+1)/2)*100.
    - If clarity/engagement/structure/relevance are all present (not "N/A"),
      a +5 completeness bonus is applied.
    - Clamped 0-100.

question_score = weighted average of the four dimensions.

IMPORTANT: These weights reflect the relative importance of each signal
in a behavioural interview context.  They are not scientifically validated
and should be treated as reasonable heuristics for an MVP.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuestionScore:
    """Scores for a single question response."""
    question_id: str
    question_number: int

    confidence_score: float      # 0-100
    body_language_score: float   # 0-100
    speech_score: float          # 0-100
    content_score: float         # 0-100
    question_score: float        # 0-100  weighted composite


@dataclass
class OverallScore:
    """Aggregated scores across all question responses."""
    overall_score: float

    confidence_score: float
    body_language_score: float
    speech_score: float
    content_score: float
    communication_score: float   # alias for content; kept for report clarity

    question_scores: list[float] = field(default_factory=list)

    # Cross-question trend data (metric → list of values per question)
    trends: dict[str, list[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "confidence":    0.25,
    "body_language": 0.15,
    "speech":        0.20,
    "content":       0.40,
}

_BLINK_PENALTY: dict[str, float] = {
    "very_low": 0,
    "low":      0,
    "normal":   0,
    "high":     -5,
    "very_high": -15,
}


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def _score_confidence(video_analysis: dict) -> float:
    """Directly use the VideoAnalyzer confidence_score (0-100)."""
    raw = float(video_analysis.get("confidence_score", 50.0))
    return max(0.0, min(100.0, raw))


def _score_body_language(video_analysis: dict) -> float:
    """
    Compute body language score from blink category and emotion distribution.

    Base: 60
    Blink penalty: 0 to -15 (for high/very_high blink rate)
    Positive emotion bonus: 0 to +20 (proportional to happy+neutral+surprise share)
    Clamped to [0, 100].
    """
    base = 60.0

    blink_cat = video_analysis.get("blink_rate_category", "normal")
    penalty = _BLINK_PENALTY.get(blink_cat, 0.0)

    emotions = video_analysis.get("emotion_percentages", {})
    positive_pct = sum(
        emotions.get(e, 0.0) for e in ["happy", "neutral", "surprise"]
    )
    # positive_pct is already in [0, 100] (sum of percentage values)
    # cap the bonus so it doesn't overwhelm
    positive_bonus = (positive_pct / 100.0) * 20.0

    raw = base + penalty + positive_bonus
    return max(0.0, min(100.0, raw))


def _score_speech(speech_analysis: dict) -> float:
    """
    Penalise for pronunciation / disfluency issues.

    Base: 100
    -8 per issue, minimum floor of 20.
    Clamped to [0, 100].
    """
    issues = speech_analysis.get("pronunciation_issues", [])
    count = len(issues) if isinstance(issues, list) else 0
    raw = max(20.0, 100.0 - count * 8.0)
    return max(0.0, min(100.0, raw))


def _score_content(content_analysis: dict) -> float:
    """
    Map tone score and qualitative completeness to a 0-100 content score.

    Tone: -1 to +1  →  0 to 100 via ((score+1)/2)*100
    Completeness bonus: +5 if all qualitative fields are non-empty and not N/A.
    Clamped to [0, 100].
    """
    tone = content_analysis.get("tone", {})
    tone_score = float(tone.get("score", 0.0))
    tone_mapped = ((tone_score + 1) / 2) * 100.0

    qualitative_fields = ["clarity", "engagement", "structure", "relevance"]
    all_present = all(
        content_analysis.get(f, "N/A") not in ("", "N/A", None)
        for f in qualitative_fields
    )
    completeness_bonus = 5.0 if all_present else 0.0

    raw = tone_mapped + completeness_bonus
    return max(0.0, min(100.0, raw))


def compute_question_score(
    question_id: str,
    question_number: int,
    video_analysis: dict,
    speech_analysis: dict,
    content_analysis: dict,
    answer_validity: dict | None = None,
) -> QuestionScore:
    """
    Compute the full score for a single question response.

    All inputs are the dict outputs from their respective analyzers
    (as stored in the LangGraph state / returned by the workflow).
    """
    conf  = _score_confidence(video_analysis)
    body  = _score_body_language(video_analysis)
    speech = _score_speech(speech_analysis)
    content = _score_content(content_analysis)

    # Hard Scoring Gate: Zero out speech/content and composite if no valid answer
    from api.schemas import AnswerStatus
    status = answer_validity.get("status") if answer_validity else AnswerStatus.VALID
    
    if status != AnswerStatus.VALID:
        speech = 0.0
        content = 0.0
        composite = 0.0
    else:
        composite = (
            conf    * _WEIGHTS["confidence"]    +
            body    * _WEIGHTS["body_language"] +
            speech  * _WEIGHTS["speech"]        +
            content * _WEIGHTS["content"]
        )

    return QuestionScore(
        question_id=question_id,
        question_number=question_number,
        confidence_score=round(conf, 1),
        body_language_score=round(body, 1),
        speech_score=round(speech, 1),
        content_score=round(content, 1),
        question_score=round(composite, 1),
    )


# ---------------------------------------------------------------------------
# Overall scoring
# ---------------------------------------------------------------------------

def compute_overall_score(question_scores: list[QuestionScore]) -> OverallScore:
    """
    Aggregate per-question scores into an interview-level OverallScore.

    Aggregation:
    - overall_score: simple mean of question_score values
    - dimension scores: simple mean of each dimension across questions
    - communication_score: alias for content_score (content quality = communication quality)
    - trends: per-question timeseries for each dimension
    """
    if not question_scores:
        return OverallScore(
            overall_score=0.0,
            confidence_score=0.0,
            body_language_score=0.0,
            speech_score=0.0,
            content_score=0.0,
            communication_score=0.0,
        )

    n = len(question_scores)

    def avg(values: list[float]) -> float:
        return round(sum(values) / n, 1)

    confs    = [q.confidence_score    for q in question_scores]
    bodies   = [q.body_language_score for q in question_scores]
    speeches = [q.speech_score        for q in question_scores]
    contents = [q.content_score       for q in question_scores]
    composites = [q.question_score    for q in question_scores]

    overall = avg(composites)
    content_avg = avg(contents)

    trends = {
        "confidence":    [round(v, 1) for v in confs],
        "body_language": [round(v, 1) for v in bodies],
        "speech":        [round(v, 1) for v in speeches],
        "content":       [round(v, 1) for v in contents],
        "question_score":[round(v, 1) for v in composites],
    }

    return OverallScore(
        overall_score=overall,
        confidence_score=avg(confs),
        body_language_score=avg(bodies),
        speech_score=avg(speeches),
        content_score=content_avg,
        communication_score=content_avg,   # alias
        question_scores=composites,
        trends=trends,
    )
