"""
Interview-level LLM analyzer.

Responsible ONLY for qualitative synthesis across all five question
responses.  No numerical scores are produced here — those come from
interview/scoring.py.

The prompt asks the LLM to:
  - identify 3-5 observable strengths
  - identify 3-5 areas to improve
  - give 3-5 concrete recommendations
  - note cross-question behavioural trends (using cautious, observable language)

Cautious language guidelines:
  - AVOID: "you are anxious", "you lied", "you are unconfident"
  - PREFER: "a higher blink rate was observed", "confidence indicators dipped",
             "filler words appeared more frequently"
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interview.service import InterviewSession

logger = logging.getLogger(__name__)


_ANALYSIS_PROMPT = """
You are an expert interview coach reviewing a candidate's 5-question mock interview.

Below is a summary of each question and the measurable signals from the candidate's response.
Use ONLY the data provided. Do not invent metrics. Use cautious, observable language for
behavioural observations (e.g. "higher blink rate was observed" rather than "candidate was anxious").

Interview Summary:
{interview_summary}

Return ONLY a valid JSON object with EXACTLY these keys (no markdown, no extra text):
{{
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "areas_to_improve": ["<area 1>", "<area 2>", "<area 3>"],
  "recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"],
  "trends": [
    {{
      "metric": "<observable metric name>",
      "observation": "<one sentence cautious observation about the pattern across questions>"
    }}
  ]
}}

Rules:
- strengths: 3-5 items, specific and based on data provided.
- areas_to_improve: 3-5 items, constructive and based on data.
- recommendations: 3-5 actionable next steps.
- trends: 2-5 items describing cross-question patterns (confidence changes, speech pace, etc.).
- All values must be strings. No nested objects except the trend structure shown above.
- Return ONLY the JSON. No preamble, no commentary.
"""


def _build_summary(session: "InterviewSession") -> str:
    """Build a text summary of the session for the LLM prompt."""
    lines: list[str] = []
    for i, result in enumerate(session.question_results, start=1):
        q = result["question"]
        score = result.get("score_data", {})
        va = result.get("video_analysis", {})
        sa = result.get("speech_analysis", {})
        ca = result.get("content_analysis", {})

        pronunciation_count = len(sa.get("pronunciation_issues", []))
        emotion_top = sorted(
            va.get("emotion_percentages", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:2]
        emotion_str = ", ".join(f"{e} {round(v, 1)}%" for e, v in emotion_top) or "N/A"

        lines.append(
            f"Q{i}: {q['question_text']}\n"
            f"  Transcript word count: {ca.get('word_count', 0)}\n"
            f"  Confidence score: {score.get('confidence_score', 'N/A')}\n"
            f"  Body language score: {score.get('body_language_score', 'N/A')}\n"
            f"  Speech score: {score.get('speech_score', 'N/A')}\n"
            f"  Content score: {score.get('content_score', 'N/A')}\n"
            f"  Question score: {score.get('question_score', 'N/A')}\n"
            f"  Blink rate category: {va.get('blink_rate_category', 'N/A')}\n"
            f"  Top emotions: {emotion_str}\n"
            f"  Pronunciation issues: {pronunciation_count}\n"
            f"  Tone: {ca.get('tone', {}).get('appropriateness', 'N/A')}\n"
            f"  Clarity: {ca.get('clarity', 'N/A')}\n"
            f"  Relevance: {ca.get('relevance', 'N/A')}\n"
            f"  Answer quality: {ca.get('answer_quality', 'N/A')}\n"
        )

    return "\n".join(lines)


def _safe_parse(text: str) -> dict:
    """Attempt to parse LLM JSON; fall back to empty structure on failure."""
    # Strip markdown fences if present
    clean = text.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    clean = clean.rstrip("```").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.error("InterviewAnalyzer: LLM returned invalid JSON. Raw: %s", text[:400])
        return {}


_FALLBACK_RESULT = {
    "strengths": ["Completed all five interview questions."],
    "areas_to_improve": ["Review your responses for clarity and conciseness."],
    "recommendations": ["Practice answering using the STAR method (Situation, Task, Action, Result)."],
    "trends": [
        {
            "metric": "completion",
            "observation": "The candidate completed all five questions in the interview session.",
        }
    ],
}


class InterviewAnalyzer:
    """
    Generates qualitative overall-interview feedback using an LLM.

    Parameters
    ----------
    llm:
        A LangChain-compatible LLM instance (OllamaLLM or similar).
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def analyze(self, session: "InterviewSession") -> dict:
        """
        Run cross-question qualitative analysis.

        Returns a dict with keys:
            strengths, areas_to_improve, recommendations, trends
        """
        if not session.question_results:
            return _FALLBACK_RESULT

        summary = _build_summary(session)
        prompt = _ANALYSIS_PROMPT.format(interview_summary=summary)

        try:
            raw: str = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error("InterviewAnalyzer: LLM call failed: %s", exc)
            return _FALLBACK_RESULT

        data = _safe_parse(raw)
        if not data:
            return _FALLBACK_RESULT

        # Validate and sanitise expected fields
        def _str_list(key: str, default: list) -> list:
            val = data.get(key, default)
            if isinstance(val, list):
                return [str(item) for item in val if item]
            return default

        trends_raw = data.get("trends", [])
        trends: list[dict] = []
        if isinstance(trends_raw, list):
            for t in trends_raw:
                if isinstance(t, dict):
                    trends.append(
                        {
                            "metric": str(t.get("metric", "")),
                            "observation": str(t.get("observation", "")),
                        }
                    )

        return {
            "strengths": _str_list("strengths", _FALLBACK_RESULT["strengths"]),
            "areas_to_improve": _str_list("areas_to_improve", _FALLBACK_RESULT["areas_to_improve"]),
            "recommendations": _str_list("recommendations", _FALLBACK_RESULT["recommendations"]),
            "trends": trends or _FALLBACK_RESULT["trends"],
        }
