import logging
import json

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from utils.helpers import parse_llm_json

logger = logging.getLogger(__name__)


ANALYSIS_TEMPLATE = """
    You are an expert interview coach. Evaluate the following candidate response to the given
    interview question. Return ONLY a valid JSON object — no markdown fences, no commentary.

    Interview Question:
    {question}

    Candidate's Answer:
    {answer}

    Word Count: {word_count}

    Evaluate using the criteria below and return the result in the EXACT JSON schema shown.

    {{
        "word_count": <integer>,
        "relevance_score": <float between 0.0 and 100.0, evaluating how directly and completely the candidate answered the specific question asked>,
        "content_score": <float between 0.0 and 100.0, overall assessment of answer quality, depth, and clarity>,
        "clarity": "<one sentence assessment>",
        "engagement": "<one sentence assessment>",
        "structure": "<one sentence assessment>",
        "grammar": [
            {{
                "issue": "<incorrect phrase>",
                "correction": "<corrected version>"
            }}
        ],
        "tone": {{
            "score": <float between -1.0 and 1.0>,
            "appropriateness": "<one sentence>"
        }},
        "relevance": "<how well the answer addresses the question>",
        "answer_quality": "<overall assessment of the answer's content and depth>",
        "suggestions": "<3-5 specific, actionable improvements>"
    }}

    Scoring Rules:
    - "relevance_score": 0 = completely irrelevant/off-topic, 50 = partially answered, 100 = directly, thoroughly answered.
    - "content_score": 0-100 overall score on clarity, structure, professionalism, and depth.
    - "grammar" should be an empty array [] if no issues are found.
    - "tone.score": -1 = very negative/nervous, 0 = neutral, 1 = very positive/confident.
    - Only if the answer has fewer than 20 words, fill all string fields with "N/A",
    set relevance_score to 0.0, set content_score to 0.0, set tone.score to 0.0, and set suggestions to "Answer is too short to evaluate."
    - Return ONLY the JSON object. No extra text.
"""


class ContentAnalyzer:
    def __init__(self, ollama_model: str, ollama_base_url: str) -> None:
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        self.prompt = PromptTemplate(
            input_variables=["question", "answer", "word_count"],
            template=ANALYSIS_TEMPLATE,
        )
        self.chain = self.prompt | self.llm

    @staticmethod
    def _normalize_result(data: dict, word_count: int = 0) -> dict:
        suggestions = (
            data.get("suggestions")
            or data.get("improvement_suggestions")
            or data.get("improvements")
            or data.get("recommendations")
            or data.get("actionable_improvements")
            or data.get("feedback")
            or ""
        )
        if isinstance(suggestions, list):
            data["suggestions"] = "\n".join(f"- {item}" for item in suggestions)
        else:
            data["suggestions"] = str(suggestions)

        if "word_count" not in data or not isinstance(data.get("word_count"), int):
            data["word_count"] = word_count

        # Normalize numerical scores (0.0 to 100.0)
        for score_field in ["relevance_score", "content_score"]:
            val = data.get(score_field, 0.0)
            try:
                val = float(val)
                if 0.0 < val <= 1.0:
                    val = val * 100.0
                val = max(0.0, min(100.0, val))
            except (ValueError, TypeError):
                val = 0.0
            data[score_field] = round(val, 1)

        for field in ["clarity", "engagement", "structure", "relevance", "answer_quality"]:
            data[field] = str(data.get(field, "") or "")

        raw_grammar = data.get("grammar", [])
        if not isinstance(raw_grammar, list):
            data["grammar"] = []
        else:
            normalized_grammar = []
            for item in raw_grammar:
                if isinstance(item, dict):
                    normalized_grammar.append({
                        "issue": str(item.get("issue", "")),
                        "correction": str(item.get("correction", "")),
                    })
            data["grammar"] = normalized_grammar

        raw_tone = data.get("tone")
        if isinstance(raw_tone, dict):
            score = raw_tone.get("score", 0.0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            appropriateness = str(raw_tone.get("appropriateness", "Neutral") or "Neutral")
            data["tone"] = {"score": score, "appropriateness": appropriateness}
        elif isinstance(raw_tone, (int, float)):
            data["tone"] = {"score": float(raw_tone), "appropriateness": "Neutral"}
        else:
            data["tone"] = {"score": 0.0, "appropriateness": str(raw_tone or "Neutral")}

        return data

    DEFAULT_QUESTION = "Self-introduction / Tell me about yourself, your background, and your key strengths."

    def analyze_content(self, answer_text: str, question: str = "") -> dict:
        effective_question = question.strip() if question and question.strip() else self.DEFAULT_QUESTION
        word_count = len(answer_text.split())
        payload = {
            "question": effective_question,
            "answer": answer_text or "(No answer provided)",
            "word_count": word_count,
        }
        raw = ""

        try:
            raw = self.chain.invoke(payload)
            data = self._normalize_result(parse_llm_json(raw), word_count=word_count)
            if word_count < 20 or data.get("suggestions") != "Answer is too short to evaluate.":
                return data

            logger.warning(
                "ContentAnalyzer: model incorrectly classified %d words as too short; retrying.",
                word_count,
            )
            retry_prompt = self.prompt.format(**payload) + """

IMPORTANT CORRECTION: The candidate answer contains more than 20 words.
You MUST evaluate it. Do not return the too-short fallback. Return only the requested JSON object.
"""
            retry_raw = self.llm.invoke(retry_prompt)
            retry_data = self._normalize_result(parse_llm_json(retry_raw), word_count=word_count)
            if word_count >= 20 and retry_data.get("suggestions") == "Answer is too short to evaluate.":
                raise ValueError("LLM returned the too-short fallback twice")
            return retry_data

        except (ValueError, json.JSONDecodeError):
            logger.error("ContentAnalyzer: LLM returned non-JSON. Raw: %s", raw[:300])
        except Exception as exc:
            logger.error("ContentAnalyzer: LLM call failed: %s", exc)

        # Fallback
        return {
            "word_count": word_count,
            "clarity": "N/A",
            "engagement": "N/A",
            "structure": "N/A",
            "grammar": [],
            "tone": {"score": 0.0, "appropriateness": "Could not evaluate."},
            "relevance": "N/A",
            "answer_quality": "N/A",
            "suggestions": "Could not analyze the content due to an internal error.",
        }