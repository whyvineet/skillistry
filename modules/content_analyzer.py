import json
import logging

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

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

    Rules:
    - "grammar" should be an empty array [] if no issues are found.
    - "tone.score": -1 = very negative/nervous, 0 = neutral, 1 = very positive/confident.
    - If the answer is too short to evaluate meaningfully, fill all string fields with "N/A",
    set tone.score to 0.0, and set suggestions to "Answer is too short to evaluate."
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

    def analyze_content(self, answer_text: str, question: str = "") -> dict:
        word_count = len(answer_text.split())

        try:
            raw: str = self.chain.invoke(
                {
                    "question": question or "(No question provided)",
                    "answer": answer_text or "(No answer provided)",
                    "word_count": word_count,
                }
            )
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)

        except json.JSONDecodeError:
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