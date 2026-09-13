"""
Question bank for the AI mock interview system.

Structure is designed so that role-specific pools can be added later
by providing a different category key (e.g. "software_engineer",
"marketing", "fresher") without changing the calling code.
"""
import json
import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionItem:
    question_id: str
    question_number: int          # 1-based position in the selected set
    question_text: str
    category: str                 # behavioural dimension
    dimension: str                # scoring dimension label


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------
# Each entry covers one of the five target dimensions:
#   1. introduction      – confidence + self-presentation
#   2. communication     – clarity, structure, engagement
#   3. problem_solving   – analytical thinking, approach
#   4. adaptability      – teamwork, handling change / conflict
#   5. self_awareness    – motivation, growth mindset
#
# We keep 3+ options per dimension so sessions are randomised.
# ---------------------------------------------------------------------------

_JSON_PATH = os.path.join(os.path.dirname(__file__), "questions.json")

def _load_pool() -> dict[str, list[dict]]:
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"general": []}

_POOL: dict[str, list[dict]] = _load_pool()

# Ordered dimensions — one question is chosen from each group so every
# session always covers all five areas.
_DIMENSION_ORDER = [
    "introduction",
    "communication",
    "problem_solving",
    "adaptability",
    "self_awareness",
]


def select_questions(n: int = 5, category: str = "general") -> list[QuestionItem]:
    """
    Select *n* questions from the given category pool.

    One question is chosen from each of the five behavioural dimensions
    in a fixed order (introduction → communication → problem_solving →
    adaptability → self_awareness).  Within each dimension the choice is
    randomised so repeated sessions receive different questions.

    Args:
        n:        Number of questions to return (default 5).
        category: Question pool key (default "general").

    Returns:
        Ordered list of QuestionItem with question_number filled in.
    """
    pool = _POOL.get(category, _POOL["general"])

    # Group by category
    by_dim: dict[str, list[dict]] = {}
    for q in pool:
        by_dim.setdefault(q["category"], []).append(q)

    selected: list[QuestionItem] = []
    dims = _DIMENSION_ORDER[:n]

    for position, dim in enumerate(dims, start=1):
        candidates = by_dim.get(dim, [])
        if not candidates:
            continue
        chosen = random.choice(candidates)
        selected.append(
            QuestionItem(
                question_id=chosen["id"],
                question_number=position,
                question_text=chosen["text"],
                category=chosen["category"],
                dimension=chosen["dimension"],
            )
        )

    return selected
