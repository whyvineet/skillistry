"""
In-memory interview session service.

This module manages the lifecycle of interview sessions.
Sessions are stored in a module-level dict (no database).
Data is lost on server restart, which is acceptable for the MVP.

The service is designed so that persistent storage (Redis, SQL, etc.)
can be added later by swapping out the _sessions dict for a repository
class without changing the calling code in routes.py.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from interview.questions import QuestionItem, select_questions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

@dataclass
class InterviewSession:
    session_id: str
    status: str                              # created | in_progress | completed | failed
    created_at: datetime
    questions: list[QuestionItem]            # 5 ordered questions

    # Accumulated results — one entry per submitted question
    # Each entry is a raw dict carrying all pipeline outputs for that question
    question_results: list[dict[str, Any]] = field(default_factory=list)

    # Set once all 5 questions are processed
    overall_analysis: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_sessions: dict[str, InterviewSession] = {}


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class InterviewService:
    """
    Manages interview sessions.

    Usage
    -----
    service = InterviewService()

    session = service.create_session()
    session = service.get_session(session_id)
    service.record_question_result(session_id, question_id, result_dict)
    service.record_overall_analysis(session_id, analysis_dict)
    """

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, category: str = "general", n_questions: int = 5) -> InterviewSession:
        """Create a new interview session and return it."""
        session_id = str(uuid.uuid4())
        questions = select_questions(n=n_questions, category=category)

        session = InterviewSession(
            session_id=session_id,
            status="created",
            created_at=datetime.now(timezone.utc),
            questions=questions,
        )
        _sessions[session_id] = session
        logger.info("Created session %s with %d questions", session_id, len(questions))
        return session

    def get_session(self, session_id: str) -> InterviewSession | None:
        """Return the session or None if not found."""
        return _sessions.get(session_id)

    def get_question(self, session: InterviewSession, question_id: str) -> QuestionItem | None:
        """Find a question within the session by its ID."""
        for q in session.questions:
            if q.question_id == question_id:
                return q
        return None

    def question_already_answered(self, session: InterviewSession, question_id: str) -> bool:
        """Return True if a result has already been submitted for this question."""
        return any(
            r.get("question", {}).get("question_id") == question_id
            for r in session.question_results
        )

    def get_question_result(self, session: InterviewSession, question_id: str) -> dict[str, Any] | None:
        """Return the pipeline result for a specific question if available."""
        for r in session.question_results:
            if r.get("question", {}).get("question_id") == question_id:
                return r
        return None

    def record_question_result(
        self,
        session: InterviewSession,
        question: QuestionItem,
        result: dict[str, Any],
    ) -> None:
        """
        Append the pipeline result for a question.

        The result dict should contain:
            transcript, video_analysis, speech_analysis, content_analysis, score_data
        """
        entry = {
            "question": {
                "question_id":     question.question_id,
                "question_number": question.question_number,
                "question_text":   question.question_text,
                "category":        question.category,
                "dimension":       question.dimension,
            },
            **result,
        }
        session.question_results.append(entry)
        session.status = "in_progress"

        answered = len(session.question_results)
        logger.info(
            "Session %s: recorded Q%d (%d/%d answered)",
            session.session_id,
            question.question_number,
            answered,
            len(session.questions),
        )

    def record_overall_analysis(self, session: InterviewSession, analysis: dict[str, Any]) -> None:
        """Store the completed overall analysis and mark the session done."""
        session.overall_analysis = analysis
        session.status = "completed"
        logger.info("Session %s: completed with overall analysis", session.session_id)

    def mark_failed(self, session: InterviewSession, reason: str) -> None:
        session.status = "failed"
        logger.error("Session %s failed: %s", session.session_id, reason)

    # ------------------------------------------------------------------
    # Convenience helpers for routes
    # ------------------------------------------------------------------

    def is_complete(self, session: InterviewSession) -> bool:
        """Return True when all questions have been answered."""
        return len(session.question_results) >= len(session.questions)

    def answered_question_ids(self, session: InterviewSession) -> set[str]:
        return {
            r.get("question", {}).get("question_id", "")
            for r in session.question_results
        }
