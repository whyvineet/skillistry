import os
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request, BackgroundTasks

from api.schemas import (
    ErrorResponse,
    CreateSessionResponse,
    QuestionItemSchema,
    SessionStatusResponse,
    QuestionResultResponse,
    QuestionScoreSchema,
    InterviewReportResponse,
    OverallAnalysisSchema,
    QuestionDetailSchema,
    TrendItemSchema,
    ConfidenceAnalysisModel,
    ContentAnalysisModel,
    ToneModel,
    PronunciationIssue,
    GrammarIssue,
    QuestionProcessingStatusResponse,
    QuestionPollResponse,
)
from interview.scoring import compute_question_score, compute_overall_score
from utils.helpers import (
    allowed_video_file,
    generate_unique_filename,
    ensure_dir_exists,
    cleanup_temp_files,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}




# ---------------------------------------------------------------------------
# Session-based interview endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/interviews",
    response_model=CreateSessionResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["AI Mock Interview"],
    summary="Create a new interview session and receive 5 questions.",
)
async def create_interview(request: Request):
    """
    Creates a new interview session.

    Returns the session ID and the list of 5 behavioural questions the
    candidate should answer.  The frontend stores the session_id and
    iterates through the questions, uploading each response in turn.
    """
    service = request.app.state.interview_service
    try:
        session = service.create_session()
    except Exception as exc:
        logger.exception("Failed to create interview session: %s", exc)
        raise HTTPException(status_code=500, detail="Could not create interview session.")

    questions = [
        QuestionItemSchema(
            question_id=q.question_id,
            question_number=q.question_number,
            question_text=q.question_text,
            category=q.category,
            dimension=q.dimension,
        )
        for q in session.questions
    ]

    return CreateSessionResponse(
        session_id=session.session_id,
        status=session.status,
        questions=questions,
    )


@router.get(
    "/interviews/{session_id}/status",
    response_model=SessionStatusResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["AI Mock Interview"],
    summary="Get the current status of an interview session.",
)
async def get_interview_status(session_id: str, request: Request):
    service = request.app.state.interview_service
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        questions_answered=len(session.question_results),
        total_questions=len(session.questions),
    )


@router.post(
    "/interviews/{session_id}/questions/{question_id}/response",
    response_model=QuestionProcessingStatusResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["AI Mock Interview"],
    summary="Upload the candidate's recorded answer and begin processing in the background.",
)
async def submit_question_response(
    session_id: str,
    question_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="Recorded answer video (mp4/mov/avi/webm)")],
):
    """
    Upload a video response for a specific question in an active session.

    The backend already knows the question text — the frontend does NOT
    need to send it. Returns a processing acknowledgment immediately.
    """
    service = request.app.state.interview_service

    # --- Validate session ---
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.status == "completed":
        raise HTTPException(status_code=409, detail="This interview session is already completed.")

    if session.status == "failed":
        raise HTTPException(status_code=409, detail="This interview session has failed. Please start a new one.")

    # --- Validate question ---
    question = service.get_question(session, question_id)
    if not question:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in session '{session_id}'.",
        )

    if service.question_already_answered(session, question_id):
        raise HTTPException(
            status_code=409,
            detail=f"Question '{question_id}' has already been answered.",
        )

    # --- Validate file ---
    if not allowed_video_file(video.filename or ""):
        raise HTTPException(status_code=400, detail="File type not allowed. Use mp4, mov, avi, or webm.")

    # --- Save file to session-scoped directory ---
    upload_folder: str = request.app.state.upload_folder
    session_dir = os.path.join(upload_folder, session_id, f"q{question.question_number}")
    ensure_dir_exists(session_dir)

    unique_name = generate_unique_filename(video.filename or "response.webm")
    video_path = os.path.join(session_dir, unique_name)
    audio_path: str | None = None

    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
        with open(video_path, "wb") as f:
            f.write(contents)
        logger.info(
            "Session %s / Q%d: saved -> %s (%d bytes)",
            session_id, question.question_number, video_path, len(contents),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save video for session %s / Q%d: %s", session_id, question.question_number, exc)
        raise HTTPException(status_code=500, detail="Could not save uploaded video.")

    # Enqueue background processing
    background_tasks.add_task(
        _process_video_background,
        request.app.state,
        session_id,
        question_id,
        question,
        video_path,
    )

    return QuestionProcessingStatusResponse()


@router.get(
    "/interviews/{session_id}/questions/{question_id}/result",
    response_model=QuestionPollResponse,
    responses={
        404: {"model": ErrorResponse},
    },
    tags=["AI Mock Interview"],
    summary="Poll for the processing result of a specific question.",
)
async def get_question_result(
    session_id: str,
    question_id: str,
    request: Request,
):
    service = request.app.state.interview_service
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if session.status == "failed":
        return QuestionPollResponse(status="failed")

    question = service.get_question(session, question_id)
    if not question:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{question_id}' not found in session '{session_id}'.",
        )

    # Check if the result is already computed and stored
    result_data = service.get_question_result(session, question_id)
    
    if result_data:
        # Build the final response format from stored data
        va = result_data.get("video_analysis", {})
        sa = result_data.get("speech_analysis", {})
        ca = result_data.get("content_analysis", {})
        sd = result_data.get("score_data", {})
        
        is_last = service.is_complete(session)
        
        next_q_item = None
        answered_numbers = {
            r["question"]["question_number"] for r in session.question_results
        }
        for q in session.questions:
            if q.question_number not in answered_numbers:
                next_q_item = QuestionItemSchema(
                    question_id=q.question_id,
                    question_number=q.question_number,
                    question_text=q.question_text,
                    category=q.category,
                    dimension=q.dimension,
                )
                break

        response_payload = QuestionResultResponse(
            session_id=session_id,
            question_id=question_id,
            question_number=question.question_number,
            question_text=question.question_text,
            score=QuestionScoreSchema(
                confidence_score=sd.get("confidence_score", 0.0),
                body_language_score=sd.get("body_language_score", 0.0),
                speech_score=sd.get("speech_score", 0.0),
                content_score=sd.get("content_score", 0.0),
                question_score=sd.get("question_score", 0.0),
            ),
            transcription=result_data.get("transcript", ""),
            confidence_analysis=_build_confidence_model(va),
            speech_analysis=_build_speech_list(sa),
            content_analysis=_build_content_model(ca),
            answer_validity=result_data.get("answer_validity"),
            is_last_question=is_last,
            next_question=next_q_item,
        )
        return QuestionPollResponse(status="completed", result=response_payload)
    else:
        return QuestionPollResponse(status="processing")


async def _process_video_background(
    app_state: Any,
    session_id: str,
    question_id: str,
    question,
    video_path: str,
):
    """Background task to run the AI pipeline on the uploaded video."""
    service = app_state.interview_service
    session = service.get_session(session_id)
    if not session:
        return
        
    audio_path = None
    try:
        workflow = app_state.workflow
        video_id = f"{session_id}_q{question.question_number}"

        initial_state = {
            "video_path": video_path,
            "question": question.question_text,
            "video_id": video_id,
        }

        logger.info("Invoking background workflow for session=%s question=%s", session_id, question_id)
        final_state = workflow.invoke(initial_state)

        if final_state.get("error"):
            logger.error("Pipeline error for session %s / Q%d: %s", session_id, question.question_number, final_state["error"])
            service.mark_failed(session, final_state["error"])
            return

        audio_path = final_state.get("audio_path")

        video_analysis = final_state.get("video_analysis", {})
        speech_raw     = final_state.get("speech_analysis", {})
        content_raw    = final_state.get("content_analysis", {})
        transcript     = final_state.get("transcript", "")
        answer_validity = final_state.get("answer_validity")

    except Exception as exc:
        logger.exception("Pipeline exception for session %s / Q%d: %s", session_id, question.question_number, exc)
        service.mark_failed(session, str(exc))
        return
    finally:
        if audio_path:
            cleanup_temp_files([audio_path])

    # --- Compute score ---
    try:
        score_data = compute_question_score(
            question_id=question_id,
            question_number=question.question_number,
            video_analysis=video_analysis,
            speech_analysis=speech_raw,
            content_analysis=content_raw,
            answer_validity=answer_validity,
        )
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        # Non-fatal: return zeroed score rather than crashing
        from interview.scoring import QuestionScore
        score_data = QuestionScore(
            question_id=question_id,
            question_number=question.question_number,
            confidence_score=0.0,
            body_language_score=0.0,
            speech_score=0.0,
            content_score=0.0,
            question_score=0.0,
        )

    # --- Record result ---
    service.record_question_result(
        session,
        question,
        result={
            "transcript":       transcript,
            "video_analysis":   video_analysis,
            "speech_analysis":  speech_raw,
            "content_analysis": content_raw,
            "answer_validity":  answer_validity,
            "score_data": {
                "confidence_score":    score_data.confidence_score,
                "body_language_score": score_data.body_language_score,
                "speech_score":        score_data.speech_score,
                "content_score":       score_data.content_score,
                "question_score":      score_data.question_score,
            },
        },
    )

    # --- Check if this was the last question; if so, run overall analysis ---
    is_last = service.is_complete(session)

    if is_last and session.overall_analysis is None:
        # We need a mock request for _run_overall_analysis or to adapt it.
        # Actually _run_overall_analysis only needs app.state.interview_analyzer.
        _run_overall_analysis_background(app_state, session, service)


def _run_overall_analysis_background(app_state: Any, session, service) -> None:
    """Compute and store the overall analysis for a completed session."""
    try:
        from interview.scoring import QuestionScore, compute_overall_score

        q_scores: list[QuestionScore] = []
        for result in session.question_results:
            sd = result.get("score_data", {})
            q_info = result.get("question", {})
            from interview.scoring import QuestionScore as QS
            q_scores.append(
                QS(
                    question_id=q_info.get("question_id", ""),
                    question_number=q_info.get("question_number", 0),
                    confidence_score=sd.get("confidence_score", 0.0),
                    body_language_score=sd.get("body_language_score", 0.0),
                    speech_score=sd.get("speech_score", 0.0),
                    content_score=sd.get("content_score", 0.0),
                    question_score=sd.get("question_score", 0.0),
                )
            )
        overall_scores = compute_overall_score(q_scores)

        # Qualitative LLM analysis
        interview_analyzer = app_state.interview_analyzer
        qualitative = interview_analyzer.analyze(session)

        service.record_overall_analysis(
            session,
            {
                "scores": {
                    "overall_score":       overall_scores.overall_score,
                    "communication_score": overall_scores.communication_score,
                    "confidence_score":    overall_scores.confidence_score,
                    "speech_score":        overall_scores.speech_score,
                    "body_language_score": overall_scores.body_language_score,
                    "content_score":       overall_scores.content_score,
                    "question_scores":     overall_scores.question_scores,
                    "confidence_trend":    overall_scores.trends.get("confidence", []),
                    "speech_trend":        overall_scores.trends.get("speech", []),
                    "content_trend":       overall_scores.trends.get("content", []),
                },
                "qualitative": qualitative,
            },
        )

    except Exception as exc:
        logger.exception("Overall analysis failed for session %s: %s", session.session_id, exc)
        service.mark_failed(session, str(exc))




@router.get(
    "/interviews/{session_id}",
    response_model=InterviewReportResponse,
    responses={
        202: {"description": "Session still in progress"},
        404: {"model": ErrorResponse},
    },
    tags=["AI Mock Interview"],
    summary="Get the full interview report for a completed session.",
)
async def get_interview_report(session_id: str, request: Request):
    """
    Returns the full interview report once all questions have been
    answered and analyzed.  If the session is still in progress,
    returns status 202 with partial data.
    """
    service = request.app.state.interview_service
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    # Build question details from stored results
    question_details: list[QuestionDetailSchema] = []
    for result in session.question_results:
        q_info = result.get("question", {})
        sd = result.get("score_data", {})
        va = result.get("video_analysis", {})
        sa = result.get("speech_analysis", {})
        ca = result.get("content_analysis", {})
        tr = result.get("transcript", "")

        question_details.append(
            QuestionDetailSchema(
                question_id=q_info.get("question_id", ""),
                question_number=q_info.get("question_number", 0),
                question_text=q_info.get("question_text", ""),
                score=sd.get("question_score", 0.0),
                confidence_score=sd.get("confidence_score", 0.0),
                body_language_score=sd.get("body_language_score", 0.0),
                speech_score=sd.get("speech_score", 0.0),
                content_score=sd.get("content_score", 0.0),
                transcription=tr,
                confidence_analysis=_build_confidence_model(va),
                speech_analysis=_build_speech_list(sa),
                content_analysis=_build_content_model(ca),
                answer_validity=result.get("answer_validity"),
            )
        )

    # Sort by question number
    question_details.sort(key=lambda x: x.question_number)

    overall_schema: OverallAnalysisSchema | None = None
    if session.overall_analysis:
        oa = session.overall_analysis
        raw_trends = oa.get("qualitative", {}).get("trends", [])
        trend_items = [
            TrendItemSchema(metric=t.get("metric", ""), observation=t.get("observation", ""))
            for t in raw_trends
        ]
        scores = oa.get("scores", {})
        overall_schema = OverallAnalysisSchema(
            overall_score=scores.get("overall_score", 0.0),
            communication_score=scores.get("communication_score", 0.0),
            confidence_score=scores.get("confidence_score", 0.0),
            speech_score=scores.get("speech_score", 0.0),
            body_language_score=scores.get("body_language_score", 0.0),
            content_score=scores.get("content_score", 0.0),
            strengths=oa.get("qualitative", {}).get("strengths", []),
            areas_to_improve=oa.get("qualitative", {}).get("areas_to_improve", []),
            recommendations=oa.get("qualitative", {}).get("recommendations", []),
            trends=trend_items,
            question_scores=scores.get("question_scores", []),
            confidence_trend=scores.get("confidence_trend", []),
            speech_trend=scores.get("speech_trend", []),
            content_trend=scores.get("content_trend", []),
        )

    return InterviewReportResponse(
        session_id=session_id,
        status=session.status,
        overall_analysis=overall_schema,
        questions=question_details,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_overall_analysis(request: Request, session, service) -> None:
    """Compute and store the overall analysis for a completed session."""
    try:
        # Numerical aggregation
        from interview.scoring import QuestionScore, compute_overall_score

        q_scores: list[QuestionScore] = []
        for result in session.question_results:
            sd = result.get("score_data", {})
            q_info = result.get("question", {})
            from interview.scoring import QuestionScore as QS
            q_scores.append(
                QS(
                    question_id=q_info.get("question_id", ""),
                    question_number=q_info.get("question_number", 0),
                    confidence_score=sd.get("confidence_score", 0.0),
                    body_language_score=sd.get("body_language_score", 0.0),
                    speech_score=sd.get("speech_score", 0.0),
                    content_score=sd.get("content_score", 0.0),
                    question_score=sd.get("question_score", 0.0),
                )
            )
        overall_scores = compute_overall_score(q_scores)

        # Qualitative LLM analysis
        interview_analyzer = request.app.state.interview_analyzer
        qualitative = interview_analyzer.analyze(session)

        service.record_overall_analysis(
            session,
            {
                "scores": {
                    "overall_score":       overall_scores.overall_score,
                    "communication_score": overall_scores.communication_score,
                    "confidence_score":    overall_scores.confidence_score,
                    "speech_score":        overall_scores.speech_score,
                    "body_language_score": overall_scores.body_language_score,
                    "content_score":       overall_scores.content_score,
                    "question_scores":     overall_scores.question_scores,
                    "confidence_trend":    overall_scores.trends.get("confidence", []),
                    "speech_trend":        overall_scores.trends.get("speech", []),
                    "content_trend":       overall_scores.trends.get("content", []),
                },
                "qualitative": qualitative,
            },
        )

    except Exception as exc:
        logger.exception("Overall analysis failed for session %s: %s", session.session_id, exc)
        service.mark_failed(session, str(exc))


def _build_confidence_model(video_analysis: dict) -> "ConfidenceAnalysisModel":
    from api.schemas import ConfidenceAnalysisModel
    return ConfidenceAnalysisModel(
        confidence_score=video_analysis.get("confidence_score", 0.0),
        frames_analyzed=video_analysis.get("frames_analyzed", 0),
        total_frames=video_analysis.get("total_frames", 0),
        blinks_per_minute=video_analysis.get("blinks_per_minute", 0.0),
        blink_rate_category=video_analysis.get("blink_rate_category", ""),
        emotion_percentages=video_analysis.get("emotion_percentages", {}),
        frame_details=video_analysis.get("frame_details", []),
    )


def _build_speech_list(speech_analysis: dict) -> list:
    from api.schemas import PronunciationIssue
    issues = speech_analysis.get("pronunciation_issues", [])
    result = []
    for issue in issues:
        if isinstance(issue, dict):
            result.append(
                PronunciationIssue(
                    word=issue.get("word", ""),
                    correction=issue.get("correction", ""),
                    phonetic=issue.get("phonetic", ""),
                )
            )
    return result


def _build_content_model(content_analysis: dict) -> "ContentAnalysisModel":
    from api.schemas import ContentAnalysisModel, ToneModel, GrammarIssue
    tone_raw = content_analysis.get("tone", {"score": 0.0, "appropriateness": ""})
    grammar_raw = content_analysis.get("grammar", [])
    grammar = []
    for g in grammar_raw:
        if isinstance(g, dict):
            grammar.append(GrammarIssue(issue=g.get("issue", ""), correction=g.get("correction", "")))

    return ContentAnalysisModel(
        word_count=content_analysis.get("word_count", 0),
        clarity=content_analysis.get("clarity", "N/A"),
        engagement=content_analysis.get("engagement", "N/A"),
        structure=content_analysis.get("structure", "N/A"),
        grammar=grammar,
        tone=ToneModel(
            score=float(tone_raw.get("score", 0.0)),
            appropriateness=tone_raw.get("appropriateness", ""),
        ),
        relevance=content_analysis.get("relevance", "N/A"),
        answer_quality=content_analysis.get("answer_quality", ""),
        suggestions=content_analysis.get("suggestions", ""),
    )
