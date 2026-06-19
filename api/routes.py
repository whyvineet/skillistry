import os
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request

from api.schemas import AnalysisResponse, ErrorResponse
from utils.helpers import allowed_video_file, generate_unique_filename, ensure_dir_exists, cleanup_temp_files

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


@router.post(
    "/upload",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Interview Analysis"],
    summary="Upload an interview video and receive a full analysis report.",
)
async def upload_video(
    request: Request,
    video: Annotated[UploadFile, File(description="Interview response video (mp4/mov/avi/webm)")],
    question: Annotated[str, Form(description="The interview question that was asked")] = "",
):
    if not allowed_video_file(video.filename or ""):
        raise HTTPException(status_code=400, detail="File type not allowed. Use mp4, mov, avi, or webm.")

    upload_folder: str = request.app.state.upload_folder
    ensure_dir_exists(upload_folder)

    unique_name = generate_unique_filename(video.filename or "upload.mp4")
    video_path = os.path.join(upload_folder, unique_name)
    audio_path: str | None = None

    # Save uploaded file
    try:
        contents = await video.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        logger.info("Saved upload → %s  (%d bytes)", video_path, len(contents))
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {exc}")

    # Run LangGraph pipeline
    try:
        workflow = request.app.state.workflow
        video_id = os.path.splitext(unique_name)[0]

        initial_state = {
            "video_path": video_path,
            "question": question,
            "video_id": video_id,
        }

        logger.info("Invoking workflow for video_id=%s", video_id)
        final_state = workflow.invoke(initial_state)

        # Handle pipeline errors
        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])

        # Build response
        audio_path = final_state.get("audio_path")  # track for cleanup

        speech = final_state.get("speech_analysis", {})
        content = final_state.get("content_analysis", {})
        video_result = final_state.get("video_analysis", {})

        response = AnalysisResponse(
            video_id=video_id,
            transcription=final_state.get("transcript", ""),
            confidence_analysis=video_result,
            speech_analysis=speech.get("pronunciation_issues", []),
            content_analysis=content,
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error processing video: {exc}")

    finally:
        # Always clean up temporary audio (video kept for potential re-analysis)
        temp_files = [p for p in [audio_path] if p]
        if temp_files:
            cleanup_temp_files(temp_files)