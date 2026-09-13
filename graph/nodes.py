import logging
from typing import Any

from modules import AudioProcessor, VideoAnalyzer, SpeechAnalyzer, ContentAnalyzer
from graph.state import InterviewState
from api.schemas import AnswerStatus

logger = logging.getLogger(__name__)


# Node 1 – Extract audio from video
def extract_audio_node(state: InterviewState) -> dict[str, Any]:
    logger.info("[Node] extract_audio  video=%s", state["video_path"])

    audio_path = AudioProcessor.extract_audio(state["video_path"])
    if not audio_path:
        return {"error": "Failed to extract audio from the video file."}

    return {"audio_path": audio_path}


# Node 2 – Video emotion + blink analysis
def analyze_video_node(state: InterviewState) -> dict[str, Any]:
    logger.info("[Node] analyze_video  video=%s", state["video_path"])

    result = VideoAnalyzer.analyze_emotions(state["video_path"])
    if "error" in result:
        return {"error": result["error"]}

    return {"video_analysis": result}


# Node 3 – Transcribe audio and detect pronunciation issues
def analyze_speech_node(state: InterviewState, speech_analyzer: SpeechAnalyzer) -> dict[str, Any]:
    audio_path = state.get("audio_path")
    if not audio_path:
        return {"error": "No audio path available for speech analysis."}

    logger.info("[Node] analyze_speech  audio=%s", audio_path)

    result = speech_analyzer.analyze_audio(audio_path)

    return {
        "transcript": result.formatted_text,
        "speech_analysis": {
            "pronunciation_issues": result.pronunciation_issues,
            "raw_transcript": result.raw_transcript,
            "formatted_text": result.formatted_text,
        },
        "answer_validity": result.validity.model_dump() if result.validity else None,
    }


# Node 4 – Content / answer quality analysis
def analyze_content_node(state: InterviewState, content_analyzer: ContentAnalyzer) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    question = state.get("question", "")
    answer_validity = state.get("answer_validity", {})
    status = answer_validity.get("status") if answer_validity else None

    # Short-circuit if no meaningful speech was detected
    if status and status != AnswerStatus.VALID:
        logger.info("[Node] analyze_content  bypassed (status=%s)", status)
        return {
            "content_analysis": {
                "word_count": answer_validity.get("transcript_word_count", 0),
                "clarity": "N/A",
                "engagement": "N/A",
                "structure": "N/A",
                "grammar": [],
                "tone": {"score": 0.0, "appropriateness": "Could not evaluate due to missing/insufficient answer."},
                "relevance": "N/A",
                "answer_quality": "No meaningful spoken answer detected.",
                "suggestions": "Please ensure your microphone is working and provide a clear, full answer to the question."
            }
        }

    logger.info("[Node] analyze_content  words=%d", len(transcript.split()))

    result = content_analyzer.analyze_content(answer_text=transcript, question=question)
    return {"content_analysis": result}


# Routing helpers
def route_after_audio(state: InterviewState) -> str:
    if state.get("error"):
        return "error"
    return "ok"