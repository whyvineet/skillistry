from typing import Any, Optional
from typing_extensions import TypedDict


class InterviewState(TypedDict, total=False):
    # Inputs
    video_path: str          # absolute path to the saved video file
    audio_path: Optional[str]  # set after audio extraction
    question: str            # the interview question text

    # Intermediate results
    video_analysis: Optional[dict[str, Any]]   # from VideoAnalyzer
    speech_analysis: Optional[dict[str, Any]]  # from SpeechAnalyzer (serialised)
    transcript: str                             # cleaned formatted transcript

    # Final results
    content_analysis: Optional[dict[str, Any]] # from ContentAnalyzer

    # Control flow
    error: Optional[str]     # set by any node that encounters a fatal error
    video_id: str            # stem of the unique filename (for client reference)