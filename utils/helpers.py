import os
import re
import uuid
import logging
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def generate_unique_filename(filename: str) -> str:
    return f"{uuid.uuid4()}_{secure_filename(filename)}"


def ensure_dir_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def cleanup_temp_files(file_paths: list[str]) -> None:
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug("Cleaned up: %s", path)
        except Exception as exc:
            logger.warning("Could not remove %s: %s", path, exc)


def format_transcribed_text(text: str) -> str:
    if not text:
        return text

    if text[-1] not in ".!?":
        text += "."

    text = re.sub(r"([.!?,;:])([^\s])", r"\1 \2", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    processed = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s[0].upper() + s[1:]
            s = re.sub(r",\s*", ", ", s)
            s = re.sub(r"\s+", " ", s)
            processed.append(s)

    result = " ".join(processed)
    result = re.sub(r"\bi\b", "I", result)
    result = re.sub(r"\s+", " ", result)
    return result