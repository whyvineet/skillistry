import os
import re
import json
import random
import logging
from typing import Any
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}


def parse_llm_json(response: str) -> dict[str, Any]:
    """Parse a JSON object from an Ollama response, including fenced output."""
    if not response or not isinstance(response, str):
        raise json.JSONDecodeError("Empty response", str(response), 0)

    text = response.strip()

    # 1. Match code fence block if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Extract substring from first '{' to last '}'
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    end = text.rfind("}")
    if end > start:
        text = text[start : end + 1]
    else:
        text = text[start:]

    # 3. Clean up trailing commas before closing braces/brackets
    text_cleaned = re.sub(r",\s*([\]}])", r"\1", text)

    try:
        data = json.loads(text_cleaned)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)

    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return data



def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def generate_unique_filename(filename: str) -> str:
    """Generate a clean filename prefixed with a 4-digit unique code (e.g. 4821_video.mp4)."""
    code = random.randint(1000, 9999)
    safe_name = secure_filename(filename) if filename else "video.mp4"
    return f"{code}_{safe_name}"


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