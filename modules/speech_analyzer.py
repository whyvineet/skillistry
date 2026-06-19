import json
import logging
from dataclasses import dataclass, field

import speech_recognition as sr
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


@dataclass
class SpeechAnalysis:
    formatted_text: str = ""
    pronunciation_issues: list[dict] = field(default_factory=list)
    raw_transcript: str = ""


class SpeechAnalyzer:

    PRONUNCIATION_PROMPT = """
        You are a speech coach. Given the following transcript from a spoken interview answer,
        identify words or phrases that are likely mispronounced, unclear, or are disfluencies
        (um, uh, filler words, etc.).

        Transcript:
        {transcript}

        Return ONLY a valid JSON object — no markdown, no commentary — with exactly these keys:
        {{
        "pronunciation_issues": [
            {{
            "word": "<word as spoken>",
            "correction": "<correct pronunciation hint>",
            "phonetic": "<IPA or simple phonetic guide>"
            }}
        ],
        "formatted_text": "<cleaned, properly capitalised and punctuated version of the transcript>"
        }}

        If there are no issues, return an empty array for "pronunciation_issues" and return the
        cleaned transcript in "formatted_text".
    """

    def __init__(self, ollama_model: str, ollama_base_url: str):
        self.recognizer = sr.Recognizer()
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)

    def _transcribe(self, audio_path: str) -> str:
        try:
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            logger.info("Transcription succeeded (%d words)", len(text.split()))
            return text
        except sr.UnknownValueError:
            logger.warning("Speech recognizer could not understand audio.")
            return ""
        except sr.RequestError as exc:
            logger.error("Speech recognition request failed: %s", exc)
            return ""

    def analyze_audio(self, audio_path: str) -> SpeechAnalysis:
        raw_transcript = self._transcribe(audio_path)

        if not raw_transcript.strip():
            return SpeechAnalysis(
                raw_transcript="",
                formatted_text="",
                pronunciation_issues=[],
            )

        prompt = self.PRONUNCIATION_PROMPT.format(transcript=raw_transcript)

        try:
            response_text: str = self.llm.invoke(prompt)
            # Strip accidental markdown fences
            clean = response_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=data.get("formatted_text", raw_transcript),
                pronunciation_issues=data.get("pronunciation_issues", []),
            )
        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON pronunciation analysis. Raw: %s", response_text[:300])
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
            )
        except Exception as exc:
            logger.error("Pronunciation LLM call failed: %s", exc)
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
            )