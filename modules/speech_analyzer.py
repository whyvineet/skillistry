import json
import logging
from dataclasses import dataclass, field

import speech_recognition as sr
from langchain_ollama import OllamaLLM
from api.schemas import AnswerValidity, AnswerStatus

logger = logging.getLogger(__name__)


@dataclass
class SpeechAnalysis:
    formatted_text: str = ""
    pronunciation_issues: list[dict] = field(default_factory=list)
    raw_transcript: str = ""
    validity: AnswerValidity | None = None


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

    def _transcribe(self, audio_path: str) -> tuple[str, AnswerStatus]:
        try:
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            logger.info("Transcription succeeded (%d words)", len(text.split()))
            return text, AnswerStatus.VALID
        except sr.UnknownValueError:
            logger.warning("Speech recognizer could not understand audio.")
            return "", AnswerStatus.NO_SPEECH
        except sr.RequestError as exc:
            logger.error("Speech recognition request failed: %s", exc)
            return "", AnswerStatus.TRANSCRIPTION_FAILED

    def analyze_audio(self, audio_path: str) -> SpeechAnalysis:
        raw_transcript, status = self._transcribe(audio_path)
        word_count = len(raw_transcript.split())

        if status == AnswerStatus.VALID and word_count < 3:
            status = AnswerStatus.INSUFFICIENT_RESPONSE

        validity = AnswerValidity(
            status=status,
            speech_detected=status in (AnswerStatus.VALID, AnswerStatus.INSUFFICIENT_RESPONSE),
            transcript_available=status in (AnswerStatus.VALID, AnswerStatus.INSUFFICIENT_RESPONSE),
            transcript_word_count=word_count,
            meaningful_answer=(status == AnswerStatus.VALID),
            reason="Insufficient words detected." if status == AnswerStatus.INSUFFICIENT_RESPONSE else (
                   "No speech detected in audio." if status == AnswerStatus.NO_SPEECH else (
                   "Transcription service failed." if status == AnswerStatus.TRANSCRIPTION_FAILED else None))
        )

        if status != AnswerStatus.VALID:
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
                validity=validity,
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
                validity=validity,
            )
        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON pronunciation analysis. Raw: %s", response_text[:300])
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
                validity=validity,
            )
        except Exception as exc:
            logger.error("Pronunciation LLM call failed: %s", exc)
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
                validity=validity,
            )