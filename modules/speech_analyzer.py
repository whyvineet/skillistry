import logging
import json
import wave
from typing import Any
from dataclasses import dataclass, field

import speech_recognition as sr
from langchain_ollama import OllamaLLM
from utils.helpers import parse_llm_json

logger = logging.getLogger(__name__)


@dataclass
class SpeechAnalysis:
    formatted_text: str = ""
    pronunciation_issues: list[dict] = field(default_factory=list)
    raw_transcript: str = ""
    speaking_rate_wpm: float = 0.0
    pacing_category: str = "optimal"


class SpeechAnalyzer:

    PRONUNCIATION_PROMPT = """
        You are an expert interview speech coach. Given the following transcript from a spoken interview answer,
        identify at most 3 to 5 notable pronunciation errors, awkward phrasing, or filler words/disfluencies
        (e.g., um, uh, like, repetitive filler words, or informal slang).

        Transcript:
        {transcript}

        Return ONLY a valid JSON object — no markdown, no commentary — with exactly these keys:
        {{
        "pronunciation_issues": [
            {{
            "word": "<word or phrase as spoken>",
            "correction": "<specific pronunciation hint or professional alternative>",
            "phonetic": "<IPA or phonetic guide>"
            }}
        ],
        "formatted_text": "<cleaned, properly capitalised and punctuated version of the transcript>"
        }}

        Rules:
        - Only include GENUINE pronunciation issues, filler sounds (um/uh), or informal slang.
        - Do NOT flag standard, common English words (such as 'marketing', 'clients', 'number', 'these') if they are used correctly.
        - If there are no clear issues, return an empty array [] for "pronunciation_issues".
        - Ensure "word" has no leading or trailing whitespace.
        - Return at most 5 items in "pronunciation_issues".
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

    @staticmethod
    def _calculate_audio_pacing(audio_path: str, word_count: int) -> tuple[float, str]:
        """Compute speaking rate (Words Per Minute) and categorize pacing."""
        if word_count == 0:
            return 0.0, "no_speech_detected"

        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate) if rate > 0 else 0.0

            if duration > 0 and word_count > 0:
                wpm = round((word_count / duration) * 60.0, 1)
                if wpm < 110:
                    category = "too_slow"
                elif wpm <= 160:
                    category = "optimal"
                else:
                    category = "too_fast"
                return wpm, category
        except Exception as exc:
            logger.debug("Could not calculate audio pacing: %s", exc)
        return 0.0, "optimal"

    @staticmethod
    def _normalize_issues(issues: Any) -> list[dict]:
        if not isinstance(issues, list):
            return []
        normalized = []
        for item in issues:
            if isinstance(item, dict):
                w = str(item.get("word", "")).strip()
                c = str(item.get("correction", "")).strip()
                p = str(item.get("phonetic", "")).strip()
                if w:
                    normalized.append({
                        "word": w,
                        "correction": c,
                        "phonetic": p,
                    })
        return normalized[:5]

    def analyze_audio(self, audio_path: str) -> SpeechAnalysis:
        raw_transcript = self._transcribe(audio_path)
        word_count = len(raw_transcript.split())
        wpm, pacing_cat = self._calculate_audio_pacing(audio_path, word_count)

        if not raw_transcript.strip():
            return SpeechAnalysis(
                raw_transcript="",
                formatted_text="",
                pronunciation_issues=[],
                speaking_rate_wpm=0.0,
                pacing_category="no_speech_detected",
            )

        prompt = self.PRONUNCIATION_PROMPT.format(transcript=raw_transcript)
        response_text = ""

        try:
            response_text = self.llm.invoke(prompt)
            data = parse_llm_json(response_text)
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=str(data.get("formatted_text", raw_transcript) or raw_transcript),
                pronunciation_issues=self._normalize_issues(data.get("pronunciation_issues", [])),
                speaking_rate_wpm=wpm,
                pacing_category=pacing_cat,
            )
        except json.JSONDecodeError:
            logger.warning("LLM returned malformed pronunciation JSON; retrying once.")
            retry_prompt = prompt + """

IMPORTANT: Your previous response was invalid JSON. Return one complete valid JSON object.
Use double quotes around every key and string value, include the colon after every key,
and return no Markdown or commentary.
"""
            retry_raw = ""
            try:
                retry_raw = self.llm.invoke(retry_prompt)
                retry_data = parse_llm_json(retry_raw)
                return SpeechAnalysis(
                    raw_transcript=raw_transcript,
                    formatted_text=str(retry_data.get("formatted_text", raw_transcript) or raw_transcript),
                    pronunciation_issues=self._normalize_issues(retry_data.get("pronunciation_issues", [])),
                    speaking_rate_wpm=wpm,
                    pacing_category=pacing_cat,
                )
            except Exception:
                logger.error(
                    "Pronunciation retry also returned invalid JSON. Raw: %s",
                    retry_raw[:300] if retry_raw else response_text[:300],
                )
                return SpeechAnalysis(
                    raw_transcript=raw_transcript,
                    formatted_text=raw_transcript,
                    pronunciation_issues=[],
                    speaking_rate_wpm=wpm,
                    pacing_category=pacing_cat,
                )
        except Exception as exc:
            logger.error("Pronunciation LLM call failed: %s", exc)
            return SpeechAnalysis(
                raw_transcript=raw_transcript,
                formatted_text=raw_transcript,
                pronunciation_issues=[],
                speaking_rate_wpm=wpm,
                pacing_category=pacing_cat,
            )
