import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    # Emotion → confidence weight  (calibrated for interview context where
    # resting/neutral/serious faces are frequently categorized as 'sad' by DeepFace)
    EMOTION_WEIGHTS: dict[str, float] = {
        "happy": 0.95,      # Smiling, positive, warm
        "neutral": 0.85,    # Composed, calm, professional
        "surprise": 0.50,   # Expressive / animated
        "sad": 0.35,        # Resting / focused face in video interview
        "angry": 0.10,      # Focused intensity or mild tension
        "disgust": -0.40,   # Discomfort
        "fear": -0.80,      # Genuine anxiety / nervousness
    }

    # Blink-rate categories and their confidence contribution
    # Research baseline: 11-20 blinks / min is normal conversation baseline
    BLINK_RATE_CONFIDENCE: dict[str, float] = {
        "very_low": 0.80,   # 0-5  bpm  – highly focused / intense gaze
        "low": 0.95,        # 6-10 bpm  – steady & confident eye contact
        "normal": 0.90,     # 11-20 bpm – relaxed, natural conversation
        "high": 0.40,       # 21-30 bpm – slight nervousness
        "very_high": -0.60, # 30+  bpm  – anxiety / rapid eye flutter
    }

    # Optimization: target frame-sample rate
    SAMPLE_FPS: float = 1.0          # analyze 1 frame per second of video
    INFERENCE_WIDTH: int  = 640      # resize frame width before DeepFace
    INFERENCE_HEIGHT: int = 360      # resize frame height before DeepFace

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.blink_threshold = 0.15   # minimum seconds between counted blinks
        self.history_length = 3       # rolling window for open/closed state

    # Optimization: model warm-up (call once at startup)
    @staticmethod
    def warm_up() -> None:
        try:
            from deepface import DeepFace  # noqa: F401

            # Blank 1×1 BGR frame – enough to trigger model init without
            # producing meaningful results.
            blank = np.zeros((1, 1, 3), dtype=np.uint8)
            DeepFace.analyze(blank, actions=["emotion"], enforce_detection=False, silent=True)
            logger.info("DeepFace models pre-warmed successfully.")
        except Exception as exc:
            # Non-fatal – the first real request will warm up instead.
            logger.warning("DeepFace warm-up skipped: %s", exc)

    def _categorize_blink_rate(self, blinks_per_minute: float) -> str:
        if blinks_per_minute <= 5:
            return "very_low"
        if blinks_per_minute <= 10:
            return "low"
        if blinks_per_minute <= 20:
            return "normal"
        if blinks_per_minute <= 30:
            return "high"
        return "very_high"

    def _detect_blinks(
        self, frame: np.ndarray, eye_status_history: list[bool]
    ) -> tuple[bool, bool, list[bool]]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        eyes_detected = False
        for x, y, w, h in faces:
            roi = gray[y : int(y + h * 0.6), x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 4, minSize=(25, 25))
            if len(eyes) >= 1:
                eyes_detected = True
                break

        eye_status_history.append(eyes_detected)
        if len(eye_status_history) > self.history_length:
            eye_status_history.pop(0)

        blink = (
            len(eye_status_history) >= 3
            and eye_status_history[-3]       # open
            and not eye_status_history[-2]   # closed
            and eye_status_history[-1]       # open again
        )
        return blink, eyes_detected, eye_status_history

    # Public API
    @classmethod
    def analyze_emotions(cls, video_path: str) -> dict[str, Any]:
        from deepface import DeepFace  # lazy import - heavy dependency

        analyzer = cls()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Unable to open video file."}

        # Derive sampling interval from the video's native FPS
        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Analyze every Nth frame to approximate SAMPLE_FPS.
        # Clamp to at least 1 so we never skip more than source_fps frames.
        sample_every: int = max(1, round(source_fps / cls.SAMPLE_FPS))
        logger.info(
            "Video FPS=%.1f  sample_every=%d  effective_rate=%.2f fps",
            source_fps, sample_every, source_fps / sample_every,
        )

        frame_count = 0
        total_weighted_confidence = 0.0
        valid_frames = 0
        frame_details: list[dict] = []
        emotion_totals: dict[str, float] = {e: 0.0 for e in cls.EMOTION_WEIGHTS}

        blink_counter = 0
        last_blink_video_time = -float("inf")
        eye_status_history: list[bool] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Blink detection on every frame (cheap; needs temporal signal)
            try:
                blink, _, eye_status_history = analyzer._detect_blinks(
                    frame, eye_status_history
                )
                video_time = frame_count / source_fps
                if blink and video_time - last_blink_video_time > analyzer.blink_threshold:
                    blink_counter += 1
                    last_blink_video_time = video_time
            except Exception as exc:
                logger.debug("Frame %d blink error: %s", frame_count, exc)

            # Emotion analysis only on sampled frames
            if frame_count % sample_every != 0:
                continue

            dominant_emotion: str | None = None
            weighted_emotion_score = 0.0
            emotion_data: dict | None = None

            # Detect face ROI using OpenCV first to prevent aspect-ratio distortion or scale misses
            gray_sampled = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces_sampled = analyzer.face_cascade.detectMultiScale(gray_sampled, 1.3, 5, minSize=(40, 40))

            if len(faces_sampled) > 0:
                # Select the largest detected face
                faces_sorted = sorted(faces_sampled, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces_sorted[0]
                margin_y = int(fh * 0.1)
                margin_x = int(fw * 0.1)
                y1 = max(0, fy - margin_y)
                y2 = min(frame.shape[0], fy + fh + margin_y)
                x1 = max(0, fx - margin_x)
                x2 = min(frame.shape[1], fx + fw + margin_x)
                face_roi = frame[y1:y2, x1:x2]

                try:
                    analysis = DeepFace.analyze(
                        face_roi, actions=["emotion"], enforce_detection=False, silent=True
                    )
                    if analysis and isinstance(analysis, list):
                        emotions: dict[str, float] = {
                            k: float(v) for k, v in analysis[0]["emotion"].items()
                        }
                        total_conf = sum(emotions.values())
                        if total_conf > 0:
                            weighted_score = 0.0
                            for emotion, conf in emotions.items():
                                norm = conf / total_conf
                                if emotion in cls.EMOTION_WEIGHTS:
                                    weighted_score += norm * cls.EMOTION_WEIGHTS[emotion]
                                    emotion_totals[emotion] += norm

                            dominant_emotion = max(emotions, key=emotions.get)
                            weighted_emotion_score = weighted_score
                            emotion_data = emotions
                            total_weighted_confidence += weighted_score
                            valid_frames += 1

                except Exception as exc:
                    logger.debug("Frame %d face ROI emotion error: %s", frame_count, exc)

            # Sample frame detail (first 10 analyzed frames)
            if emotion_data and len(frame_details) < 10:
                frame_details.append(
                    {
                        "frame": int(frame_count),
                        "dominant_emotion": dominant_emotion,
                        "weighted_score": float(round(weighted_emotion_score, 4)),
                    }
                )

        cap.release()

        # Final scoring
        expected_sampled_frames = max(1, frame_count // sample_every)
        face_presence_rate = valid_frames / expected_sampled_frames

        # If no face was detected
        if valid_frames == 0 or (expected_sampled_frames >= 4 and valid_frames < 2):
            return {
                "confidence_score": 0.0,
                "frames_analyzed": int(valid_frames),
                "total_frames": int(frame_count),
                "blinks_per_minute": 0.0,
                "blink_rate_category": "no_face_detected",
                "speaking_rate_wpm": 0.0,
                "pacing_category": "no_speech_detected",
                "emotion_percentages": {},
                "frame_details": [],
            }

        video_duration_seconds = frame_count / source_fps if source_fps > 0 else 0.0
        total_minutes = video_duration_seconds / 60.0
        blinks_per_minute = blink_counter / total_minutes if total_minutes > 0 else 0.0

        blink_category = analyzer._categorize_blink_rate(blinks_per_minute)
        blink_conf_raw = cls.BLINK_RATE_CONFIDENCE[blink_category]

        # Multi-signal confidence calculation
        # 1. Emotion score (scaled 0-100)
        emotion_conf_raw = total_weighted_confidence / valid_frames
        emotion_conf_scaled = ((emotion_conf_raw + 1) / 2) * 100

        # 2. Blink score (scaled 0-100)
        blink_conf_scaled = ((blink_conf_raw + 1) / 2) * 100

        # 3. Face presence stability (percentage of expected sampled frames with clear face detection)
        expected_sampled_frames = max(1, frame_count // sample_every)
        face_presence_rate = min(1.0, valid_frames / expected_sampled_frames)
        face_stability_scaled = face_presence_rate * 100

        # Weighted combination: 55% facial composure, 35% eye contact, 10% camera presence
        combined = (
            0.55 * emotion_conf_scaled
            + 0.35 * blink_conf_scaled
            + 0.10 * face_stability_scaled
        )
        combined = max(0.0, min(100.0, combined))

        emotion_percentages = {
            e: float(round((emotion_totals[e] / valid_frames) * 100, 2))
            for e in emotion_totals
        }

        return {
            "confidence_score": float(round(combined, 2)),
            "frames_analyzed": int(valid_frames),
            "total_frames": int(frame_count),
            "blinks_per_minute": float(round(blinks_per_minute, 2)),
            "blink_rate_category": blink_category,
            "emotion_percentages": emotion_percentages,
            "frame_details": frame_details,
        }