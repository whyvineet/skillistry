import os
import subprocess
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:

    @staticmethod
    def extract_audio(video_path: str) -> str | None:
        audio_path = f"{os.path.splitext(video_path)[0]}.wav"

        # primary: MoviePy
        try:
            import moviepy as mp  # type: ignore

            clip = mp.VideoFileClip(video_path)
            if clip.audio is None:
                logger.warning("No audio stream found in %s", video_path)
                clip.close()
                return None

            clip.audio.write_audiofile(
                audio_path,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],
                logger=None,  # suppress moviepy progress bar
            )
            clip.close()
            logger.info("Audio extracted via MoviePy → %s", audio_path)
            return audio_path

        except Exception as primary_err:
            logger.warning("MoviePy extraction failed (%s); trying ffmpeg …", primary_err)

        # fallback: direct ffmpeg subprocess
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Audio extracted via ffmpeg fallback → %s", audio_path)
            return audio_path

        except Exception as fallback_err:
            logger.error("ffmpeg fallback also failed: %s", fallback_err)
            return None