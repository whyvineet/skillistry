import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def _ffmpeg_exe() -> str:
    """
    Return the path to a working ffmpeg binary.

    Preference order:
    1. imageio_ffmpeg bundled binary  (ships with moviepy — always present)
    2. System PATH 'ffmpeg'           (fallback if imageio is somehow absent)
    """
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


class AudioProcessor:

    @staticmethod
    def extract_audio(video_path: str) -> str | None:
        """
        Extract a 16 kHz mono WAV file from *video_path*.

        Uses the ffmpeg binary bundled with imageio_ffmpeg (a moviepy
        dependency) so no separate system ffmpeg installation is required.

        Falls back to MoviePy if the direct subprocess call fails, and
        finally falls back to a bare 'ffmpeg' system call.
        """
        audio_path = f"{os.path.splitext(video_path)[0]}.wav"

        # Primary: direct subprocess with bundled ffmpeg
        # This is the most reliable path for Chrome/Firefox webm recordings.
        ffmpeg = _ffmpeg_exe()
        try:
            cmd = [
                ffmpeg, "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path,
            ]
            result = subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Audio extracted via bundled ffmpeg → %s", audio_path)
            return audio_path

        except subprocess.CalledProcessError as exc:
            logger.warning(
                "bundled ffmpeg failed (exit %d): %s; trying MoviePy …",
                exc.returncode,
                exc.stderr.decode(errors="replace")[:300],
            )
        except Exception as exc:
            logger.warning("bundled ffmpeg unavailable (%s); trying MoviePy …", exc)

        # Secondary: MoviePy (uses its own ffmpeg internally)
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
                logger=None,
            )
            clip.close()
            logger.info("Audio extracted via MoviePy → %s", audio_path)
            return audio_path

        except Exception as moviepy_err:
            logger.warning("MoviePy extraction failed (%s); trying bare ffmpeg …", moviepy_err)

        # Last resort: system ffmpeg on PATH
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
            logger.info("Audio extracted via system ffmpeg → %s", audio_path)
            return audio_path

        except Exception as fallback_err:
            logger.error("All audio extraction methods failed: %s", fallback_err)
            return None