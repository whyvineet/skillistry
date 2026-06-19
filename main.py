import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from graph.workflow import build_workflow
from modules import SpeechAnalyzer, ContentAnalyzer
from utils.helpers import ensure_dir_exists

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# Lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup; release on shutdown."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    ollama_model: str = os.getenv("OLLAMA_MODEL")
    upload_folder: str = os.getenv("UPLOAD_FOLDER")

    logger.info("Connecting to Ollama at %s  model=%s", ollama_base_url, ollama_model)

    speech_analyzer = SpeechAnalyzer(
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )
    content_analyzer = ContentAnalyzer(
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )

    workflow = build_workflow(speech_analyzer, content_analyzer)

    ensure_dir_exists(upload_folder)

    # Attach to app state so routes can access them
    app.state.workflow = workflow
    app.state.upload_folder = upload_folder

    logger.info("Application startup complete.  Upload folder: %s", upload_folder)

    yield  # ← app is running

    logger.info("Application shutdown.")


# App factory
def create_app() -> FastAPI:
    _app = FastAPI(
        title="AI Interview Analyzer",
        description=(
            "Analyses a candidate's video interview response using computer vision "
            "(emotion + blink detection), speech processing (transcription + "
            "pronunciation), and an LLM (grammar, tone, relevance, answer quality)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS – open for development; restrict origins in production
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(router)

    return _app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app" , reload=True)