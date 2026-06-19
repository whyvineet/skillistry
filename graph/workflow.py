import asyncio
import functools
import logging
from typing import Any

from langgraph.graph import StateGraph, END

from graph.state import InterviewState
from graph.nodes import (
    extract_audio_node,
    analyze_video_node,
    analyze_speech_node,
    analyze_content_node,
    route_after_audio,
)
from modules import SpeechAnalyzer, ContentAnalyzer

logger = logging.getLogger(__name__)


# Parallel fan-out node
def parallel_video_and_speech_node(
    state: InterviewState,
    speech_analyzer: SpeechAnalyzer,
) -> dict[str, Any]:

    async def _run() -> tuple[dict, dict]:
        loop = asyncio.get_event_loop()

        video_future = loop.run_in_executor(
            None,
            analyze_video_node,
            state,
        )
        speech_future = loop.run_in_executor(
            None,
            functools.partial(analyze_speech_node, speech_analyzer=speech_analyzer),
            state,
        )
        return await asyncio.gather(video_future, speech_future)

    # Run the coroutine.  asyncio.run() works fine here because LangGraph
    # invokes nodes synchronously; a new event loop is created per call.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async context (e.g. pytest-asyncio or some
            # ASGI middleware).  Fall back to sequential execution to avoid
            # nested-loop errors.
            logger.debug("Existing event loop detected; running branches sequentially.")
            video_result  = analyze_video_node(state)
            speech_result = functools.partial(
                analyze_speech_node, speech_analyzer=speech_analyzer
            )(state)
        else:
            video_result, speech_result = loop.run_until_complete(_run())
    except RuntimeError:
        # No current event loop – create one.
        video_result, speech_result = asyncio.run(_run())

    # Bail early if either branch hit a fatal error.
    if video_result.get("error"):
        return video_result
    if speech_result.get("error"):
        return speech_result

    return {**video_result, **speech_result}


# Graph builder
def build_workflow(
    speech_analyzer: SpeechAnalyzer,
    content_analyzer: ContentAnalyzer,
) -> Any:

    _parallel = functools.partial(
        parallel_video_and_speech_node, speech_analyzer=speech_analyzer
    )
    _analyze_content = functools.partial(
        analyze_content_node, content_analyzer=content_analyzer
    )

    graph = StateGraph(InterviewState)

    graph.add_node("extract_audio",          extract_audio_node)
    graph.add_node("analyze_video_speech",   _parallel)        # merged parallel node
    graph.add_node("analyze_content",        _analyze_content)

    graph.set_entry_point("extract_audio")

    graph.add_conditional_edges(
        "extract_audio",
        route_after_audio,
        {
            "ok": "analyze_video_speech",
            "error": END,
        },
    )

    graph.add_edge("analyze_video_speech", "analyze_content")
    graph.add_edge("analyze_content",      END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully (parallel video+speech branch).")
    return compiled