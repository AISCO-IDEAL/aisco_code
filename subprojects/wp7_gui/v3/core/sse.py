"""Server-Sent Events helpers.

Provides a thin wrapper so route handlers only deal with typed event dicts
rather than raw SSE wire format.

Usage in a route:
    from fastapi.responses import StreamingResponse
    from subprojects.wp7_gui.v3.core.sse import event, stream_response

    async def my_generator():
        yield event("log", {"line": "hello"})
        yield event("epoch", {"epoch": 1})

    return stream_response(my_generator())
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse


def event(event_type: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def stream_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
