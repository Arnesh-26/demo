from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.websocket import router as ws_router
from app.services.session import session_manager
from app.services.stt_manager import stt_manager
from app.utils.logging import logger
import asyncio
import time

app = FastAPI(
    title=settings.APP_NAME,
    description="Low-latency real-time PCM audio streaming ingestion layer.",
    version="1.0.0"
)

# Standard CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production domain restrictions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ws_router, prefix=settings.API_V1_STR)

# Startup logging
@app.on_event("startup")
async def startup_event():
    logger.info(
        "Application starting up",
        extra={
            "app_name": settings.APP_NAME,
            "sample_rate_hz": settings.AUDIO_SAMPLE_RATE,
            "min_frame_bytes": settings.min_frame_bytes,
            "max_frame_bytes": settings.max_frame_bytes,
            "buffer_chunk_bytes": settings.buffer_chunk_bytes,
            "diarization_mode": settings.DIARIZATION_MODE
        }
    )
    # Start background STT worker reconciliation orchestrator loop
    stt_manager.start_orchestrator()

    # Pre-warm the Pyannote pipeline in production mode so the first
    # inference window incurs no cold-start delay.
    if settings.DIARIZATION_MODE == "production":
        try:
            from app.services.diarization_worker import diarization_worker_manager
            await diarization_worker_manager.preload_pipeline()
            logger.info("DiarizationWorker: Pipeline pre-warmed successfully at startup.")
        except Exception:
            logger.error(
                "DiarizationWorker: Failed to pre-warm Pyannote pipeline at startup.",
                exc_info=True,
            )

# Shutdown logging
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
    # Clean up and stop background STT workers
    await stt_manager.stop_orchestrator()

# Production Health Check Endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Returns app health, including count of active sessions for orchestration routing.
    """
    active_sessions = list(session_manager.list_active_sessions())
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions_count": len(active_sessions),
        "active_sessions": active_sessions
    }

# Dynamic Verification Endpoint for Step 2 Transcripts
@app.get("/v1/transcripts/{session_id}", status_code=status.HTTP_200_OK)
async def get_transcripts(session_id: str):
    """
    Retrieves buffered transcript events for a given session.
    """
    from app.services.transcript_bus import transcript_bus
    events = transcript_bus.get_recent_events(session_id)
    return [event.dict() for event in events]
