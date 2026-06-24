# =====================================================================
# BACKGROUND STT WORKER LOOP (app/services/stt_worker.py)
# =====================================================================
# Purpose: Pipes client audio chunks to ElevenLabs Scribe and receives
#          transcripts back in the background.
#
# Recent Modifications:
# 1. Added a 10-second timeout guard to provider.receive() using wait_for().
# 2. If the connection hangs for more than 10 seconds:
#    - Logs a warning message.
#    - Runs provider.disconnect() and provider.connect() to reconnect.
#    - Jumps back to wait for the next transcript event without crashing.
# =====================================================================

import asyncio
import time
from app.services.providers.models import TranscriptEvent
from app.services.transcript_bus import transcript_bus
from app.utils.logging import logger

async def receive_loop(session_id: str, role: str, provider) -> None:
    """
    Background loop that continuously listens for transcription events
    from Scribe V2 and publishes them to the TranscriptEventBus.
    """
    logger.info("STTWorker: Starting Scribe receive loop", extra={"session_id": session_id, "role": role})
    try:
        while True:
            # Wait for up to 10 seconds to receive a transcript event from ElevenLabs
            try:
                res = await asyncio.wait_for(provider.receive(), timeout=10.0)
            except TimeoutError:
                # If commit was requested, a timeout means we should exit instead of reconnecting
                if getattr(provider, "commit_requested", False):
                    logger.info(
                        "STTWorker: Timeout after commit request. Exiting receive loop.",
                        extra={"session_id": session_id, "role": role}
                    )
                    break
                # If 10 seconds pass with no message, assume connection hung and reconnect
                logger.warning(
                    "STTWorker: Scribe receive timeout occurred, reconnecting",
                    extra={"session_id": session_id, "role": role}
                )
                try:
                    await provider.disconnect()
                except Exception as e:
                    logger.error(
                        "STTWorker: Error disconnecting during reconnect",
                        exc_info=True,
                        extra={"session_id": session_id, "role": role, "error": str(e)}
                    )
                try:
                    await provider.connect()
                except Exception as e:
                    logger.error(
                        "STTWorker: Error connecting during reconnect",
                        exc_info=True,
                        extra={"session_id": session_id, "role": role, "error": str(e)}
                    )
                # Retry receiving on the new connection
                continue

            if res.get("type") == "ignored":
                continue
                
            from app.services.speaker_alignment import speaker_alignment_service
            events = speaker_alignment_service.align_and_segment(session_id, role, res, provider)
            for event in events:
                logger.info(
                    f"STTWorker: Broadcasting transcript event ({'FINAL' if event.is_final else 'PARTIAL'})",
                    extra={
                        "session_id": session_id,
                        "role": role,
                        "seq": event.sequence_number,
                        "text": event.transcript,
                        "speaker": event.speaker_id
                    }
                )
                transcript_bus.publish(event)
            
            # If commit was requested and we got the final committed transcript, we exit the loop
            is_commit_done = any(event.is_final for event in events)
            if getattr(provider, "commit_requested", False) and is_commit_done:
                logger.info(
                    "STTWorker: Received final committed transcript. Exiting receive loop.",
                    extra={"session_id": session_id, "role": role}
                )
                break

            
    except asyncio.CancelledError:
        logger.info("STTWorker: Scribe receive loop cancelled", extra={"session_id": session_id, "role": role})
    except Exception as e:
        logger.error(
            "STTWorker: Error in Scribe receive loop",
            exc_info=True,
            extra={"session_id": session_id, "role": role, "error": str(e)}
        )

async def stt_worker_task(stream, provider) -> None:
    """
    Main async worker task that consumes validated audio chunks from
    the participant's queue and pipes them to the STT provider.
    """
    session_id = stream.session_id
    role = stream.role
    queue = stream.buffer.ready_chunks_queue
    
    logger.info("STTWorker: Starting STT ingestion worker loop", extra={"session_id": session_id, "role": role})
    
    # Establish initial connection
    try:
        await provider.connect()
    except Exception as e:
        logger.error(
            "STTWorker: Failed to connect to Scribe during setup. Worker terminating.",
            exc_info=True,
            extra={"session_id": session_id, "role": role}
        )
        return

    # Start the receive loop concurrently
    receive_task = asyncio.create_task(receive_loop(session_id, role, provider))
    
    try:
        while True:
            # Wait for next aggregated audio chunk (carrying metadata)
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            
            try:
                # Pipe raw PCM bytes to Scribe
                await provider.send_audio(chunk)
            except Exception as e:
                logger.error(
                    "STTWorker: Failed to stream chunk to Scribe",
                    exc_info=True,
                    extra={
                        "session_id": session_id,
                        "role": role,
                        "chunk_seq": getattr(chunk, "sequence_number", None),
                        "error": str(e)
                    }
                )
                # If streaming fails, we sleep briefly before next try
                await asyncio.sleep(0.5)
            finally:
                queue.task_done()

        # Draining finished! Send commit to provider.
        await provider.send_commit()

        # Wait for the receive_task to finish naturally with a 5.0 second safety timeout
        try:
            await asyncio.wait_for(receive_task, timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "STTWorker: Timeout waiting for receive task to complete naturally",
                extra={"session_id": session_id, "role": role}
            )
                
    except asyncio.CancelledError:
        logger.info("STTWorker: Ingestion worker task cancelled", extra={"session_id": session_id, "role": role})
    except Exception as e:
        logger.error(
            "STTWorker: Critical error in ingestion worker task",
            exc_info=True,
            extra={"session_id": session_id, "role": role}
        )
    finally:
        # Stop receive loop if it's not already finished
        if not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        # Clean up provider connection
        await provider.disconnect()

