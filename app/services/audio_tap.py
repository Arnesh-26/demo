# =====================================================================
# AUDIO TAP ROUTER (app/services/audio_tap.py)
# =====================================================================
# Purpose: Minimally routes raw PCM chunks and EOF sentinels from the
#          AudioSessionBuffer and WebSocket handler to the parallel
#          diarization pipeline asynchronously using a fire-and-forget
#          queue, preserving STT ingestion latency.
# =====================================================================

import asyncio
import threading
from app.utils.logging import logger

class AudioTapService:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._consumer_task = None
        self._lock = threading.Lock()

    def start_consumer(self) -> None:
        """Starts the background queue consumer if not already running."""
        if self._consumer_task is None or self._consumer_task.done():
            with self._lock:
                if self._consumer_task is None or self._consumer_task.done():
                    self._consumer_task = asyncio.create_task(self._consume_loop())

    def feed_audio(self, session_id: str, chunk: bytes) -> None:
        """Puts the audio chunk into the fire-and-forget queue."""
        self.start_consumer()
        try:
            self._queue.put_nowait(("audio", session_id, chunk))
        except Exception as e:
            logger.error(
                "AudioTapService: put_nowait failed for audio chunk",
                exc_info=True,
                extra={"session_id": session_id, "error": str(e)}
            )

    def feed_sentinel(self, session_id: str) -> None:
        """Puts the end-of-stream sentinel into the fire-and-forget queue."""
        self.start_consumer()
        try:
            self._queue.put_nowait(("sentinel", session_id, None))
        except Exception as e:
            logger.error(
                "AudioTapService: put_nowait failed for sentinel",
                exc_info=True,
                extra={"session_id": session_id, "error": str(e)}
            )

    async def _consume_loop(self) -> None:
        logger.info("AudioTapService: Background consumer loop started")
        while True:
            try:
                msg_type, session_id, chunk = await self._queue.get()
                try:
                    from app.services.diarization_worker import diarization_worker_manager
                    if msg_type == "sentinel":
                        await diarization_worker_manager.stop_worker(session_id)
                    else:
                        diarization_worker_manager.feed_audio(session_id, chunk)
                except Exception as ex:
                    # Capture exceptions so they don't crash the consumer or block other sessions
                    logger.error(
                        "AudioTapService: Error forwarding task in consumer",
                        exc_info=True,
                        extra={"session_id": session_id, "error": str(ex)}
                    )
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                logger.info("AudioTapService: Background consumer loop cancelled")
                break
            except Exception as e:
                logger.error("AudioTapService: Critical error in consumer loop", exc_info=True)
                await asyncio.sleep(0.1)  # Prevent hot loop on critical failure

# Singleton Instance
audio_tap = AudioTapService()
