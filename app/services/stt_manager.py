import asyncio
from typing import Dict, Tuple, Set, Optional
from app.services.session import session_manager, ParticipantStream
from app.services.providers.scribe_v2 import ScribeV2Provider
from app.services.stt_worker import stt_worker_task
from app.core.config import settings
from app.utils.logging import logger

class STTManager:
    """
    STTManager coordinates Speech-to-Text workers for active streams.
    Runs a reconciliation loop that automatically starts/stops workers 
    to match the current state of session_manager without monkey-patching.
    """
    def __init__(self):
        # Key: (session_id, role) -> Value: (asyncio.Task, ScribeV2Provider)
        self._workers: Dict[Tuple[str, str], Tuple[asyncio.Task, ScribeV2Provider]] = {}
        self._is_orchestrating = False
        self._orchestrator_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def start_orchestrator(self) -> None:
        """Starts the background reconciliation orchestrator task."""
        if not self._is_orchestrating:
            self._is_orchestrating = True
            self._orchestrator_task = asyncio.create_task(self._orchestrator_loop())
            logger.info("STTManager: Orchestrator loop started.")

    async def stop_orchestrator(self) -> None:
        """Stops the background reconciliation orchestrator task and cleans up active workers."""
        self._is_orchestrating = False
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
            self._orchestrator_task = None
            
        # Clean up active workers
        async with self._lock:
            active_keys = list(self._workers.keys())
            for key in active_keys:
                await self._stop_worker_by_key(key)
        logger.info("STTManager: Orchestrator loop stopped.")

    async def _stop_worker_by_key(self, key: Tuple[str, str]) -> None:
        if key in self._workers:
            task, provider = self._workers[key]
            logger.info("STTManager: Stopping worker task", extra={"session_id": key[0], "role": key[1]})
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._workers[key]

    async def reconcile(self) -> None:
        """
        Reconciliation step: compares session_manager streams with active workers,
        starting missing workers and terminating orphaned ones.
        """
        async with self._lock:
            # 1. Collect currently active streams in session_manager
            active_streams: Dict[Tuple[str, str], ParticipantStream] = {}
            for session_id in list(session_manager.list_active_sessions()):
                session = session_manager.get_session(session_id)
                if session:
                    for role, stream in list(session.streams.items()):
                        active_streams[(session_id, role)] = stream

            # 2. Reconcile: Start workers for new streams
            for key, stream in active_streams.items():
                if key not in self._workers:
                    session_id, role = key
                    logger.info("STTManager: Reconciled new stream. Starting worker.", extra={"session_id": session_id, "role": role})
                    
                    provider = ScribeV2Provider(
                        api_key=settings.ELEVENLABS_API_KEY,
                        mode=settings.STT_PROVIDER_MODE
                    )
                    task = asyncio.create_task(stt_worker_task(stream, provider))
                    self._workers[key] = (task, provider)

            # 3. Reconcile: Clean up finished workers, and handle active vs inactive
            registered_keys = list(self._workers.keys())
            for key in registered_keys:
                task, provider = self._workers[key]
                if task.done():
                    session_id, role = key
                    logger.info("STTManager: Worker task completed. Cleaning up.", extra={"session_id": session_id, "role": role})
                    try:
                        await task  # propagate any exceptions
                    except Exception as e:
                        logger.error("STTManager: Worker task failed", exc_info=True, extra={"session_id": session_id, "role": role})
                    del self._workers[key]

            # 4. Clean up any empty sessions that have no running workers anymore
            active_session_ids_with_workers = {k[0] for k in self._workers.keys()}
            await session_manager.clean_empty_sessions(active_session_ids_with_workers)



    async def _orchestrator_loop(self) -> None:
        """Infinite loop polling for reconciliation changes."""
        while self._is_orchestrating:
            try:
                await self.reconcile()
            except Exception as e:
                logger.error("STTManager: Error in reconciliation loop", exc_info=True)
            await asyncio.sleep(0.1)  # Check every 100ms

# Singleton STT Manager
stt_manager = STTManager()
