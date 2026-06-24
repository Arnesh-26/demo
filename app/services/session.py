import asyncio
from typing import Dict, Set, Optional
from fastapi import WebSocket
from app.services.audio_buffer import AudioSessionBuffer
from app.utils.logging import logger

class ParticipantStream:
    def __init__(self, session_id: str, role: str, websocket: WebSocket):
        self.session_id = session_id
        self.role = role
        self.websocket = websocket
        self.buffer = AudioSessionBuffer(session_id, role)
        self.connected_at = asyncio.get_event_loop().time()

class ConsultationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.streams: Dict[str, ParticipantStream] = {}  # Key: role (doctor/patient/attender)
        self.transcript_seq_counters: Dict[str, int] = {
            "doctor": 0,
            "patient": 0,
            "attender": 0
        }

    def get_and_increment_transcript_seq(self, role: str) -> int:
        """Atomically returns and increments the transcript sequence counter for a role."""
        val = self.transcript_seq_counters.get(role, 0)
        self.transcript_seq_counters[role] = val + 1
        return val


    def register_stream(self, role: str, websocket: WebSocket) -> ParticipantStream:
        """Registers a participant stream under the consultation session."""
        if role in self.streams:
            # Reconnect scenario: Clean up or overwrite the previous stream
            logger.info(
                "Participant reconnected; overwriting active stream",
                extra={"session_id": self.session_id, "role": role}
            )
            # Reuses the existing buffer or resets it. We reuse the buffer to preserve state.
            old_stream = self.streams[role]
            stream = ParticipantStream(self.session_id, role, websocket)
            # Carry over the buffer to avoid losing buffered data on reconnect
            stream.buffer = old_stream.buffer
            self.streams[role] = stream
        else:
            stream = ParticipantStream(self.session_id, role, websocket)
            self.streams[role] = stream
            
        return stream

    def remove_stream(self, role: str) -> None:
        """Removes a stream from the session."""
        if role in self.streams:
            del self.streams[role]

    def is_empty(self) -> bool:
        """Returns True if no participants are streaming."""
        return len(self.streams) == 0

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ConsultationSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str) -> ConsultationSession:
        async with self._lock:
            if session_id not in self._sessions:
                logger.info("Created new consultation session", extra={"session_id": session_id})
                self._sessions[session_id] = ConsultationSession(session_id)
            return self._sessions[session_id]

    async def unregister_stream(self, session_id: str, role: str) -> None:
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.remove_stream(role)
                logger.info("Unregistered participant stream", extra={"session_id": session_id, "role": role})

    async def clean_empty_sessions(self, active_session_ids_with_workers: Set[str]) -> None:
        """Deletes sessions that have no active streams and no running worker tasks."""
        async with self._lock:
            for session_id in list(self._sessions.keys()):
                session = self._sessions[session_id]
                if session.is_empty() and session_id not in active_session_ids_with_workers:
                    del self._sessions[session_id]
                    logger.info(
                        "Consultation session closed (no active streams or workers)",
                        extra={"session_id": session_id}
                    )
                    # Cleanup parallel diarization pipeline resources
                    try:
                        from app.services.diarization_worker import diarization_worker_manager
                        from app.services.speaker_timeline import speaker_timeline_manager
                        diarization_worker_manager.clear_session(session_id)
                        speaker_timeline_manager.clear_timeline(session_id)
                    except Exception as e:
                        logger.error(
                            "SessionManager: Error cleaning up parallel diarization resources",
                            exc_info=True,
                            extra={"session_id": session_id, "error": str(e)}
                        )


    def get_session(self, session_id: str) -> Optional[ConsultationSession]:
        return self._sessions.get(session_id)

    def list_active_sessions(self) -> Set[str]:
        return set(self._sessions.keys())

session_manager = SessionManager()
