import json
import base64
import asyncio
import websockets
from typing import Optional
from app.services.providers.base_stt import BaseSTTProvider
from app.core.config import settings
from app.utils.logging import logger

class ScribeV2Provider(BaseSTTProvider):
    """
    STT Provider implementation for ElevenLabs Scribe V2 streaming Speech-to-Text.
    Supports secure WebSocket streaming, automatic reconnects, and a mock development fallback.
    """
    def __init__(self, api_key: str, mode: str = "development"):
        self.api_key = api_key
        self.mode = mode
        self.ws_url = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime&include_timestamps=true"
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._event_seq_counter = 0
        self._is_connected = False
        self._commit_requested = False
        
        # Mock mode attributes
        self._mock_buffer_bytes = 0
        self._mock_chunk_count = 0

    @property
    def commit_requested(self) -> bool:
        """Returns True if a commit signal has been requested for this session."""
        return self._commit_requested


    def get_and_increment_event_seq(self) -> int:
        """Returns and increments the sequential index for transcript events."""
        val = self._event_seq_counter
        self._event_seq_counter += 1
        return val

    async def connect(self) -> None:
        """Establishes connection to the transcription service."""
        if self.mode == "development" and not self.api_key:
            logger.info("Scribe V2: Initializing in DEVELOPMENT MOCK mode (No API Key).")
            self._is_connected = True
            return

        # Production / Real connection flow
        await self._connect_with_retry()

    async def _connect_with_retry(self) -> None:
        """Establishes the WebSocket connection with exponential backoff retry logic."""
        max_retries = 5
        base_delay = 1.0  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Scribe V2: Attempting connection to ElevenLabs (Attempt {attempt}/{max_retries})..."
                )
                headers = {"xi-api-key": self.api_key}
                try:
                    self._websocket = await websockets.connect(
                        self.ws_url,
                        additional_headers=headers
                    )
                except TypeError:
                    self._websocket = await websockets.connect(
                        self.ws_url,
                        extra_headers=headers
                    )

                
                # Wait for session_started confirmation handshake
                handshake_msg = await self._websocket.recv()
                handshake_data = json.loads(handshake_msg)
                
                if handshake_data.get("message_type") == "session_started":
                    logger.info(
                        "Scribe V2: Connection handshake successful",
                        extra={"session_id": handshake_data.get("session_id")}
                    )
                    self._is_connected = True
                    return
                else:
                    raise RuntimeError(
                        f"Unexpected handshake message type: {handshake_data.get('message_type')}"
                    )
                    
            except Exception as e:
                delay = base_delay * (2 ** (attempt - 1))
                logger.error(
                    f"Scribe V2: Connection failed (Attempt {attempt}): {str(e)}. Retrying in {delay}s.",
                    exc_info=True
                )
                if self._websocket:
                    try:
                        await self._websocket.close()
                    except Exception:
                        pass
                await asyncio.sleep(delay)
                
        raise ConnectionError("Scribe V2: Failed to connect to ElevenLabs Scribe WebSocket after max retries.")

    async def send_audio(self, chunk: bytes) -> None:
        """Sends an audio chunk to Scribe."""
        if not self._is_connected:
            logger.warning("Scribe V2: Cannot send audio, connection inactive. Retrying reconnect...")
            await self._connect_with_retry()

        if self.mode == "development" and not self.api_key:
            # Mock mode: accumulate stats
            self._mock_buffer_bytes += len(chunk)
            self._mock_chunk_count += 1
            return

        # Pack and send over WebSocket (base64 JSON wrapper)
        try:
            base64_str = base64.b64encode(chunk).decode("utf-8")
            payload = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64_str,
                "commit": False
            }
            await self._websocket.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"Scribe V2: Send failed ({str(e)}). Flagging connection inactive.", exc_info=True)
            self._is_connected = False
            raise e

    async def send_commit(self) -> None:
        """Sends a commit message to finalize the transcription."""
        self._commit_requested = True
        if not self._is_connected:
            return
        if self.mode == "development" and not self.api_key:
            return
            
        try:
            payload = {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True
            }
            await self._websocket.send(json.dumps(payload))
            logger.info("Scribe V2: Sent commit message to ElevenLabs API.")
        except Exception as e:
            logger.error(f"Scribe V2: Commit failed ({str(e)})", exc_info=True)

    async def receive(self) -> dict:
        """Receives real-time transcriptions/metadata from the provider."""
        if self.mode == "development" and not self.api_key:
            # Simulated transcription delivery loop
            await asyncio.sleep(1.0)  # mimic response interval
            
            if self._mock_chunk_count > 0 or self._commit_requested:
                self._mock_chunk_count = 0
                mock_text = f"Mock transcription text for chunk {self._event_seq_counter}"
                # If commit was requested, force is_committed = True
                is_committed = self._commit_requested or (self._event_seq_counter % 2 == 1)
                
                words_list = []
                if is_committed:
                    words = mock_text.split()
                    # Each chunk is 1 second, simulate words within that offset
                    start_offset = float(self._event_seq_counter) * 1.0
                    word_duration = 1.0 / max(1, len(words))
                    for i, w in enumerate(words):
                        words_list.append({
                            "text": w,
                            "start": start_offset + (i * word_duration),
                            "end": start_offset + ((i + 1) * word_duration)
                        })
                
                return {
                    "type": "committed" if is_committed else "partial",
                    "text": mock_text,
                    "confidence": 0.95,
                    "words": words_list
                }
            return {"type": "ignored"}

        # Real receiver loop
        try:
            msg = await self._websocket.recv()
            data = json.loads(msg)
            msg_type = data.get("message_type")
            
            if msg_type == "partial_transcript":
                return {
                    "type": "partial",
                    "text": data.get("text", ""),
                    "confidence": 0.90
                }
            elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                return {
                    "type": "committed",
                    "text": data.get("text", ""),
                    "confidence": 0.97,
                    "words": data.get("words", [])
                }
            return {"type": "ignored"}
            
        except Exception as e:
            logger.error(f"Scribe V2: Receive failed ({str(e)}). Flagging connection inactive.", exc_info=True)
            self._is_connected = False
            raise e

    async def disconnect(self) -> None:
        """Cleans up resources and closes the connection."""
        logger.info("Scribe V2: Disconnecting provider.")
        self._is_connected = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.error(f"Scribe V2: Error closing websocket: {str(e)}")
            self._websocket = None
