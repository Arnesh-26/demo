import asyncio
import time
from typing import Dict, Optional, Generator
from app.core.config import settings
from app.utils.logging import logger

class InvalidAudioFrameError(ValueError):
    """Raised when an incoming audio frame fails validation checks."""
    pass

class AudioChunk(bytes):
    """
    Additive wrapper class representing a processed chunk of audio.
    Subclasses bytes to remain 100% backward compatible with downstream consumers expecting raw bytes.
    """
    def __new__(cls, data: bytes, session_id: str, role: str, sequence_number: int, timestamp: float):
        obj = super().__new__(cls, data)
        obj.session_id = session_id
        obj.role = role
        obj.sequence_number = sequence_number
        obj.timestamp = timestamp
        return obj

    def __repr__(self) -> str:
        return (
            f"AudioChunk(session_id={self.session_id!r}, role={self.role!r}, "
            f"seq={self.sequence_number}, ts={self.timestamp}, len={len(self)})"
        )

class AudioFrameValidator:
    @staticmethod
    def validate_pcm_frame(data: bytes) -> None:
        """
        Validates that the incoming byte buffer aligns with PCM 16-bit, mono, 16kHz specifications
        and fits within the expected 20ms - 50ms frame size window.
        """
        frame_size = len(data)
        min_size = settings.min_frame_bytes
        max_size = settings.max_frame_bytes
        
        # Check size constraints
        if frame_size < min_size or frame_size > max_size:
            raise InvalidAudioFrameError(
                f"Audio frame size of {frame_size} bytes is outside the allowed range of "
                f"[{min_size}, {max_size}] bytes (corresponding to {settings.MIN_FRAME_DURATION_MS}ms "
                f"and {settings.MAX_FRAME_DURATION_MS}ms frames)."
            )
            
        # Check alignment: 16-bit audio must have a frame size that is a multiple of 2 bytes
        if frame_size % settings.bytes_per_sample != 0:
            raise InvalidAudioFrameError(
                f"Audio frame size of {frame_size} bytes is not aligned with 16-bit PCM "
                f"({settings.bytes_per_sample}-byte sample boundary)."
            )

class AudioSessionBuffer:
    def __init__(self, session_id: str, role: str):
        self.session_id = session_id
        self.role = role
        self.raw_buffer = bytearray()
        self.ready_chunks_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.QUEUE_MAX_SIZE)
        self.total_bytes_received: int = 0
        self.total_frames_received: int = 0
        self.expected_seq: int = 0
        self.dropped_packets_counter: int = 0
        self.chunk_seq_counter: int = 0
        self.queue_overflows_counter: int = 0

    async def append_frame(self, data: bytes, seq: Optional[int] = None) -> None:
        """
        Validates the frame, tracks sequence indicators to count drops, and appends to the session buffer.
        When the buffer accumulates enough bytes (e.g. 1.0 second), a chunk is released to the ready queue.
        """
        # Validate frame structure
        AudioFrameValidator.validate_pcm_frame(data)
        
        # Track sequence to measure packet drops
        if seq is not None:
            if seq != self.expected_seq:
                diff = seq - self.expected_seq
                if diff > 0:
                    self.dropped_packets_counter += diff
                    logger.warning(
                        "Detected packet drop", 
                        extra={
                            "session_id": self.session_id,
                            "role": self.role,
                            "expected_seq": self.expected_seq,
                            "received_seq": seq,
                            "dropped_count": diff
                        }
                    )
            self.expected_seq = seq + 1
        else:
            self.expected_seq += 1

        # Memory threshold check to prevent Denial of Service / RAM exhaustion
        if len(self.raw_buffer) + len(data) > settings.MAX_BUFFER_MEM_LIMIT_BYTES:
            logger.error(
                "Buffer memory limit exceeded, clearing buffer",
                extra={"session_id": self.session_id, "role": self.role, "limit_bytes": settings.MAX_BUFFER_MEM_LIMIT_BYTES}
            )
            self.raw_buffer.clear()
            raise InvalidAudioFrameError("Session buffer memory threshold exceeded.")

        # Append to buffer
        self.raw_buffer.extend(data)
        self.total_bytes_received += len(data)
        self.total_frames_received += 1

        # Extract larger chunks once the configured duration (e.g. 1.0 sec) is met
        target_chunk_bytes = settings.buffer_chunk_bytes
        while len(self.raw_buffer) >= target_chunk_bytes:
            raw_chunk = bytes(self.raw_buffer[:target_chunk_bytes])
            # Remove chunk from buffer
            del self.raw_buffer[:target_chunk_bytes]
            
            # Wrap as AudioChunk (subclass of bytes) to maintain absolute backward compatibility
            chunk = AudioChunk(
                raw_chunk,
                self.session_id,
                self.role,
                self.chunk_seq_counter,
                time.time()
            )
            self.chunk_seq_counter += 1
            
            # Enqueue using backpressure policies
            strategy = settings.QUEUE_OVERFLOW_STRATEGY
            if self.ready_chunks_queue.full():
                self.queue_overflows_counter += 1
                if strategy == "drop_oldest":
                    try:
                        dropped = self.ready_chunks_queue.get_nowait()
                        logger.warning(
                            "Queue overflow: dropped oldest chunk",
                            extra={
                                "session_id": self.session_id,
                                "role": self.role,
                                "dropped_chunk_seq": getattr(dropped, "sequence_number", None),
                                "strategy": strategy
                            }
                        )
                    except asyncio.QueueEmpty:
                        pass
                    await self.ready_chunks_queue.put(chunk)
                elif strategy == "block":
                    logger.warning(
                        "Queue full: blocking ingestion to apply backpressure",
                        extra={
                            "session_id": self.session_id,
                            "role": self.role,
                            "strategy": strategy
                        }
                    )
                    await self.ready_chunks_queue.put(chunk)
                else:  # "ignore_and_warn"
                    logger.warning(
                        "Queue full: discarding incoming chunk",
                        extra={
                            "session_id": self.session_id,
                            "role": self.role,
                            "discarded_chunk_seq": chunk.sequence_number,
                            "strategy": strategy
                        }
                    )
            else:
                await self.ready_chunks_queue.put(chunk)
            
            # Feed audio tap in parallel for speaker diarization
            from app.services.audio_tap import audio_tap
            audio_tap.feed_audio(self.session_id, chunk)
            
            logger.debug(
                "Released audio chunk for transcription",
                extra={
                    "session_id": self.session_id, 
                    "role": self.role, 
                    "chunk_size_bytes": len(chunk),
                    "queue_size": self.ready_chunks_queue.qsize()
                }
            )

    async def get_next_chunk(self) -> AudioChunk:
        """
        Retrieves the next aggregated audio chunk from the queue.
        Blocks until a chunk is available.
        """
        return await self.ready_chunks_queue.get()

    def flush(self) -> Optional[AudioChunk]:
        """
        Flushes any remaining audio frames in the buffer, even if the total size
        is less than the target chunk size. Useful when ending a session.
        """
        if len(self.raw_buffer) > 0:
            remaining_chunk = bytes(self.raw_buffer)
            self.raw_buffer.clear()
            chunk = AudioChunk(
                remaining_chunk,
                self.session_id,
                self.role,
                self.chunk_seq_counter,
                time.time()
            )
            self.chunk_seq_counter += 1
            
            # Feed audio tap in parallel for speaker diarization
            from app.services.audio_tap import audio_tap
            audio_tap.feed_audio(self.session_id, chunk)
            
            return chunk
        return None
