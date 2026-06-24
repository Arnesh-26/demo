from pydantic import BaseModel
from typing import Optional

class TranscriptEvent(BaseModel):
    """
    Standardized event schema representing a real-time transcription block.
    """
    session_id: str
    role: str
    sequence_number: int
    timestamp: float
    transcript: str
    is_partial: bool
    is_final: bool
    confidence: Optional[float] = None
    provider: str = "scribe_v2"
    
    # Decoupled speaker diarization and word timestamp fields
    event_id: str  # Unique event ID for database tracking and UPSERTs
    speaker_id: Optional[str] = None
    speaker_label: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: Optional[str] = None
    
    # Versioning and update tracking for database persistence and event consumers
    version: int = 1
    updated_at: Optional[float] = None

