# =====================================================================
# CONFIGURATION SETTINGS (app/core/config.py)
# =====================================================================
# Purpose: Holds all global settings loaded via pydantic-settings.
#
# Recent Modifications:
# 1. Added `TRANSCRIPT_BUFFER_SIZE` configuration setting (defaulting to 500).
#    This defines the max sliding window size for the transcript event bus
#    in-memory cache buffers.
# =====================================================================

from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import List

class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "Clinical Consultation Audio Ingestion"
    API_V1_STR: str = "/v1"
    
    # Audio Parameters
    AUDIO_SAMPLE_RATE: int = 16000  # 16 kHz
    AUDIO_CHANNELS: int = 1         # Mono
    AUDIO_BITS_PER_SAMPLE: int = 16 # 16-bit PCM
    
    # Audio Frame Sizes in Milliseconds (20ms to 50ms)
    MIN_FRAME_DURATION_MS: int = 20
    MAX_FRAME_DURATION_MS: int = 50
    
    # Buffer Configuration
    # We buffer incoming client frames into larger chunks (e.g. 1 second of audio)
    # before they are ready for Scribe V2 transcription layer to optimize latency vs payload size.
    BUFFER_CHUNK_DURATION_SEC: float = 1.0
    
    # Derived Audio Calculation Properties
    @property
    def bytes_per_sample(self) -> int:
        return self.AUDIO_BITS_PER_SAMPLE // 8

    @property
    def min_frame_bytes(self) -> int:
        # samples_per_ms = 16000 / 1000 = 16 samples/ms
        # min_frame_samples = 16 * 20 = 320 samples
        # min_frame_bytes = 320 * 2 = 640 bytes
        samples_per_ms = self.AUDIO_SAMPLE_RATE // 1000
        return samples_per_ms * self.MIN_FRAME_DURATION_MS * self.bytes_per_sample

    @property
    def max_frame_bytes(self) -> int:
        # max_frame_samples = 16 * 50 = 800 samples
        # max_frame_bytes = 800 * 2 = 1600 bytes
        samples_per_ms = self.AUDIO_SAMPLE_RATE // 1000
        return samples_per_ms * self.MAX_FRAME_DURATION_MS * self.bytes_per_sample

    @property
    def buffer_chunk_bytes(self) -> int:
        # 1.0s * 16000 samples/sec * 2 bytes/sample = 32000 bytes
        return int(self.BUFFER_CHUNK_DURATION_SEC * self.AUDIO_SAMPLE_RATE * self.bytes_per_sample)

    # Session Management
    MAX_ACTIVE_SESSIONS: int = 1000
    SESSION_TIMEOUT_SEC: int = 300  # Automatically close connection after inactivity
    MAX_BUFFER_MEM_LIMIT_BYTES: int = 10 * 1024 * 1024  # 10MB safety threshold per connection
    
    # Security / Auth
    AUTH_SECRET_TOKEN: str = "production-secure-token-change-me"
    ALLOWED_ROLES: List[str] = ["doctor", "patient", "attender"]
    
    # JWT Configurations
    JWT_SECRET_KEY: str = "production-jwt-secret-key-change-me"
    JWT_ALGORITHM: str = "HS256"
    
    # Backpressure Queue Configurations
    QUEUE_MAX_SIZE: int = 100
    QUEUE_OVERFLOW_STRATEGY: str = "drop_oldest"  # options: "drop_oldest", "block", "ignore_and_warn"

    # Transcript Retention Configurations
    # Configures the sliding history buffer window size per session_id (default: 500)
    TRANSCRIPT_BUFFER_SIZE: int = 500

    # ElevenLabs / STT Configurations
    ELEVENLABS_API_KEY: str = ""
    STT_PROVIDER_MODE: str = "development"  # choices: "development", "production"

    # Pyannote / Diarization Configurations
    HUGGINGFACE_TOKEN: str = ""
    DIARIZATION_MODE: str = "development"  # choices: "development", "production"
    # Rolling audio buffer kept per session (seconds)
    DIARIZATION_BUFFER_SECONDS: int = 7
    # Window fed into Pyannote on each inference cycle (seconds, must be <= DIARIZATION_BUFFER_SECONDS)
    DIARIZATION_WINDOW_SECONDS: int = 5
    # How often the inference cycle fires (seconds)
    DIARIZATION_INTERVAL_SECONDS: int = 2

    @model_validator(mode="after")
    def validate_provider_mode(self) -> 'Settings':
        if self.STT_PROVIDER_MODE == "production":
            if not self.ELEVENLABS_API_KEY or self.ELEVENLABS_API_KEY.strip() == "":
                raise ValueError(
                    "ELEVENLABS_API_KEY must be configured when STT_PROVIDER_MODE is set to 'production'."
                )
        if self.DIARIZATION_MODE == "production":
            if not self.HUGGINGFACE_TOKEN or self.HUGGINGFACE_TOKEN.strip() == "":
                raise ValueError(
                    "HUGGINGFACE_TOKEN must be configured when DIARIZATION_MODE is set to 'production'."
                )
        return self

    class Config:
        case_sensitive = True
        env_prefix = "CLINICAL_COPILOT_"
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
