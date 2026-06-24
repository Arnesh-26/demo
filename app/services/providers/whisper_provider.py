from app.services.providers.base_stt import BaseSTTProvider
from app.utils.logging import logger

class WhisperProvider(BaseSTTProvider):
    """
    STT Provider implementation stub for OpenAI Whisper / custom Whisper endpoints.
    Allows for future engine switching.
    """
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    async def connect(self) -> None:
        logger.info("Whisper Provider: Initializing streaming connection...")
        pass

    async def send_audio(self, chunk: bytes) -> None:
        logger.debug(f"Whisper Provider: Sending {len(chunk)} bytes to Whisper endpoint")
        pass

    async def receive(self) -> dict:
        return {"provider": "whisper", "transcript": "", "words": []}

    async def disconnect(self) -> None:
        logger.info("Whisper Provider: Terminating session.")
        pass
