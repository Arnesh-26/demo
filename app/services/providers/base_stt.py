from abc import ABC, abstractmethod

class BaseSTTProvider(ABC):
    """
    Abstract Base Class for Real-Time Speech-to-Text Providers.
    Provides standard interface hooks for Step 2 and future steps.
    """
    @abstractmethod
    async def connect(self) -> None:
        """Establishes connection to the provider's streaming service."""
        pass

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Sends an audio chunk to the transcription service."""
        pass

    @abstractmethod
    async def receive(self) -> dict:
        """Receives real-time transcriptions/metadata from the provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleans up resources and closes the connection."""
        pass

    async def send_commit(self) -> None:
        """Sends a commit signal to finalize transcription. Optional for providers."""
        pass

