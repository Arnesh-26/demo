import asyncio
import websockets
import sys
import json
import urllib.request
import time

try:
    import pyaudio
except ImportError:
    print("Error: 'pyaudio' library is required to stream from your microphone.")
    print("Please install it by running: .\\.venv\\Scripts\\pip.exe install pyaudio")
    sys.exit(1)

WS_URL = "ws://127.0.0.1:8000/v1/streaming/audio"
TOKEN = "production-secure-token-change-me"
SESSION_ID = "consultation-xyz-123"

# Audio parameters matching backend constraints
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 800  # 800 samples = 50ms of audio

async def stream_mic(role: str):
    p = pyaudio.PyAudio()
    
    # Open mic stream
    try:
        audio_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Error opening microphone: {e}", file=sys.stderr)
        p.terminate()
        return

    url = f"{WS_URL}?session_id={SESSION_ID}&role={role}&token={TOKEN}"
    print(f"Connecting to WebSocket: {url}")
    print("Recording and streaming. Press CTRL+C to stop...")
    
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully! Start speaking into your microphone...")
            
            # Start streaming
            while True:
                try:
                    # Read raw PCM data from microphone (non-blocking read)
                    data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await ws.send(data)
                except IOError:
                    # Input overflow, skip frame
                    pass
                # Short yield to allow event loop processing
                await asyncio.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nWebSocket error: {e}", file=sys.stderr)
    finally:
        # Clean up audio stream
        try:
            audio_stream.stop_stream()
            audio_stream.close()
        except Exception:
            pass
        p.terminate()

def fetch_transcripts():
    print("\nWaiting 1.5 seconds for final STT worker flushes to settle...")
    time.sleep(1.5)
    print("Retrieving transcripts from backend...")
    try:
        url = f"http://127.0.0.1:8000/v1/transcripts/{SESSION_ID}"
        with urllib.request.urlopen(url) as response:
            transcripts = json.loads(response.read().decode())
            print(f"Fetched {len(transcripts)} transcript events successfully:")
            for event in transcripts:
                role = event['role'].upper()
                speaker = event.get('speaker_id', 'UNKNOWN')
                print(f" - [{role} / {speaker}] [final: {event['is_final']}]: '{event['transcript']}'")

    except Exception as e:
        print(f"Failed to retrieve transcripts: {e}", file=sys.stderr)

if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "doctor"
    if role not in ["doctor", "patient", "attender"]:
        print(f"Error: Invalid role '{role}'. Must be doctor, patient, or attender.")
        sys.exit(1)
        
    try:
        asyncio.run(stream_mic(role))
    except KeyboardInterrupt:
        print("\nStopping microphone stream...")
    finally:
        fetch_transcripts()
