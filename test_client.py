import asyncio
import websockets
import json
import base64
import math
import sys
from typing import Dict, Any

from app.utils.jwt_utils import generate_jwt_token

# Connection details
WS_URL = "ws://127.0.0.1:8000/v1/streaming/audio"
TOKEN = "production-secure-token-change-me"
SESSION_ID = "consultation-xyz-123"

# PCM Sine wave generator parameters
SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8

def generate_pcm_sine_wave(duration_ms: int, frequency: float = 440.0) -> bytes:
    """Generates a raw 16-bit Mono 16kHz PCM sine wave chunk."""
    num_samples = int((SAMPLE_RATE * duration_ms) / 1000)
    audio_data = bytearray()
    
    for i in range(num_samples):
        # Time step
        t = i / SAMPLE_RATE
        # Sine amplitude -32768 to 32767 for 16-bit PCM
        sample = int(32767.0 * math.sin(2.0 * math.pi * frequency * t))
        # Pack to 16-bit signed integer (little endian)
        audio_data.extend(sample.to_bytes(2, byteorder='little', signed=True))
        
    return bytes(audio_data)

async def stream_audio_for_role(role: str, mode: str = "binary", token: str = None):
    """
    Connects to the FastAPI WebSocket server and streams audio in real-time.
    Supports either 'binary' mode or 'json' mode.
    """
    auth_token = token or TOKEN
    url = f"{WS_URL}?session_id={SESSION_ID}&role={role}&token={auth_token}"
    print(f"[{role.upper()}] Connecting to WebSocket URL: {url}")
    
    try:
        async with websockets.connect(url) as ws:
            print(f"[{role.upper()}] Connected successfully (Mode: {mode})")
            
            # Send 50 frames (approx 1 to 2 seconds of audio)
            # Frame size: 20ms or 40ms or 50ms
            frame_durations = [20, 30, 40, 50]
            
            for seq in range(50):
                # Cycle frame size to test server-side dynamic validation
                dur_ms = frame_durations[seq % len(frame_durations)]
                pcm_data = generate_pcm_sine_wave(dur_ms, frequency=300.0 + (seq * 10))
                
                # Verify local calculations
                expected_bytes = int((SAMPLE_RATE * dur_ms / 1000) * BYTES_PER_SAMPLE)
                assert len(pcm_data) == expected_bytes
                
                if mode == "binary":
                    # Stream raw binary frame
                    await ws.send(pcm_data)
                else:
                    # Stream base64 encoded JSON frame
                    payload = {
                        "seq": seq,
                        "audio": base64.b64encode(pcm_data).decode('utf-8')
                    }
                    await ws.send(json.dumps(payload))
                
                # Check for any warning/error feedback messages from server (non-blocking read)
                try:
                    # Wait a tiny amount or do a non-blocking check
                    res = await asyncio.wait_for(ws.recv(), timeout=0.001)
                    print(f"[{role.upper()}] Server notification: {res}")
                except asyncio.TimeoutError:
                    pass
                
                # Sleep in real-time alignment with the frame duration
                await asyncio.sleep(dur_ms / 1000.0)
                
            print(f"[{role.upper()}] Finished streaming 50 frames. Closing connection.")
            
    except websockets.exceptions.ConnectionClosedOK:
        print(f"[{role.upper()}] Connection closed normally.")
    except Exception as e:
        print(f"[{role.upper()}] Error: {e}", file=sys.stderr)

async def main():
    print("Starting client audio streaming simulations...")
    # Generate dynamic JWT tokens for Doctor and Attender
    doctor_jwt = generate_jwt_token(SESSION_ID, "doctor")
    attender_jwt = generate_jwt_token(SESSION_ID, "attender")
    
    # Stream simultaneously for:
    # 1. Doctor (using new JWT token)
    # 2. Patient (using legacy Token to prove backward compatibility)
    # 3. Attender (using new JWT token)
    await asyncio.gather(
        stream_audio_for_role("doctor", mode="binary", token=doctor_jwt),
        stream_audio_for_role("patient", mode="json", token=TOKEN),
        stream_audio_for_role("attender", mode="binary", token=attender_jwt)
    )
    print("Simulation completed.")
    
    # Wait for the reconciliation loop and worker flushes to settle
    print("Waiting 1.5 seconds for final worker reconciliation and flush...")
    await asyncio.sleep(1.5)
    
    # Retrieve and print the generated transcript events
    print("\nRetrieving buffered transcript events from backend...")
    import urllib.request
    try:
        url = f"http://127.0.0.1:8000/v1/transcripts/{SESSION_ID}"
        with urllib.request.urlopen(url) as response:
            transcripts = json.loads(response.read().decode())
            print(f"Fetched {len(transcripts)} transcript events successfully:")
            for event in transcripts:
                print(
                    f" - [{event['role'].upper()}] (seq: {event['sequence_number']}) "
                    f"[final: {event['is_final']}]: '{event['transcript']}'"
                )
    except Exception as e:
        print(f"Failed to retrieve transcripts: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
