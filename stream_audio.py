import asyncio
import websockets
import wave
import sys
import json
import urllib.request
import audioop

WS_URL = "ws://127.0.0.1:8000/v1/streaming/audio"
TOKEN = "production-secure-token-change-me"
SESSION_ID = "consultation-xyz-123"

async def stream_audio(file_path: str, role: str):
    print(f"Opening WAV file: {file_path}", flush=True)
    try:
        wf = wave.open(file_path, 'rb')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr, flush=True)
        return
    except Exception as e:
        print(f"Error opening WAV file: {e}", file=sys.stderr, flush=True)
        return

    # Check WAV properties
    channels = wf.getnchannels()
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    
    print(f"Original WAV Format: Channels={channels}, Rate={sample_rate}Hz, Width={sample_width*8}-bit", flush=True)
    
    # Read raw audio frames
    raw_data = wf.readframes(wf.getnframes())
    wf.close()
    
    # 1. Convert to mono if it is stereo or multi-channel
    if channels > 1:
        print(f"Converting stereo/multi-channel ({channels} channels) to mono...", flush=True)
        raw_data = audioop.tomono(raw_data, sample_width, 0.5, 0.5)
        channels = 1
        
    # 2. Convert sample width to 16-bit PCM (2 bytes)
    if sample_width != 2:
        print(f"Converting sample width from {sample_width} bytes to 2 bytes (16-bit)...", flush=True)
        raw_data = audioop.lin2lin(raw_data, sample_width, 2)
        sample_width = 2
        
    # 3. Resample audio sample rate to 16000Hz
    if sample_rate != 16000:
        print(f"Resampling audio from {sample_rate}Hz to 16000Hz...", flush=True)
        state = None
        raw_data, state = audioop.ratecv(raw_data, sample_width, channels, sample_rate, 16000, state)
        sample_rate = 16000

    url = f"{WS_URL}?session_id={SESSION_ID}&role={role}&token={TOKEN}"
    print(f"Connecting to WebSocket: {url}", flush=True)
    
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully. Streaming live audio in real-time...", flush=True)
            
            # Send in 50ms chunks (1600 bytes at 16kHz mono 16-bit PCM)
            chunk_size = 1600
            total_bytes = len(raw_data)
            
            frames_sent = 0
            offset = 0
            while offset < total_bytes:
                data = raw_data[offset:offset+chunk_size]
                offset += chunk_size
                
                await ws.send(data)
                frames_sent += 1
                
                # Sleep 50ms to match real-time recording speed
                await asyncio.sleep(0.05)
                
                if frames_sent % 20 == 0:
                    print(f"Sent {frames_sent} chunks...", flush=True)

            print(f"Finished streaming {frames_sent} audio chunks. Closing connection.", flush=True)
            
    except Exception as e:
        print(f"WebSocket error: {e}", file=sys.stderr, flush=True)

    # Wait 1.5 seconds for final worker reconciliation and flush
    print("\nWaiting 1.5 seconds for final STT worker flushes to settle...", flush=True)
    await asyncio.sleep(1.5)

    # Retrieve transcripts
    print("Retrieving transcripts from backend...", flush=True)
    try:
        url = f"http://127.0.0.1:8000/v1/transcripts/{SESSION_ID}"
        with urllib.request.urlopen(url) as response:
            transcripts = json.loads(response.read().decode())
            print(f"Fetched {len(transcripts)} transcript events successfully:", flush=True)
            for event in transcripts:
                role = event['role'].upper()
                speaker = event.get('speaker_id', 'UNKNOWN')
                print(f" - [{role} / {speaker}] [final: {event['is_final']}]: '{event['transcript']}'", flush=True)

    except Exception as e:
        print(f"Failed to retrieve transcripts: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stream_audio.py <path_to_wav_file> [role: doctor|patient|attender]", flush=True)
        sys.exit(1)
        
    wav_path = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "doctor"
    
    if role not in ["doctor", "patient", "attender"]:
        print(f"Error: Invalid role '{role}'. Must be doctor, patient, or attender.", flush=True)
        sys.exit(1)
        
    asyncio.run(stream_audio(wav_path, role))
