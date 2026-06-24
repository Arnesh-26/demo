# Clinical Consultation Copilot - Step 1: Audio Streaming Layer

This repository contains the backend implementation of the **Audio Streaming Layer** (Step 1 of Milestone 1), built using **FastAPI** and designed for high-availability, low-latency clinical session recording. It accepts concurrent audio streams from multiple actors (Doctor, Patient, and Attender) during a consultation, validates the raw PCM data on-the-fly, buffers the incoming frames, and packages them into chunks ready for downstream ingestion (e.g. ElevenLabs Scribe V2 in Step 2).

---

## 1. Protocol Evaluation: WebRTC vs. WebSockets

For real-time streaming audio ingestion, the primary protocol choices are **WebRTC** and **WebSockets**. Below is an engineering evaluation of their roles in the system.

### WebRTC (Real-Time Communication)
WebRTC is designed for low-latency media transport directly over UDP (using RTP/SRTP). 

*   **Why WebRTC is Chosen (for Client Ingestion):**
    *   **Ultra-Low Latency:** Works over UDP, bypassing TCP retransmission delays (head-of-line blocking). Latency is usually sub-100ms.
    *   **Native Jitter Buffering:** WebRTC client stacks (in browsers and mobile OSs) automatically handle packet reordering, echo cancellation, and network jitter adjustments.
    *   **Bandwidth Adaptation:** Dynamically scales audio quality down under weak network conditions to avoid drops.
*   **Disadvantages:**
    *   **Server CPU Overhead:** Terminating WebRTC in standard Python servers (e.g., using `aiortc`) is single-threaded, CPU-bound, and does not scale well.
    *   **Complexity:** Requires ICE negotiation (STUN/TURN servers) to bypass firewalls and NATs.

### WebSocket (Bidirectional TCP Stream)
WebSockets provide a persistent, full-duplex communication channel over a single TCP connection.

*   **Why WebSocket is Used (for Internal & Edge Gateway Ingestion):**
    *   **Implementation Simplicity:** Native support in standard application gateways (FastAPI/ASGI) with standard HTTP/S port binding (80/443), making firewall traversal effortless.
    *   **Guaranteed Delivery:** Built on TCP, ensuring that no audio frame is lost due to packet drops, making downstream transcription processing deterministic.
    *   **Stateless Scaling:** Standard HTTP load balancers can distribute WebSocket connections easily (with sticky routing or session sharing).
*   **Disadvantages:**
    *   **Head-of-Line Blocking:** If a packet is lost, TCP pauses the stream until the packet is retransmitted. This can cause temporary spikes in streaming latency over weak connections.

| Feature | WebRTC (UDP/RTP) | WebSocket (TCP) |
| :--- | :--- | :--- |
| **Transport Protocol** | UDP (primarily) | TCP |
| **Latency** | Extremely low (<150ms) | Low (150ms - 500ms) |
| **Reliability** | Best effort (packets can drop) | Guaranteed delivery (no drops) |
| **Infrastructure Overhead** | High (Requires STUN/TURN, SFU/Gateway) | Low (Standard HTTPS Load Balancer) |
| **Usage in Copilot** | **Edge Client Ingestion** (App to Media Gateway) | **Gateway to Audio Processing Core** (FastAPI) |

---

## 2. Technical Architecture Diagram

```
 +------------------+      +-------------------+      +---------------------+
 |    Doctor Mic    |      |    Patient Mic    |      |     Attender Mic    |
 | (Client App Web) |      | (Client App Web)  |      |  (Client App Web)   |
 +--------+---------+      +---------+---------+      +----------+----------+
          |                          |                           |
          | WebRTC Media Stream      | WebRTC Media Stream       | WebRTC Media Stream
          v                          v                           v
 +--------------------------------------------------------------------------+
 |                    WebRTC Media Gateway (Edge SFU)                       |
 |  (Terminates WebRTC UDP packets, converts to raw PCM 16-bit Mono 16kHz)  |
 +-------------------------------------+------------------------------------+
                                       |
                                       | WebSocket Connections (TCP, TLS)
                                       v
 +--------------------------------------------------------------------------+
 |                      FastAPI Ingestion Gateway                           |
 |      - Handshake Authentication (Token validation)                       |
 |      - Active Session Registry (SessionManager)                          |
 |      - Endpoint: /v1/streaming/audio                                     |
 +-------------------------------------+------------------------------------+
                                       |
                                       | Decoded Byte Streams
                                       v
 +--------------------------------------------------------------------------+
 |                       Audio Streaming Service                            |
 |      - Dynamic Audio Frame Validation (Checks bit depth, rate, length)   |
 |      - Buffer Manager (Accumulates 20-50ms frames into 1.0s chunks)      |
 +-------------------------------------+------------------------------------+
                                       |
                                       | Buffered PCM Chunks (1.0s / 32KB)
                                       v
 +--------------------------------------------------------------------------+
 |                       Audio Frame Buffer Queue                           |
 |       - Exposes FIFO asyncio.Queue per role                              |
 |       - Ready to be read by downstream Scribe V2 transcription service   |
 +--------------------------------------------------------------------------+
```

---

## 3. Directory Structure

This project follows a clean, production-grade modular design pattern:

```
c:/Intern/transcriptmodel/
├── requirements.txt                  # Python dependencies
├── test_client.py                    # Multi-role simulation client
├── app/
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # FastAPI application entrypoint & health checks
│   ├── api/
│   │   ├── __init__.py
│   │   └── websocket.py              # WebSocket endpoint handler
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Systems configurations (PCM parameters, constraints)
│   │   └── security.py               # Token authentication and role verification
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_buffer.py           # Audio buffer manager & PCM frame validator
│   │   └── session.py                # Consultation session and participant registry
│   └── utils/
│       ├── __init__.py
│       └── logging.py                # Custom JSON structured logging
```

---

## 4. Backend Audio Ingestion Contract

To integrate frontend or mobile clients, follow this contract.

### Connection Handshake
*   **Protocol:** Secure WebSocket (`wss://`) or standard WebSocket (`ws://` for local development)
*   **Path:** `/v1/streaming/audio`
*   **Query Parameters:**
    *   `session_id` (string, Required): The UUID or unique identifier of the consultation session.
    *   `role` (string, Required): One of `doctor`, `patient`, or `attender`.
    *   `token` (string, Required): The authorization token corresponding to this consultation session.
*   **Example URL:**
    ```
    ws://127.0.0.1:8000/v1/streaming/audio?session_id=consultation-123&role=doctor&token=production-secure-token-change-me
    ```

### Expected Audio Frame Format
*   **Encoding:** Raw Signed PCM (LPCM)
*   **Sample Rate:** 16,000 Hz (16 kHz)
*   **Bit Depth:** 16-bit
*   **Endianness:** Little-endian (default for standard audio captures)
*   **Channels:** 1 (Mono)
*   **Transmission Mode:** Continuous streaming
*   **Frame Duration:** 20ms to 50ms per packet
*   **Byte Size Constraints:**
    *   Formula: `Sample Rate * Channel Count * (Bits Per Sample / 8) * Frame Duration`
    *   **20ms frame:** `16000 * 1 * 2 * 0.02 = 640 bytes` (Minimum)
    *   **50ms frame:** `16000 * 1 * 2 * 0.05 = 1600 bytes` (Maximum)
    *   *Validation rule:* Any packet not aligning with a 2-byte boundary or outside `[640, 1600]` bytes will trigger a validation warning response.

### Ingestion Message Schema

#### Option A: Binary Frames (Recommended for Low Overhead)
Clients stream raw bytes of size `[640, 1600]` over the socket. No headers or wrapper needed.

#### Option B: Text Frames (JSON wrapper)
Useful if the client needs to convey packet sequence metadata to help monitor network packet loss.
```json
{
  "seq": 45,
  "audio": "base64_encoded_pcm_bytes..."
}
```

---

## 5. Audio Frame Buffer Strategy

### Why Buffering is Needed
Clients capture and stream audio in tiny slices (20–50ms) to ensure minimal local recording latency and avoid network queue delays. However, sending such small buffers to Speech-to-Text APIs (like ElevenLabs Scribe V2) would lead to massive API transaction overhead and network jitter issues. The buffer aggregates these micro-frames into stable **1.0 second chunks** (32,000 bytes) before passing them to the transcription engine, establishing a sweet spot between latency and throughput.

### Memory Considerations
*   **Per-stream Buffer Limit:** A hard memory cap is set at **10 MB** per connection. If a network outage or backlog exceeds this, the buffer clears to prevent RAM exhaustion.
*   **Data Structure:** Bytes are appended to a standard Python `bytearray` and sliced when the target size is met, ensuring $O(1)$ memory copies.

### Drop and Reconnect Policies
*   **Dropped Packets:** The system detects dropped sequence numbers for JSON streams, logging telemetry on connection reliability.
*   **Reconnection Handling:** When a client reconnects (e.g., due to cellular handoff), the session manager identifies the session and role, binds the new socket, and **reuses the existing buffer**, preventing data loss.

---

## 6. End-to-End Latency Analysis

E2E Ingestion Latency measures the time elapsed from the physical microphone vibration to the frame being buffered and ready in memory.

### Latency Pipeline
1.  **Microphone Capture:** The local device buffers audio samples until the frame size (20-50ms) is reached.
2.  **Frame Creation:** The client application packages the raw PCM data (binary or base64 JSON).
3.  **Network Transfer:** The data travels over the cellular/broadband connection to the cloud endpoint.
4.  **Backend Buffer:** The server receives, parses, validates, and aggregates the frame.

### Production Latency Cases

```
  BEST CASE (Favorable Fiber Network)
  [Microphone Capture: 20ms] -> [Frame Pack: 1ms] -> [Network: 10ms] -> [Ingest: 1ms]
  Total: 32ms

  TYPICAL PRODUCTION (Standard 4G/5G Network)
  [Microphone Capture: 30ms] -> [Frame Pack: 2ms] -> [Network: 45ms] -> [Ingest: 2ms]
  Total: 79ms

  WORST CASE (Poor Cellular Signal with TCP Jitter)
  [Microphone Capture: 50ms] -> [Frame Pack: 10ms] -> [Network: 250ms] -> [Ingest: 15ms]
  Total: 325ms
```

---

## 7. Global Production Scalability

To scale this service from 100 to 10,000+ concurrent consultations globally, we recommend:

1.  **100 Users (Single Server Instance):**
    *   FastAPI running on standard hardware (e.g., AWS EC2 `t3.medium`) can easily handle 300 active connections (100 sessions $\times$ 3 roles). Since Python is single-threaded, uvicorn should be run with 2–4 workers.
2.  **1,000 Users (Horizontal Auto-scaling):**
    *   Deploy nodes behind an Application Load Balancer (ALB). Enable WebSockets with **Sticky Sessions** (session affinity) so that all 3 streams of a single consultation route to the same instance (necessary for local buffering), OR migrate buffer state to a distributed cache like **Redis**.
3.  **10,000 Users (Global Edge Orchestration):**
    *   Deploy regional clusters (e.g., US-East, US-West, EU-Central, AP-South) using a geo-DNS router (like Route 53 Geo Routing) to direct clients to the nearest region, minimizing TCP handshakes and network latency.

---

## 8. Production Recommendations & Guardrails

*   **Logging:** Output only structured JSON logs (no plain text). Profile ingestion speed and frame metadata.
*   **Monitoring:** Track Prometheus metrics:
    *   `active_connections_count`
    *   `dropped_frames_total`
    *   `buffer_latency_ms`
*   **Security & Authentication:** Encrypt all audio traffic over TLS (`wss://`). Authenticate the connection handshake using short-lived tokens.
*   **Rate Limiting:** Implement limits on connection attempts per IP/API token using Redis Rate Limiter to prevent brute force attacks.
*   **Health Checks:** Load balancers should monitor the `/health` endpoint to route traffic only to active, responsive nodes.
