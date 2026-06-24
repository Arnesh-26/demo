import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from app.core.config import settings
from app.core.security import verify_session_token, verify_participant_role
from app.services.session import session_manager
from app.services.audio_buffer import InvalidAudioFrameError
from app.utils.logging import logger

router = APIRouter()

@router.websocket("/streaming/audio")
async def audio_streaming_endpoint(
    websocket: WebSocket,
    session_id: str = Query(..., description="Unique consultation session ID"),
    role: str = Query(..., description="Role of the participant: doctor, patient, or attender"),
    token: str = Query(..., description="Authentication token for the session")
):
    # 1. Connection Validation and Handshake Authentication
    if not verify_session_token(token, expected_session_id=session_id, expected_role=role):
        logger.warning(
            "WebSocket rejected: Invalid authentication token or JWT claims",
            extra={"session_id": session_id, "role": role}
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid authentication token")
        return

    if not verify_participant_role(role):
        logger.warning(
            "WebSocket rejected: Unauthorized participant role",
            extra={"session_id": session_id, "role": role}
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid participant role")
        return

    # Accept the connection
    await websocket.accept()
    
    # 2. Register participant stream with Session Manager
    consultation = await session_manager.get_or_create_session(session_id)
    stream = consultation.register_stream(role, websocket)
    
    logger.info(
        "WebSocket connection established",
        extra={"session_id": session_id, "role": role, "event": "connection_established"}
    )

    try:
        # Ingestion Loop
        while True:
            # We support both Binary frames (raw PCM) and Text frames (JSON wrapper with metadata)
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
                
            if "bytes" in message:
                # Optimized binary streaming (Zero JSON overhead)
                audio_payload = message["bytes"]
                try:
                    await stream.buffer.append_frame(audio_payload)
                except InvalidAudioFrameError as e:
                    logger.warning(
                        "Frame validation error on binary message",
                        extra={"session_id": session_id, "role": role, "error": str(e)}
                    )
                    # For production, we can choose to drop the frame or notify the client
                    # We send an error frame warning to the client but keep the socket alive
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif "text" in message:
                # Structured JSON streaming
                try:
                    data = json.loads(message["text"])
                    
                    # Validate JSON schema
                    if "audio" not in data:
                        raise ValueError("Missing 'audio' field in JSON payload")
                        
                    # Extract sequence number if sent by client to monitor UDP-like reliability
                    seq = data.get("seq")
                    
                    # Decode base64 audio payload
                    audio_payload = base64.b64decode(data["audio"])
                    
                    await stream.buffer.append_frame(audio_payload, seq=seq)
                except Exception as e:
                    logger.warning(
                        "Frame processing error on JSON message",
                        extra={"session_id": session_id, "role": role, "error": str(e)}
                    )
                    await websocket.send_json({"type": "error", "message": f"Invalid JSON or Payload: {str(e)}"})

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected normally",
            extra={"session_id": session_id, "role": role, "event": "disconnect"}
        )
    except Exception as e:
        logger.error(
            "WebSocket error occurred during streaming",
            exc_info=True,
            extra={"session_id": session_id, "role": role, "event": "error"}
        )
    finally:
        # Clean up session
        # Flush any remaining bytes from buffer
        remaining_chunk = stream.buffer.flush()
        if remaining_chunk:
            logger.info(
                "Flushing trailing audio bytes from session close",
                extra={"session_id": session_id, "role": role, "flushed_bytes": len(remaining_chunk)}
            )
            # Option B: Enqueue remaining chunk into ready queue for downstream STT processing
            try:
                stream.buffer.ready_chunks_queue.put_nowait(remaining_chunk)
            except Exception as e:
                logger.error(
                    "Failed to enqueue trailing flush chunk",
                    extra={"session_id": session_id, "role": role, "error": str(e)}
                )
            
        # Send sentinel value (None) to signal end of stream
        try:
            stream.buffer.ready_chunks_queue.put_nowait(None)
            
            # Signal end of stream to diarization worker
            from app.services.audio_tap import audio_tap
            audio_tap.feed_sentinel(session_id)
        except Exception as e:
            logger.error(
                "Failed to enqueue end-of-stream sentinel",
                extra={"session_id": session_id, "role": role, "error": str(e)}
            )
            
        await session_manager.unregister_stream(session_id, role)

