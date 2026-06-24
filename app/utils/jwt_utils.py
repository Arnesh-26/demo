# pyrefly: ignore [missing-import]
import jwt
import time
from typing import Optional, Dict, Any
from app.core.config import settings
from app.utils.logging import logger

def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT token using the configured secret and algorithm.
    Returns the decoded dictionary payload if valid, otherwise returns None.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT verification failed: Expired signature", extra={"event": "jwt_error", "reason": "expired"})
    except jwt.InvalidTokenError as e:
        logger.warning(f"JWT verification failed: Invalid token ({str(e)})", extra={"event": "jwt_error", "reason": "invalid_token"})
    except Exception as e:
        logger.error(f"JWT verification failed: Unexpected error ({str(e)})", exc_info=True, extra={"event": "jwt_error", "reason": "unexpected"})
    return None

def generate_jwt_token(session_id: str, role: str, expires_in_sec: int = 3600) -> str:
    """
    Generates a valid JWT token for testing and validation purposes.
    """
    payload = {
        "session_id": session_id,
        "role": role,
        "exp": int(time.time()) + expires_in_sec,
        "iat": int(time.time())
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
