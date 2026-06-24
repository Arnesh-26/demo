from typing import Optional
from app.core.config import settings
from app.utils.logging import logger
from app.utils.jwt_utils import decode_jwt_token

def verify_session_token(
    token: str,
    expected_session_id: Optional[str] = None,
    expected_role: Optional[str] = None
) -> bool:
    """
    Validates the connection token. Supports both legacy AUTH_SECRET_TOKEN verification 
    and production-safe JWT claim verification (checking expiration and role/session matches).
    """
    if not token:
        logger.warning("Authentication failed: Missing token", extra={"event": "auth_failure", "reason": "missing_token"})
        return False
        
    # 1. Legacy auth check (for backward compatibility)
    if token == settings.AUTH_SECRET_TOKEN:
        return True
        
    # 2. JWT auth check
    payload = decode_jwt_token(token)
    if payload is not None:
        # If expected claims are provided, enforce matching logic
        if expected_session_id and payload.get("session_id") != expected_session_id:
            logger.warning(
                "JWT verification failed: Session ID mismatch",
                extra={"expected": expected_session_id, "received": payload.get("session_id")}
            )
            return False
            
        if expected_role and payload.get("role") != expected_role:
            logger.warning(
                "JWT verification failed: Role mismatch",
                extra={"expected": expected_role, "received": payload.get("role")}
            )
            return False
            
        return True
        
    logger.warning("Authentication failed: Invalid token or JWT claims", extra={"event": "auth_failure", "reason": "invalid_token"})
    return False

def verify_participant_role(role: str) -> bool:
    """
    Ensures that the client identifies as a valid consultation participant.
    """
    if role not in settings.ALLOWED_ROLES:
        logger.warning("Authorization failed: Invalid role", extra={"event": "auth_failure", "reason": "invalid_role", "role": role})
        return False
    return True
