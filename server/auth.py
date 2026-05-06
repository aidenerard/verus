"""
server/auth.py
JWT auth helpers for FastAPI endpoints (optional / non-blocking).
"""

from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)


def _decode_jwt_user_id(token: str) -> Optional[str]:
    """Extract 'sub' (user_id) from JWT without a network call."""
    try:
        import base64, json
        payload_b64 = token.split('.')[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
        return payload.get('sub')
    except Exception:
        return None


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """
    Extract user_id from JWT without blocking.
    Returns None if no token — unauthenticated users can still run analysis.
    """
    if credentials is None:
        return None
    return _decode_jwt_user_id(credentials.credentials)
