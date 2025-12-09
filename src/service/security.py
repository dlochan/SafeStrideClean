# src/service/security.py
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


bearer_scheme = HTTPBearer(auto_error=False)


def verify_bearer_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> None:
    """Validate Authorization: Bearer <token> header against env SERVICE_TOKEN.
    Deny if missing env, missing header, or mismatch.
    """
    expected = os.getenv("SERVICE_TOKEN")
    if not expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Service token not configured")
    if creds is None or not creds.scheme or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    if creds.credentials != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    # Return None on success (used as dependency)
    return None
