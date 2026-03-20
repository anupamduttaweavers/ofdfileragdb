"""
app.core.admin_auth
─────────────────────
Admin authentication module.

Security layers:
  - Passwords hashed with bcrypt (one-way, salt built-in)
  - Sessions via JWT (short-lived, configurable expiry)
  - Dual-source authentication: SQLite DB first, .env super admin as fallback

FastAPI dependency: require_admin() validates JWT Bearer tokens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

from app.config import Settings, get_settings

log = logging.getLogger("app.core.admin_auth")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def create_jwt(username: str, role: str, settings: Optional[Settings] = None) -> str:
    if settings is None:
        settings = get_settings()
    payload = {
        "sub": username,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiry_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_jwt(token: str, settings: Optional[Settings] = None) -> Optional[Dict[str, Any]]:
    if settings is None:
        settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        log.debug("JWT expired.")
        return None
    except jwt.InvalidTokenError as exc:
        log.debug("Invalid JWT: %s", exc)
        return None


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user. Checks SQLite first, then .env super admin as fallback.
    Returns dict with 'username' and 'role' on success, None on failure.
    """
    from app.core import config_db

    if config_db.is_initialized():
        user = config_db.get_admin_user(username)
        if user and verify_password(password, user["password_hash"]):
            return {"username": user["username"], "role": user["role"]}

    settings = get_settings()
    if username == settings.super_admin_username and password == settings.super_admin_password:
        log.info("Authenticated via .env fallback for user '%s'.", username)
        return {"username": username, "role": "superadmin"}

    return None
