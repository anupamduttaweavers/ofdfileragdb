"""
app/routes/admin.py
────────────────────
Admin management endpoints: login, user CRUD, table listing and selection.

Auth:
  - POST /admin/login — no auth required
  - All other endpoints — JWT Bearer required
  - User CRUD — superadmin role required
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header

from app.config import Settings, get_settings
from app.core.admin_auth import authenticate, create_jwt, verify_jwt
from app.exceptions import AuthenticationError, ForbiddenError, ResourceNotFoundError
from app.models.requests import AdminCreateUserRequest, AdminLoginRequest, TableSelectionRequest
from app.models.responses import (
    AdminLoginResponse,
    AdminUserListResponse,
    AdminUserResponse,
    TableConfigItem,
    TableConfigListResponse,
    TableSelectionResponse,
)

log = logging.getLogger("app.routes.admin")
router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


# ── Auth dependency ──────────────────────────────────────────────

def _extract_jwt_payload(
    authorization: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Missing or invalid Authorization header. Expected: Bearer <jwt>")
    token = authorization[7:]
    payload = verify_jwt(token, settings)
    if payload is None:
        raise AuthenticationError("Invalid or expired JWT token.")
    return payload


def _require_superadmin(payload: dict = Depends(_extract_jwt_payload)) -> dict:
    if payload.get("role") != "superadmin":
        raise ForbiddenError("This action requires superadmin privileges.")
    return payload


# ── Login ────────────────────────────────────────────────────────

@router.post("/login", response_model=AdminLoginResponse, summary="Authenticate admin and receive JWT")
async def admin_login(body: AdminLoginRequest, settings: Settings = Depends(get_settings)):
    user = authenticate(body.username, body.password)
    if user is None:
        raise AuthenticationError("Invalid username or password.")

    token = create_jwt(user["username"], user["role"], settings)
    return AdminLoginResponse(
        token=token,
        username=user["username"],
        role=user["role"],
        expires_in_hours=settings.jwt_expiry_hours,
    )


# ── Current user ─────────────────────────────────────────────────

@router.get("/me", response_model=AdminUserResponse, summary="Get current admin info")
async def admin_me(payload: dict = Depends(_extract_jwt_payload)):
    from app.core.config_db import get_admin_user
    user = get_admin_user(payload["sub"])
    if user:
        return AdminUserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            is_active=bool(user["is_active"]),
            created_at=user.get("created_at"),
        )
    return AdminUserResponse(
        id=0,
        username=payload["sub"],
        role=payload.get("role", "superadmin"),
        is_active=True,
    )


# ── User CRUD (superadmin only) ──────────────────────────────────

@router.get("/users", response_model=AdminUserListResponse, summary="List admin users")
async def list_users(_: dict = Depends(_require_superadmin)):
    from app.core.config_db import list_admin_users
    users = list_admin_users()
    items = [
        AdminUserResponse(
            id=u["id"],
            username=u["username"],
            role=u["role"],
            is_active=bool(u["is_active"]),
            created_at=u.get("created_at"),
        )
        for u in users
    ]
    return AdminUserListResponse(users=items, total=len(items))


@router.post("/users", response_model=AdminUserResponse, summary="Create admin user")
async def create_user(body: AdminCreateUserRequest, _: dict = Depends(_require_superadmin)):
    from app.core.config_db import create_admin_user, get_admin_user
    existing = get_admin_user(body.username)
    if existing:
        raise ForbiddenError(f"User '{body.username}' already exists.")

    user = create_admin_user(body.username, body.password, body.role)
    return AdminUserResponse(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user.get("created_at"),
    )


@router.delete("/users/{user_id}", summary="Deactivate admin user")
async def deactivate_user(user_id: int, _: dict = Depends(_require_superadmin)):
    from app.core.config_db import deactivate_admin_user
    if not deactivate_admin_user(user_id):
        raise ResourceNotFoundError("admin user", str(user_id))
    return {"status": "deactivated", "user_id": user_id}


# ── Table listing and selection ──────────────────────────────────

@router.get(
    "/databases/{name}/tables",
    response_model=TableConfigListResponse,
    summary="List tables for a database with selection state",
)
async def list_tables_for_db(name: str, _: dict = Depends(_extract_jwt_payload)):
    from app.core.config_db import get_table_configs_for_db, connection_exists
    if not connection_exists(name):
        raise ResourceNotFoundError("database", name)

    configs = get_table_configs_for_db(name)
    items = [
        TableConfigItem(
            table_name=c["table_name"],
            label=c.get("label"),
            description=c.get("description"),
            is_selected=c["is_selected"],
            source=c.get("source", "auto"),
            pk_column=c.get("pk_column"),
            text_columns=c.get("text_columns", []),
            metadata_columns=c.get("metadata_columns", []),
            file_columns=c.get("file_columns", []),
            date_column=c.get("date_column"),
        )
        for c in configs
    ]
    selected = sum(1 for i in items if i.is_selected)
    return TableConfigListResponse(database=name, tables=items, total=len(items), selected_count=selected)


@router.put(
    "/databases/{name}/tables",
    response_model=TableSelectionResponse,
    summary="Update table selection (select/deselect individual tables)",
)
async def update_table_selection(name: str, body: TableSelectionRequest, _: dict = Depends(_extract_jwt_payload)):
    from app.core.config_db import toggle_table_selection, connection_exists
    if not connection_exists(name):
        raise ResourceNotFoundError("database", name)

    updated = 0
    for table_name, selected in body.selections.items():
        if toggle_table_selection(name, table_name, selected):
            updated += 1

    return TableSelectionResponse(database=name, updated=updated, message=f"Updated {updated} table(s).")


@router.post(
    "/databases/{name}/tables/select-all",
    response_model=TableSelectionResponse,
    summary="Select all tables for a database",
)
async def select_all_tables(name: str, _: dict = Depends(_extract_jwt_payload)):
    from app.core.config_db import set_all_tables_selection, connection_exists
    if not connection_exists(name):
        raise ResourceNotFoundError("database", name)

    count = set_all_tables_selection(name, True)
    return TableSelectionResponse(database=name, updated=count, message=f"Selected all {count} table(s).")


@router.post(
    "/databases/{name}/tables/deselect-all",
    response_model=TableSelectionResponse,
    summary="Deselect all tables for a database",
)
async def deselect_all_tables(name: str, _: dict = Depends(_extract_jwt_payload)):
    from app.core.config_db import set_all_tables_selection, connection_exists
    if not connection_exists(name):
        raise ResourceNotFoundError("database", name)

    count = set_all_tables_selection(name, False)
    return TableSelectionResponse(database=name, updated=count, message=f"Deselected all {count} table(s).")
