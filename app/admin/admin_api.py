"""Admin API — user management, stats, audit log."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.admin import require_admin
from app.auth.jwt_user import get_current_user_id
from app.db import make_engine
from app.admin.admin_service import (
    ensure_admin_tables,
    get_admin_stats,
    get_users_paginated,
    get_user_detail,
    grant_pro,
    revoke_to_free,
    grant_credits,
    grant_temporary_pro,
    process_refund,
    extend_billing_cycle,
    get_user_cost_breakdown,
    reset_password,
    disable_user,
    enable_user,
    log_audit,
    get_audit_log,
)
from app.admin.schemas import (
    AdminMeResponse,
    AdminStatsResponse,
    AdminUserListResponse,
    AdminUserDetailResponse,
    SubscriptionActionRequest,
    AdminRoleRequest,
    AuditLogResponse,
    CostBreakdownResponse,
    MessageResponse,
    BulkActionRequest,
    BulkActionResult,
    BulkActionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


# ---------------------------------------------------------------------------
# GET /v1/admin/me — check if current user is admin (no admin required)
# ---------------------------------------------------------------------------

@router.get("/me", response_model=AdminMeResponse)
def admin_me(
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    if user_id is None:
        return AdminMeResponse(is_admin=False)

    with engine.connect() as conn:
        ensure_admin_tables(conn)
        row = conn.execute(
            text("SELECT is_admin FROM public.users WHERE user_id = :uid"),
            {"uid": user_id},
        ).mappings().first()

    is_admin = bool(row["is_admin"]) if row else False
    return AdminMeResponse(is_admin=is_admin)


# ---------------------------------------------------------------------------
# GET /v1/admin/stats
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=AdminStatsResponse)
def stats(
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        data = get_admin_stats(conn)
    return AdminStatsResponse(**data)


# ---------------------------------------------------------------------------
# GET /v1/admin/users
# ---------------------------------------------------------------------------

@router.get("/users", response_model=AdminUserListResponse)
def list_users(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    plan: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_dir: str = Query("desc"),
    status: Optional[str] = Query(None),
    min_credits: Optional[float] = Query(None),
    max_credits: Optional[float] = Query(None),
    created_after: Optional[str] = Query(None),
    created_before: Optional[str] = Query(None),
    active_since: Optional[str] = Query(None),
    is_admin: Optional[bool] = Query(None),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        data = get_users_paginated(
            conn, search, page, per_page, plan, sort_by, sort_dir,
            status=status,
            min_credits=min_credits,
            max_credits=max_credits,
            created_after=created_after,
            created_before=created_before,
            active_since=active_since,
            is_admin=is_admin,
        )
    return AdminUserListResponse(**data)


# ---------------------------------------------------------------------------
# POST /v1/admin/users/bulk-action
# ---------------------------------------------------------------------------

@router.post("/users/bulk-action", response_model=BulkActionResponse)
def bulk_action(
    req: BulkActionRequest,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    valid_actions = {"grant_pro", "revoke_to_free", "grant_credits", "disable", "enable"}
    if req.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}. Must be one of {sorted(valid_actions)}")

    if req.action == "grant_credits" and (not req.credits_amount or req.credits_amount <= 0):
        raise HTTPException(status_code=400, detail="credits_amount must be positive for grant_credits action")

    results: list[BulkActionResult] = []

    for uid_str in req.user_ids:
        try:
            uid = UUID(uid_str)
        except ValueError:
            results.append(BulkActionResult(user_id=uid_str, success=False, message=f"Invalid UUID: {uid_str}"))
            continue

        try:
            with engine.begin() as conn:
                ensure_admin_tables(conn)

                if req.action == "grant_pro":
                    grant_pro(conn, admin_id, uid)
                    msg = "Pro plan granted"

                elif req.action == "revoke_to_free":
                    revoke_to_free(conn, admin_id, uid)
                    msg = "Revoked to free plan"

                elif req.action == "grant_credits":
                    grant_credits(conn, admin_id, uid, req.credits_amount or 0.0)
                    msg = f"Granted {req.credits_amount} credits"

                elif req.action == "disable":
                    disable_user(uid)
                    log_audit(conn, admin_id, "disable_user", uid)
                    msg = "User disabled"

                elif req.action == "enable":
                    enable_user(uid)
                    log_audit(conn, admin_id, "enable_user", uid)
                    msg = "User enabled"

                else:
                    msg = "No-op"

            results.append(BulkActionResult(user_id=uid_str, success=True, message=msg))

        except Exception as e:
            logger.warning("Bulk action %s failed for user %s: %s", req.action, uid_str, str(e))
            results.append(BulkActionResult(user_id=uid_str, success=False, message=str(e)))

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    return BulkActionResponse(results=results, total=len(results), succeeded=succeeded, failed=failed)


# ---------------------------------------------------------------------------
# GET /v1/admin/users/{user_id}
# ---------------------------------------------------------------------------

@router.get("/users/{user_id}", response_model=AdminUserDetailResponse)
def user_detail(
    user_id: UUID,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        data = get_user_detail(conn, user_id)

    if not data:
        raise HTTPException(status_code=404, detail="User not found")

    return AdminUserDetailResponse(**data)


# ---------------------------------------------------------------------------
# POST /v1/admin/users/{user_id}/subscription
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/subscription", response_model=MessageResponse)
def manage_subscription(
    user_id: UUID,
    req: SubscriptionActionRequest,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)

        if req.action == "grant_pro":
            grant_pro(conn, admin_id, user_id)
            conn.commit()
            return MessageResponse(message="Pro plan granted")

        elif req.action == "revoke_to_free":
            revoke_to_free(conn, admin_id, user_id)
            conn.commit()
            return MessageResponse(message="Revoked to free plan")

        elif req.action == "grant_credits":
            if not req.credits_amount or req.credits_amount <= 0:
                raise HTTPException(status_code=400, detail="credits_amount must be positive")
            grant_credits(conn, admin_id, user_id, req.credits_amount)
            conn.commit()
            return MessageResponse(message=f"Granted {req.credits_amount} credits")

        elif req.action == "grant_temporary_pro":
            if not req.expires_at:
                raise HTTPException(status_code=400, detail="expires_at is required")
            grant_temporary_pro(conn, admin_id, user_id, req.expires_at)
            conn.commit()
            return MessageResponse(message=f"Temporary Pro granted until {req.expires_at}")

        elif req.action == "process_refund":
            if not req.credits_amount or req.credits_amount <= 0:
                raise HTTPException(status_code=400, detail="credits_amount must be positive")
            process_refund(conn, admin_id, user_id, req.credits_amount, req.refund_reason or "No reason provided")
            conn.commit()
            return MessageResponse(message=f"Refunded {req.credits_amount} credits")

        elif req.action == "extend_billing":
            if not req.new_period_end:
                raise HTTPException(status_code=400, detail="new_period_end is required")
            extend_billing_cycle(conn, admin_id, user_id, req.new_period_end)
            conn.commit()
            return MessageResponse(message=f"Billing extended to {req.new_period_end}")

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


# ---------------------------------------------------------------------------
# POST /v1/admin/users/{user_id}/reset-password
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/reset-password", response_model=MessageResponse)
def user_reset_password(
    user_id: UUID,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        log_audit(conn, admin_id, "reset_password", user_id)
        conn.commit()

    reset_password(user_id)
    return MessageResponse(message="Password reset link generated")


# ---------------------------------------------------------------------------
# POST /v1/admin/users/{user_id}/admin-role
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/admin-role", response_model=MessageResponse)
def set_admin_role(
    user_id: UUID,
    req: AdminRoleRequest,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    if not req.is_admin and user_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin role")

    with engine.connect() as conn:
        ensure_admin_tables(conn)

        # Ensure public.users row exists
        conn.execute(
            text("""
                INSERT INTO public.users (user_id, is_admin)
                VALUES (:uid, :is_admin)
                ON CONFLICT (user_id) DO UPDATE SET is_admin = :is_admin, updated_at = now()
            """),
            {"uid": user_id, "is_admin": req.is_admin},
        )
        action = "grant_admin" if req.is_admin else "revoke_admin"
        log_audit(conn, admin_id, action, user_id)
        conn.commit()

    msg = "Admin role granted" if req.is_admin else "Admin role revoked"
    return MessageResponse(message=msg)


# ---------------------------------------------------------------------------
# POST /v1/admin/users/{user_id}/disable
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/disable", response_model=MessageResponse)
def user_disable(
    user_id: UUID,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    # Cannot disable admins
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        row = conn.execute(
            text("SELECT is_admin FROM public.users WHERE user_id = :uid"),
            {"uid": user_id},
        ).mappings().first()

        if row and row["is_admin"]:
            raise HTTPException(status_code=400, detail="Cannot disable an admin user")

        disable_user(user_id)
        log_audit(conn, admin_id, "disable_user", user_id)
        conn.commit()

    return MessageResponse(message="User disabled")


# ---------------------------------------------------------------------------
# POST /v1/admin/users/{user_id}/enable
# ---------------------------------------------------------------------------

@router.post("/users/{user_id}/enable", response_model=MessageResponse)
def user_enable(
    user_id: UUID,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        enable_user(user_id)
        log_audit(conn, admin_id, "enable_user", user_id)
        conn.commit()

    return MessageResponse(message="User enabled")


# ---------------------------------------------------------------------------
# GET /v1/admin/users/{user_id}/cost-breakdown
# ---------------------------------------------------------------------------

@router.get("/users/{user_id}/cost-breakdown", response_model=CostBreakdownResponse)
def user_cost_breakdown(
    user_id: UUID,
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        data = get_user_cost_breakdown(conn, user_id)
    return CostBreakdownResponse(**data)


# ---------------------------------------------------------------------------
# GET /v1/admin/audit-log
# ---------------------------------------------------------------------------

@router.get("/audit-log", response_model=AuditLogResponse)
def audit_log(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    with engine.connect() as conn:
        ensure_admin_tables(conn)
        data = get_audit_log(conn, page, per_page)
    return AuditLogResponse(**data)
