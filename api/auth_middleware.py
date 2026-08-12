from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


DEFAULT_USER_ROLE = "product_lead"
VALID_USER_ROLES = {
    "product_lead",
    "tech_lead",
    "compliance_lead",
    "bank_alliance_lead",
}


def normalize_user_role(role: str | None) -> str:
    role = (role or DEFAULT_USER_ROLE).strip()
    return role if role in VALID_USER_ROLES else DEFAULT_USER_ROLE


def identity(headers) -> dict[str, str]:
    return {
        "role": normalize_user_role(headers.get("X-User-Role")),
        "user_id": headers.get("X-User-Id", "anonymous"),
        "tenant_id": headers.get("X-Tenant-Id", "demo-fintech"),
    }


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        current_user = identity(request.headers)
        request.state.user_role = current_user["role"]
        request.state.user = current_user
        return await call_next(request)
