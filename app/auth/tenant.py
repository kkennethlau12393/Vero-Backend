import os
from uuid import UUID

def get_tenant_id() -> UUID:
    # Dev stub: single-tenant ID from env
    val = os.getenv("DEV_TENANT_ID")
    if not val:
        raise RuntimeError("DEV_TENANT_ID is not set in .env")
    return UUID(val)
