from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# US-006: User institutional access
# ---------------------------------------------------------------------------

class UserInstitutionalAccessResponse(BaseModel):
    libkey_library_id: Optional[str] = None
    has_libkey: bool = False


class UserInstitutionalAccessUpdate(BaseModel):
    libkey_library_id: Optional[str] = None


# ---------------------------------------------------------------------------
# US-007: Institution list (LibKey library catalogue)
# ---------------------------------------------------------------------------

class InstitutionItem(BaseModel):
    library_id: str
    name: str
    homepage_url: Optional[str] = None
    image_url: Optional[str] = None


class InstitutionListResponse(BaseModel):
    institutions: List[InstitutionItem]


# ---------------------------------------------------------------------------
# US-008: Batch access resolution
# ---------------------------------------------------------------------------

class ResolveDoisRequest(BaseModel):
    dois: List[str]


class AccessResult(BaseModel):
    doi: str
    access_status: str  # "open" | "closed" | "unknown"
    pdf_url: Optional[str] = None
    source: Optional[str] = None  # "libkey" | "oa_direct" | "proxy" | "doi" | None


class ResolveAccessResponse(BaseModel):
    results: List[AccessResult]
