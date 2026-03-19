from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
import json


# ─── Message ──────────────────────────────────────────────────────────────────

class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:            UUID
    session_id:    UUID
    role:          str
    content:       str
    sources:       Optional[List[Any]] = None   # deserialized from sources_json
    metadata:      Optional[dict]      = None   # deserialized from metadata_json
    turn_index:    int
    created_at:    datetime

    @classmethod
    def from_orm_with_json(cls, obj):
        """Deserialize JSON fields before validation."""
        sources  = None
        metadata = None
        try:
            if obj.sources_json:
                sources = json.loads(obj.sources_json)
        except Exception:
            pass
        try:
            if obj.metadata_json:
                metadata = json.loads(obj.metadata_json)
        except Exception:
            pass

        return cls(
            id=obj.id,
            session_id=obj.session_id,
            role=obj.role,
            content=obj.content,
            sources=sources,
            metadata=metadata,
            turn_index=obj.turn_index,
            created_at=obj.created_at,
        )


# ─── Session ──────────────────────────────────────────────────────────────────

class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         UUID
    user_id:    UUID
    title:      str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None   # populated in list views


class ChatSessionDetailResponse(ChatSessionResponse):
    """Session + all messages — returned when opening a session."""
    messages: List[ChatMessageResponse] = []


class ChatSessionListResponse(BaseModel):
    items: List[ChatSessionResponse]
    total: int


# ─── Requests ────────────────────────────────────────────────────────────────

class RenameChatSessionRequest(BaseModel):
    title: str