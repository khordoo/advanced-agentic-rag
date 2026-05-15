from fastapi import Request

from app.services.rag_service import RagService


def get_rag_service(request: Request) -> RagService:
    """Builds the service using the pre-compiled global state."""
    return request.app.state.rag_service
