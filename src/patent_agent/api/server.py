"""
FastAPI server with CopilotKit integration for Patent Agent System.

This module provides the HTTP API layer for the Patent Agent System,
enabling integration with React/Next.js frontend via CopilotKit.
"""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from patent_agent.config import settings

logger = structlog.get_logger()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    debug: bool


class WorkflowRequest(BaseModel):
    """Request model for starting a workflow."""

    workflow_type: str
    input_text: str
    session_id: str | None = None


class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""

    session_id: str
    status: str
    current_step: str | None = None
    message: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("starting_server", port=8000)
    yield
    logger.info("shutting_down_server")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    application = FastAPI(
        title="Patent Agent System API",
        description="CopilotKit-enabled Patent Agent System for Korean patent workflows",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS configuration for Next.js frontend
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js dev server
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(application)

    # Register CopilotKit endpoint
    _register_copilotkit(application)

    return application


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.2.0",
            debug=settings.debug,
        )

    @app.get("/api/workflows")
    async def list_workflows() -> dict[str, Any]:
        """List available workflows."""
        return {
            "workflows": [
                {
                    "id": "spec_writing",
                    "name": "명세서 작성",
                    "description": "발명 정보로부터 특허 명세서 자동 생성",
                    "steps": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
                    "hitl_checkpoints": ["E4", "E6", "E9"],
                },
                {
                    "id": "oa_response",
                    "name": "OA 대응",
                    "description": "거절이유통지서 분석 및 대응서 작성",
                    "steps": ["P1", "P2", "P3", "P4", "P5"],
                    "hitl_checkpoints": ["P4", "P5"],
                },
                {
                    "id": "prior_art",
                    "name": "선행기술조사",
                    "description": "다중 데이터베이스 선행기술 검색",
                    "steps": ["S1", "S2", "S3", "S4", "S5"],
                    "hitl_checkpoints": ["S5"],
                },
                {
                    "id": "translation",
                    "name": "특허번역",
                    "description": "영문 특허 → 한국어 특허 번역",
                    "steps": ["T1", "T2", "T3", "T4", "T5", "T6"],
                    "hitl_checkpoints": ["T6"],
                },
                {
                    "id": "analysis",
                    "name": "특허분석",
                    "description": "특허 포트폴리오 분석 및 경쟁사 분석",
                    "steps": ["A1", "A2", "A3", "A4", "A5"],
                    "hitl_checkpoints": ["A5"],
                },
            ]
        }

    @app.post("/api/workflows/start", response_model=WorkflowResponse)
    async def start_workflow(request: WorkflowRequest) -> WorkflowResponse:
        """Start a new workflow session."""
        from datetime import datetime
        from uuid import uuid4

        session_id = request.session_id or f"{request.workflow_type}_{uuid4().hex[:8]}"

        logger.info(
            "workflow_started",
            workflow=request.workflow_type,
            session_id=session_id,
        )

        return WorkflowResponse(
            session_id=session_id,
            status="started",
            current_step="initializing",
            message=f"{request.workflow_type} 워크플로우가 시작되었습니다.",
        )

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        """Get session status and state."""
        # TODO: Implement actual session retrieval from checkpointer
        return {
            "session_id": session_id,
            "status": "active",
            "message": "세션 상태 조회 기능은 구현 예정입니다.",
        }


def _register_copilotkit(app: FastAPI) -> None:
    """Register CopilotKit endpoint."""
    try:
        from copilotkit.integrations.fastapi import add_fastapi_endpoint
        from copilotkit import CopilotKitRemoteEndpoint

        from patent_agent.api.copilotkit_adapter import create_patent_agents

        # Create CopilotKit endpoint with patent agents
        agents = create_patent_agents()
        sdk = CopilotKitRemoteEndpoint(agents=agents)

        # Add CopilotKit endpoint at /copilotkit
        add_fastapi_endpoint(app, sdk, "/copilotkit")

        logger.info("copilotkit_endpoint_registered", path="/copilotkit")

    except ImportError as e:
        logger.warning(
            "copilotkit_not_available",
            error=str(e),
            message="CopilotKit endpoint not registered. Install copilotkit package.",
        )

        # Fallback endpoint
        @app.post("/copilotkit")
        async def copilotkit_fallback():
            raise HTTPException(
                status_code=503,
                detail="CopilotKit not available. Please install the copilotkit package.",
            )


# Create default app instance
app = create_app()


def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "patent_agent.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
