"""
Patent Agent System API module.

Provides FastAPI server with CopilotKit integration for UI.
"""

from patent_agent.api.server import app, create_app

__all__ = ["app", "create_app"]
