"""
Patent Agent System - Main Entry Point

Usage:
    patent-agent [OPTIONS] COMMAND [ARGS]...

Commands:
    run       Run a patent task interactively
    server    Start the API server
    version   Show version information
"""

import asyncio
import sys
from typing import Optional

import structlog

from patent_agent import __version__
from patent_agent.config import settings
from patent_agent.graphs.main_router import MainRouter, create_main_graph, run_patent_task
from patent_agent.state.base import create_initial_state


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def print_banner():
    """Print application banner."""
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║           Patent Agent System v{__version__:<26}║
║  Enterprise-grade Multi-Agent System for Patent Workflows     ║
╠═══════════════════════════════════════════════════════════════╣
║  Workflows:                                                   ║
║    • 명세서 작성 (Patent Specification Writing)               ║
║    • OA 대응 (Office Action Response)                         ║
║    • 선행기술조사 (Prior Art Search)                          ║
║    • 특허번역 (Patent Translation EN→KR)                      ║
║    • 특허분석 (Patent Analysis)                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    print("\nType your request (or 'quit' to exit):")
    print("-" * 50)

    router = MainRouter()
    graph = create_main_graph(router)

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n감사합니다. 프로그램을 종료합니다.")
                break

            if not user_input:
                continue

            # Detect workflow
            workflow = router.detect_workflow(user_input)
            print(f"\n🔍 Detected workflow: {workflow}")

            # Run the task
            print("⏳ Processing...")
            result = await run_patent_task(user_input)

            # Display result summary
            print("\n" + "=" * 50)
            print("📋 Result Summary")
            print("=" * 50)
            print(f"  Workflow: {result.get('current_workflow')}")
            print(f"  Step: {result.get('current_step')}")
            print(f"  Quality Score: {result.get('quality_score', {}).get('overall', 'N/A')}")

            if result.get("is_error_state"):
                print(f"  ⚠️ Errors: {result.get('error_messages')}")

        except KeyboardInterrupt:
            print("\n\n중단되었습니다.")
            break
        except Exception as e:
            logger.exception("Error processing request", error=str(e))
            print(f"\n❌ Error: {e}")


def main(args: Optional[list[str]] = None):
    """Main entry point."""
    args = args or sys.argv[1:]

    if not args or args[0] == "run":
        # Interactive mode
        asyncio.run(interactive_mode())
    elif args[0] == "version":
        print(f"Patent Agent System v{__version__}")
    elif args[0] == "server":
        # Start FastAPI server with CopilotKit integration
        from patent_agent.api.server import main as run_server
        print_banner()
        print("\n🚀 Starting API server with CopilotKit integration...")
        print("   Server: http://localhost:8000")
        print("   CopilotKit: http://localhost:8000/copilotkit")
        print("   Health: http://localhost:8000/health")
        print("-" * 50)
        run_server()
    else:
        print(f"Unknown command: {args[0]}")
        print("Available commands: run, server, version")
        sys.exit(1)


if __name__ == "__main__":
    main()
