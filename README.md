# Patent Agent System

Enterprise-grade multi-agent system for Korean patent workflows.

## Overview

This system automates 5 key patent workflows using LangGraph-based multi-agent architecture:

| Workflow | Description | Key Features |
|----------|-------------|--------------|
| **명세서 작성** | Patent Specification Writing | 9-step workflow (E1-E9), PatentSpec-KR v2.0 |
| **OA 대응** | Office Action Response | Zero Hallucination, PALLAS-EVIDENCE v3.0 |
| **선행기술조사** | Prior Art Search | Multi-DB (KIPRIS, USPTO), KorPatELECTRA |
| **특허번역** | Patent Translation (EN→KR) | 4 Absolute Laws, 5C Quality |
| **특허분석** | Patent Analysis | Portfolio, Competitor, Infringement |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Router                               │
│   Routes user requests to appropriate workflow               │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────┬─────────┬─────────┬─────────┬─────────┐
        ▼         ▼         ▼         ▼         ▼         │
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐│
   │ 명세서  ││  OA     ││ 선행    ││  번역   ││  분석   ││
   │ 작성    ││ 대응    ││ 기술    ││         ││         ││
   │ Graph   ││ Graph   ││ Graph   ││ Graph   ││ Graph   ││
   └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘│
        │         │         │         │         │         │
        └─────────┴─────────┴─────────┴─────────┴─────────┘
                              │
                    ┌─────────────────┐
                    │ Reflexion Loop  │
                    │ Draft→Critique  │
                    │   →Revise       │
                    └─────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd patent-agent-system

# Install dependencies with Poetry
poetry install

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
```

## Configuration

### Environment Variables

```bash
# LLM API Keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Patent Database APIs
KIPRIS_API_KEY=your_key_here
```

### Settings

Edit `configs/settings.yaml` for workflow configuration:

```yaml
workflow:
  max_iterations: 3
  enable_human_in_loop: true
  quality_threshold: 80
```

## Usage

### Interactive Mode

```bash
poetry run patent-agent run
```

### Programmatic Usage

```python
from patent_agent.graphs.main_router import run_patent_task

# Run a patent task
result = await run_patent_task("OA 거절이유 분석해줘")
print(result["current_workflow"])  # "oa_response"
```

## Project Structure

```
patent-agent-system/
├── src/patent_agent/
│   ├── agents/          # Domain-specific agents (TODO)
│   ├── graphs/          # LangGraph workflows
│   │   ├── base_graph.py
│   │   ├── reflexion.py
│   │   └── main_router.py
│   ├── rules/           # Parlant-style guidelines
│   │   ├── base.py
│   │   ├── translation_laws.py
│   │   ├── oa_response_laws.py
│   │   └── spec_writing_laws.py
│   ├── state/           # State definitions
│   │   ├── base.py
│   │   ├── spec_writing.py
│   │   ├── oa_response.py
│   │   ├── prior_art.py
│   │   ├── translation.py
│   │   └── analysis.py
│   ├── tools/           # External tool wrappers (TODO)
│   └── templates/       # Output templates (TODO)
├── tests/
│   ├── conftest.py
│   └── unit/
├── configs/
│   └── settings.yaml
└── pyproject.toml
```

## Key Concepts

### Reflexion Loop

All workflows use Draft → Critique → Revise pattern for quality assurance:
- Maximum 3 iterations
- Quality threshold: 80/100
- Human-in-the-Loop at critical checkpoints

### Zero Hallucination (OA Response)

- **LAW-6**: Verbatim quoting only
- **LAW-7**: Source traceability `[Doc:Page:Para]`
- **LAW-8**: No content creation from unsupported sources
- **LAW-9**: Confidence indicators (🟢🟡🔴)

### Translation Laws

- **LAW-T-1**: '상기' usage (claims only)
- **LAW-T-2**: Single sentence claims
- **LAW-T-3**: Critical terms (comprising ≠ consisting of)
- **LAW-T-4**: Drawing references in parentheses

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/patent_agent

# Run specific test file
poetry run pytest tests/unit/test_rules.py -v
```

## Development Roadmap

- [x] Phase 0: Claude Code Skills setup
- [x] Phase 1: Core infrastructure
- [ ] Phase 2: Data integration (KIPRIS, USPTO)
- [ ] Phase 3: Spec writing workflow
- [ ] Phase 4: OA response workflow
- [ ] Phase 5: Prior art search
- [ ] Phase 6: Translation workflow
- [ ] Phase 7: Analysis workflow
- [ ] Phase 8: UI/Monitoring

## License

[Add license information]

## Contributing

[Add contributing guidelines]
