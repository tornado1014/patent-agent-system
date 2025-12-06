# Contributing to Patent Agent System

Thank you for your interest in contributing to Patent Agent System!

## Development Setup

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/patent-agent-system.git
cd patent-agent-system

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

## Development Workflow

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:
```bash
feat(router): add LLM-based workflow detection
fix(translation): correct '상기' validation in claims
docs(readme): add installation instructions
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/patent_agent

# Run specific test file
poetry run pytest tests/unit/test_rules.py -v
```

### Code Quality

```bash
# Run linter
poetry run ruff check src/ tests/

# Run formatter
poetry run ruff format src/ tests/

# Run type checker
poetry run mypy src/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a PR to `develop`

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Conventional commit messages used

## Release Process

1. PRs are merged to `develop`
2. When ready for release, create PR from `develop` to `main`
3. After merge, create a version tag:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```
4. GitHub Actions will automatically create a release

## Code of Conduct

Please be respectful and professional in all interactions.

## Questions?

Open an issue with the `question` label.
