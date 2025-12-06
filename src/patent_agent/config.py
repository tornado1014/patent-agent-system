"""
Configuration management using Pydantic Settings.

Loads settings from:
1. Environment variables
2. .env file
3. configs/settings.yaml
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096


class AnthropicSettings(BaseSettings):
    """Anthropic API settings."""

    api_key: SecretStr = Field(default=..., alias="ANTHROPIC_API_KEY")


class OpenAISettings(BaseSettings):
    """OpenAI API settings."""

    api_key: SecretStr = Field(default="", alias="OPENAI_API_KEY")


class KIPRISSettings(BaseSettings):
    """KIPRIS (Korean Patent DB) settings."""

    api_key: SecretStr = Field(default="", alias="KIPRIS_API_KEY")
    base_url: str = "http://plus.kipris.or.kr/openapi/rest"
    timeout: int = 30


class USPTOSettings(BaseSettings):
    """USPTO (US Patent DB) settings."""

    # USPTO APIs are generally public but may require API key for some endpoints
    api_key: SecretStr = Field(default="", alias="USPTO_API_KEY")
    timeout: int = 30


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    provider: Literal["korpatelectra", "openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-small"  # For OpenAI
    device: str = "cpu"  # For local models


class DocumentParserSettings(BaseSettings):
    """Document parser settings."""

    pdf_parser: Literal["pymupdf", "docling"] = "pymupdf"
    max_file_size_mb: int = 50
    extract_images: bool = False


class VectorStoreSettings(BaseSettings):
    """Vector store settings."""

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    provider: Literal["chroma", "pinecone"] = "chroma"
    persist_directory: Path = Field(
        default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIRECTORY"
    )
    collection_name: str = "patent_documents"


class WorkflowSettings(BaseSettings):
    """Workflow execution settings."""

    max_iterations: int = 3
    enable_human_in_loop: bool = True
    quality_threshold: int = 80  # 0-100 score


class MonitoringSettings(BaseSettings):
    """Monitoring and tracing settings."""

    enabled: bool = False
    provider: Literal["opik", "langsmith"] = "opik"
    project_name: str = "patent-agent-system"


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Patent Agent System"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = False

    # Sub-settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    kipris: KIPRISSettings = Field(default_factory=KIPRISSettings)
    uspto: USPTOSettings = Field(default_factory=USPTOSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    document_parser: DocumentParserSettings = Field(default_factory=DocumentParserSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Convenience properties for API keys
    @property
    def kipris_api_key(self) -> str | None:
        """Get KIPRIS API key."""
        key = self.kipris.api_key.get_secret_value()
        return key if key else None

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key."""
        key = self.openai.api_key.get_secret_value()
        return key if key else None

    @property
    def anthropic_api_key(self) -> str | None:
        """Get Anthropic API key."""
        key = self.anthropic.api_key.get_secret_value()
        return key if key else None

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "Settings":
        """Load settings from YAML file with env overrides."""
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f) or {}
            return cls(**yaml_data)
        return cls()


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Loads from configs/settings.yaml if exists, then overrides with env vars.
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
    return Settings.from_yaml(config_path)


# Convenience accessor
settings = get_settings()
