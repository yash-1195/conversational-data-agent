from enum import Enum
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Verbosity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"


class OnViolation(str, Enum):
    REJECT = "reject"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    provider: str
    base_url: str
    api_key_env_var: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    max_tool_retries: int = Field(ge=1)
    max_context_turns: int = Field(ge=1)

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env_var)
        if not key:
            raise EnvironmentError(
                f"Environment variable '{self.api_key_env_var}' is not set. "
                f"Add it to your .env file."
            )
        return key


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_plan_attempts: int = Field(ge=1)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout_s: int = Field(gt=0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verbosity: Verbosity
    log_to_file: bool


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_upload_size_mb: int = Field(gt=0)
    max_row_limit: int = Field(gt=0)
    on_violation: OnViolation


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_rows: int = Field(gt=0)
    max_chat_history_display: int = Field(gt=0)
    plot_dir: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm: LLMConfig
    agent: AgentConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    data: DataConfig
    ui: UIConfig


def load_config(path: str = "configs/config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    try:
        return AppConfig(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {config_path}:\n{e}")