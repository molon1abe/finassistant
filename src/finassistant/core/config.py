from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudModelConfig(BaseModel):
    type: Literal["cloud"] = "cloud"
    openai_api_key: str
    model_name: str = "gpt-4o-mini"


class LocalModelConfig(BaseModel):
    type: Literal["local"] = "local"
    model_name: str = "mistral"
    base_url: str = "http://localhost:11434"


ModelConfig = Annotated[
    Union[CloudModelConfig, LocalModelConfig], Field(discriminator="type")
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    model: ModelConfig = CloudModelConfig(openai_api_key="")
    source_file: str = "tests/data/bank_statement.pdf"
    db_path: str = "db"
    log_level: str = "INFO"
