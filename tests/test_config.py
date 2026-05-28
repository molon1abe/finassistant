import pytest
from pydantic import ValidationError

from finassistant.core.config import CloudModelConfig, LocalModelConfig, Settings


class TestCloudModelConfig:
    def test_defaults(self):
        config = CloudModelConfig(
            openai_api_key="test-api-key"  # pragma: allowlist secret
        )
        assert config.type == "cloud"
        assert config.model_name == "gpt-4o-mini"
        assert config.openai_api_key

    def test_requires_api_key(self):
        with pytest.raises(ValidationError):
            CloudModelConfig()


class TestLocalModelConfig:
    def test_defaults(self):
        config = LocalModelConfig()
        assert config.type == "local"
        assert config.model_name == "mistral"
        assert config.base_url == "http://localhost:11434"

    def test_custom_values(self):
        config = LocalModelConfig(
            model_name="qwen2.5:7b", base_url="http://myhost:11434"
        )
        assert config.model_name == "qwen2.5:7b"
        assert config.base_url == "http://myhost:11434"


class TestSettings:
    def test_default_model_is_cloud(self, monkeypatch):
        monkeypatch.setenv(
            "MODEL",
            '{"type": "cloud", "openai_api_key": "test-api-key"}',  # pragma: allowlist secret
        )
        settings = Settings()
        assert isinstance(settings.model, CloudModelConfig)

    def test_json_env_selects_local_model(self, monkeypatch):
        monkeypatch.setenv(
            "MODEL",
            '{"type": "local", "model_name": "mistral", "base_url": "http://localhost:11434"}',
        )
        settings = Settings()
        assert isinstance(settings.model, LocalModelConfig)
        assert settings.model.model_name == "mistral"

    def test_json_env_selects_cloud_model(self, monkeypatch):
        monkeypatch.setenv(
            "MODEL",
            '{"type": "cloud", "openai_api_key": "test-api-key", "model_name": "gpt-4o-mini"}',  # pragma: allowlist secret
        )
        settings = Settings()
        assert isinstance(settings.model, CloudModelConfig)
        assert (
            settings.model.openai_api_key == "test-api-key"  # pragma: allowlist secret
        )

    def test_invalid_model_type_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL", '{"type": "unknown"}')
        with pytest.raises(ValidationError):
            Settings()

    def test_default_paths(self):
        settings = Settings()
        assert settings.db_path == "db"
        assert settings.source_file == "tests/data/bank_statement.pdf"

    def test_log_level_default(self):
        settings = Settings()
        assert settings.log_level == "INFO"
