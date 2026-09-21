"""Model registry loading and request-kwargs mapping (T5.0.8)."""

from __future__ import annotations

import pytest

from src.llm.client import LLMClient
from src.llm.models import ModelConfigError, ModelRegistry

_EXAMPLE_TOML = """\
version = 1

[roles]
agent = "main"
router = "cheap"
summarizer = "cheap"

[profiles.main]
provider = "openai-compatible"
base_url = "${MAIN_BASE_URL}"
api_key_env = "MAIN_API_KEY"
model = "your-main-model"
temperature = 0.0
max_tokens = 4096
supports_tools = true
tool_choice = "auto"
token_param = "max_tokens"
thinking = "extra_body"
thinking_key = "enable_thinking"
thinking_value = false

[profiles.cheap]
provider = "openai-compatible"
base_url = "${CHEAP_BASE_URL}"
api_key_env = "CHEAP_API_KEY"
model = "your-cheap-model"
max_tokens = 1024
supports_tools = false

[profiles.local]
provider = "openai-compatible"
base_url = "http://127.0.0.1:11434/v1"
api_key_env = ""
model = "your-local-model"
max_tokens = 2048
supports_tools = true
timeout_s = 300.0
"""


def _clear_legacy_env(monkeypatch) -> None:
    for name in (
        "MODEL",
        "BASE_URL",
        "API_KEY",
        "model",
        "base_url",
        "api_key",
        "AGENT_MODELS_CONFIG",
        "MAIN_BASE_URL",
        "MAIN_API_KEY",
        "CHEAP_BASE_URL",
        "CHEAP_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_env_fallback_builds_main_profile(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    monkeypatch.setenv("MODEL", "env-model-x")
    monkeypatch.setenv("BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("API_KEY", "env-key")

    missing = tmp_path / "missing.toml"
    registry = ModelRegistry.load(missing)

    profile = registry.resolve("agent")
    assert profile.name == "main"
    assert profile.model == "env-model-x"
    assert "main" in registry.profiles


def test_missing_config_and_model_raises(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    missing = tmp_path / "missing.toml"

    with pytest.raises(ModelConfigError) as exc:
        ModelRegistry.load(missing)

    assert "create config/models.toml or set MODEL/BASE_URL/API_KEY" in str(exc.value)


def test_example_toml_loads_profiles_and_roles(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    monkeypatch.setenv("MAIN_BASE_URL", "http://main.test/v1")
    monkeypatch.setenv("MAIN_API_KEY", "main-secret")
    monkeypatch.setenv("CHEAP_BASE_URL", "http://cheap.test/v1")
    monkeypatch.setenv("CHEAP_API_KEY", "cheap-secret")

    path = tmp_path / "models.toml"
    path.write_text(_EXAMPLE_TOML, encoding="utf-8")
    registry = ModelRegistry.load(path)

    assert set(registry.profiles) >= {"main", "cheap", "local"}
    assert registry.roles["router"] == "cheap"
    assert registry.roles["agent"] == "main"
    assert registry.roles["summarizer"] == "cheap"


def test_role_pointing_at_unknown_profile_raises(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "ghost"
[profiles.main]
model = "m"
supports_tools = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError):
        ModelRegistry.load(path)


def test_agent_role_requires_tool_support(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "cheap"
router = "cheap"
[profiles.cheap]
model = "cheap-model"
supports_tools = false
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError) as exc:
        ModelRegistry.load(path)

    assert "supports_tools" in str(exc.value)


def test_api_key_env_is_resolved(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    monkeypatch.setenv("X", "secret")
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "main"
[profiles.main]
model = "m"
api_key_env = "X"
supports_tools = true
""",
        encoding="utf-8",
    )

    registry = ModelRegistry.load(path)
    assert registry.profile("main").api_key == "secret"


def test_base_url_env_placeholder_expands(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    monkeypatch.setenv("MAIN_BASE_URL", "http://expanded.test/v1")
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "main"
[profiles.main]
model = "m"
base_url = "${MAIN_BASE_URL}"
supports_tools = true
""",
        encoding="utf-8",
    )

    registry = ModelRegistry.load(path)
    assert registry.profile("main").base_url == "http://expanded.test/v1"


def test_thinking_extra_body_lands_in_request_kwargs(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "main"
[profiles.main]
model = "m"
supports_tools = true
thinking = "extra_body"
thinking_key = "enable_thinking"
thinking_value = false
""",
        encoding="utf-8",
    )

    profile = ModelRegistry.load(path).profile("main")
    kwargs = LLMClient(profile).request_kwargs([{"role": "user", "content": "hi"}])
    assert kwargs["extra_body"] == {"enable_thinking": False}


def test_token_param_max_completion_tokens(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "main"
[profiles.main]
model = "m"
supports_tools = true
token_param = "max_completion_tokens"
max_tokens = 256
""",
        encoding="utf-8",
    )

    profile = ModelRegistry.load(path).profile("main")
    kwargs = LLMClient(profile).request_kwargs([{"role": "user", "content": "hi"}])
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 256


def test_tools_omitted_when_profile_disables_them(tmp_path, monkeypatch):
    _clear_legacy_env(monkeypatch)
    path = tmp_path / "models.toml"
    path.write_text(
        """
version = 1
[roles]
agent = "main"
router = "cheap"
[profiles.main]
model = "m"
supports_tools = true
[profiles.cheap]
model = "c"
supports_tools = false
""",
        encoding="utf-8",
    )

    profile = ModelRegistry.load(path).profile("cheap")
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    kwargs = LLMClient(profile).request_kwargs(
        [{"role": "user", "content": "hi"}],
        tools=tools,
    )
    assert "tools" not in kwargs
