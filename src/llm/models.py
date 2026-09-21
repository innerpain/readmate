"""Model registry: model names live in configuration, never in code.

Resolution order:
  1. $AGENT_MODELS_CONFIG, else config/models.toml, when that file exists;
  2. otherwise a single profile built from the legacy env trio
     (MODEL / BASE_URL / API_KEY), so every existing entry point keeps working.

Adding a model is a config edit, not a code change.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from pathlib import Path

from src.config.settings import ModelSettings
from src.llm.types import ModelProfile

DEFAULT_CONFIG_PATH = "config/models.toml"
logger = logging.getLogger(__name__)

ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# D32 前置: the window assumed when a profile does not declare one.  Announced
# with a warning rather than applied silently -- a wrong value wastes or
# overflows the window, and the history budget is derived from it.
DEFAULT_CONTEXT_LENGTH = 32768
TOOL_REQUIRED_ROLES = ("agent",)
THINKING_MODES = {"none", "extra_body", "reasoning_effort"}
TOKEN_PARAMS = {"max_tokens", "max_completion_tokens"}


class ModelConfigError(RuntimeError):
    """Raised when the configuration cannot produce a usable profile."""


class ModelRegistry:
    def __init__(self, profiles: dict[str, ModelProfile], roles: dict[str, str]) -> None:
        self.profiles = dict(profiles)
        self.roles = dict(roles)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelRegistry":
        config_path = Path(path or os.getenv("AGENT_MODELS_CONFIG") or DEFAULT_CONFIG_PATH)
        if not config_path.exists():
            return cls._from_env()
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        raw_profiles = payload.get("profiles") or {}
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise ModelConfigError(f"{config_path} defines no [profiles.*] section")
        profiles = {
            str(name): _profile_from(str(name), dict(body))
            for name, body in raw_profiles.items()
            if isinstance(body, dict)
        }
        roles = {
            str(role): str(name)
            for role, name in (payload.get("roles") or {}).items()
            if str(name).strip()
        }
        registry = cls(profiles, roles)
        registry.validate()
        return registry

    @classmethod
    def _from_env(cls) -> "ModelRegistry":
        legacy = ModelSettings.from_env()
        if not legacy.llm_model.strip():
            raise ModelConfigError(
                "no model configured: create config/models.toml or set MODEL/BASE_URL/API_KEY"
            )
        profile = ModelProfile(
            name="main",
            model=legacy.llm_model,
            base_url=legacy.llm_base_url,
            api_key=legacy.llm_api_key,
            temperature=legacy.temperature,
            max_tokens=legacy.max_tokens,
            context_length=_resolve_context_length(None, "main"),
        )
        return cls({"main": profile}, {"agent": "main", "router": "main", "summarizer": "main"})

    # -------------------------------------------------------------- lookups
    def profile(self, name: str) -> ModelProfile:
        if name not in self.profiles:
            raise ModelConfigError(f"unknown model profile: {name}")
        return self.profiles[name]

    def resolve(self, role: str) -> ModelProfile:
        """Role -> profile, falling back to the agent role, then any profile."""

        name = self.roles.get(role) or self.roles.get("agent") or next(iter(self.profiles))
        return self.profile(name)

    def validate(self) -> None:
        if not self.profiles:
            raise ModelConfigError("no model profiles defined")
        for role, name in self.roles.items():
            if name not in self.profiles:
                raise ModelConfigError(f"role {role} points at unknown profile {name}")
        for role in TOOL_REQUIRED_ROLES:
            profile = self.resolve(role)
            if not profile.supports_tools:
                raise ModelConfigError(
                    f"profile {profile.name} is used for role {role} but supports_tools = false"
                )
        for profile in self.profiles.values():
            if not profile.model.strip():
                raise ModelConfigError(f"profile {profile.name} has no model")
            if profile.thinking not in THINKING_MODES:
                raise ModelConfigError(
                    f"profile {profile.name}: thinking must be one of {sorted(THINKING_MODES)}"
                )
            if profile.token_param not in TOKEN_PARAMS:
                raise ModelConfigError(
                    f"profile {profile.name}: token_param must be one of {sorted(TOKEN_PARAMS)}"
                )


def _profile_from(name: str, body: dict) -> ModelProfile:
    def text(key: str, default: str = "") -> str:
        return str(_expand(body.get(key, default)) or "")

    api_key = text("api_key")
    api_key_env = text("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(api_key_env, "")
    return ModelProfile(
        name=name,
        model=text("model"),
        base_url=text("base_url"),
        api_key=api_key,
        provider=text("provider", "openai-compatible") or "openai-compatible",
        temperature=float(body.get("temperature", 0.0)),
        max_tokens=int(body.get("max_tokens", 4096)),
        token_param=text("token_param", "max_tokens") or "max_tokens",
        supports_tools=bool(body.get("supports_tools", True)),
        tool_choice=text("tool_choice", "auto") or "auto",
        thinking=text("thinking", "none") or "none",
        thinking_key=text("thinking_key", "enable_thinking") or "enable_thinking",
        thinking_value=body.get("thinking_value", False),
        reasoning_effort=text("reasoning_effort"),
        extra_headers={str(k): str(v) for k, v in (body.get("extra_headers") or {}).items()},
        extra_body=dict(body.get("extra_body") or {}),
        timeout_s=float(body.get("timeout_s", 120.0)),
        max_retries=int(body.get("max_retries", 1)),
        context_length=_resolve_context_length(body.get("context_length"), name),
    )


def _resolve_context_length(declared: object, profile_name: str) -> int:
    """D32 前置: the model's window, in tokens.

    Priority: ``MODEL_CONTEXT_LENGTH`` env override -> ``context_length`` in
    models.toml -> 32768 with a warning.  A wrong value silently wastes or
    overflows the window, so the fallback is announced rather than assumed.
    """

    override = os.getenv("MODEL_CONTEXT_LENGTH", "").strip()
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            logger.warning("MODEL_CONTEXT_LENGTH=%r is not an integer; ignoring it", override)
    if declared is not None:
        try:
            value = int(declared)
            if value > 0:
                return value
        except (TypeError, ValueError):
            logger.warning("profile %s: context_length=%r is not an integer; using the default", profile_name, declared)
    logger.warning(
        "profile %s declares no context_length (and MODEL_CONTEXT_LENGTH is unset): "
        "assuming %d tokens -- set it, or the history budget is a guess",
        profile_name,
        DEFAULT_CONTEXT_LENGTH,
    )
    return DEFAULT_CONTEXT_LENGTH


def _expand(value: object) -> object:
    """Expand ${ENV_VAR} in a config string; missing vars become empty."""

    if not isinstance(value, str):
        return value
    return ENV_PLACEHOLDER_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
