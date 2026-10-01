"""Runtime settings, read from environment variables or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    # Comma-separated numeric Telegram user IDs allowed to use the bot.
    allowed_telegram_user_ids: str

    # Optional: enables the AI tutor (questions, deeper explanations, grading, new quizzes,
    # generated lessons). Without it the authored course still works fully.
    anthropic_api_key: str | None = None
    tutor_model: str = "claude-opus-5-5"
    tutor_effort: str = "medium"
    tutor_language: str = "English"

    data_dir: Path = Path("./data")
    # Local hour (0-23) for a gentle "continue where you left off" nudge; -1 disables it.
    nudge_hour: int = 19

    # Audio versions of lessons (🎧 buttons and /listen). "auto" uses the macOS built-in voice when
    # it is available, "none" turns audio off. List installed voices with `say -v '?'`.
    tts_backend: str = "auto"
    tts_voice: str | None = "Daniel"
    tts_rate: int = 175

    @field_validator("anthropic_api_key", "tts_voice")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None

    @property
    def allowed_ids(self) -> set[int]:
        ids = {
            int(part) for part in self.allowed_telegram_user_ids.replace(" ", "").split(",") if part
        }
        if not ids:
            raise ValueError("ALLOWED_TELEGRAM_USER_IDS must list at least one Telegram user ID")
        return ids
