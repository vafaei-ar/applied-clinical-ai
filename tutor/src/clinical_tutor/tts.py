"""Text-to-speech backends and a cache for rendered audio.

A backend turns a *script* (see ``speech.py``: strings to speak and floats of silence) into an
audio file Telegram can play. The only backend shipped is the macOS built-in voice, which needs no
installation or account. Anything with a ``synthesize(script, out)`` coroutine, a ``name``, and a
``fingerprint`` can be plugged in, so better voices can be added without touching the bot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from .speech import Script

log = logging.getLogger(__name__)

DEFAULT_RATE = 175


class TTSError(Exception):
    """Synthesis failed; the message is safe to show the learner."""


class TTS(Protocol):
    name: str

    @property
    def fingerprint(self) -> str:
        """Identifies the voice settings, so cached audio is reused only for the same voice."""
        ...

    async def synthesize(self, script: Script, out: Path) -> None: ...


async def _run(*cmd: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[:300]
        raise TTSError(f"{cmd[0]} failed ({proc.returncode}): {detail}")


def macos_voices() -> list[str]:
    """Names of the voices installed on this Mac (empty if ``say`` is unavailable)."""
    import subprocess

    if not shutil.which("say"):
        return []
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False)
    names = []
    for line in result.stdout.splitlines():
        head = line.split("#")[0].rstrip()
        parts = head.rsplit(None, 1)  # "Samantha (English (US))   en_US" -> name, locale
        if len(parts) == 2:
            full = parts[0].strip()
            names.append(full)
            base = full.split(" (")[
                0
            ]  # "Samantha (English (US))" is also addressable as "Samantha"
            if base != full:
                names.append(base)
    return list(dict.fromkeys(names))


class MacSay:
    """The macOS ``say`` voice, encoded to AAC (m4a) with ``afconvert``."""

    name = "macos-say"

    def __init__(self, voice: str | None = None, rate: int = DEFAULT_RATE):
        self.voice = voice
        self.rate = rate

    @property
    def fingerprint(self) -> str:
        return f"{self.name}|{self.voice or 'default'}|{self.rate}"

    @staticmethod
    def render(script: Script) -> str:
        """The text handed to ``say``: speech, with silences as embedded ``[[slnc ms]]`` commands."""
        parts: list[str] = []
        for item in script:
            if isinstance(item, float):
                parts.append(f"[[slnc {int(item * 1000)}]]")
            else:
                # Never let lesson text smuggle in its own embedded speech commands.
                parts.append(item.replace("[[", "[").replace("]]", "]"))
        return " ".join(parts)

    async def synthesize(self, script: Script, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tutor-tts-") as tmp:
            text_file = Path(tmp) / "script.txt"
            aiff = Path(tmp) / "speech.aiff"
            text_file.write_text(self.render(script), encoding="utf-8")
            cmd = ["say", "-r", str(self.rate), "-o", str(aiff), "-f", str(text_file)]
            if self.voice:
                cmd[1:1] = ["-v", self.voice]
            await _run(*cmd)
            await _run("afconvert", "-f", "m4af", "-d", "aac", "-b", "64000", str(aiff), str(out))


def make_tts(
    backend: str = "auto", voice: str | None = None, rate: int = DEFAULT_RATE
) -> TTS | None:
    """Build the configured backend, or ``None`` when there isn't one (the 🎧 buttons then hide)."""
    backend = backend.lower()
    if backend == "none":
        return None
    if backend in {"auto", "macos"}:
        if shutil.which("say") and shutil.which("afconvert"):
            if voice and voice not in macos_voices():
                log.warning("Voice %r is not installed; using the system default voice.", voice)
                voice = None
            return MacSay(voice, rate)
        if backend == "macos":
            raise TTSError("TTS_BACKEND=macos needs the macOS `say` and `afconvert` commands.")
        return None
    raise TTSError(f"Unknown TTS_BACKEND {backend!r} (use auto, macos, or none).")


class AudioCache:
    """Renders scripts to audio files, once per distinct script and voice."""

    def __init__(self, directory: Path, tts: TTS, concurrency: int = 1):
        self.directory = directory
        self.tts = tts
        self._locks: dict[str, asyncio.Lock] = {}
        self._slots = asyncio.Semaphore(concurrency)

    def key(self, script: Script) -> str:
        payload = json.dumps([self.tts.fingerprint, script], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def cached(self, script: Script) -> bool:
        path = self.directory / f"{self.key(script)}.m4a"
        return path.is_file() and path.stat().st_size > 0

    async def get(self, script: Script) -> Path:
        key = self.key(script)
        path = self.directory / f"{key}.m4a"
        if path.is_file() and path.stat().st_size > 0:
            return path
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if path.is_file() and path.stat().st_size > 0:
                return path
            async with self._slots:
                self.directory.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(".partial.m4a")
                try:
                    await self.tts.synthesize(script, partial)
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
        return path
