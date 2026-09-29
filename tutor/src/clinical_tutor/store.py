"""SQLite persistence for learner state: position, answers, spaced review, laptop queue, chat."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    first_name  TEXT,
    lesson_id   TEXT NOT NULL,
    cursor      INTEGER NOT NULL DEFAULT 0,
    awaiting    TEXT,
    created_at  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    last_nudge  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS completions (
    user_id      INTEGER NOT NULL,
    lesson_id    TEXT NOT NULL,
    completed_at REAL NOT NULL,
    PRIMARY KEY (user_id, lesson_id)
);
CREATE TABLE IF NOT EXISTS quiz_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    item_key    TEXT NOT NULL,
    context     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    chosen      INTEGER,
    correct     INTEGER,
    created_at  REAL NOT NULL,
    answered_at REAL
);
CREATE TABLE IF NOT EXISTS think_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    item_key    TEXT NOT NULL,
    answer      TEXT,
    score       INTEGER,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    user_id  INTEGER NOT NULL,
    item_key TEXT NOT NULL,
    kind     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    box      INTEGER NOT NULL,
    due_at   REAL NOT NULL,
    PRIMARY KEY (user_id, item_key)
);
CREATE TABLE IF NOT EXISTS later (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    lesson_id  TEXT NOT NULL,
    title      TEXT NOT NULL,
    task       TEXT NOT NULL,
    repo_path  TEXT,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE (user_id, lesson_id, title)
);
CREATE TABLE IF NOT EXISTS chat (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

DAY = 86_400.0
# Leitner intervals: box index -> days until the next review. Clearing the last box retires it.
REVIEW_INTERVALS_DAYS = [1, 3, 7, 21]


@dataclass
class User:
    user_id: int
    first_name: str | None
    lesson_id: str
    cursor: int
    awaiting: dict[str, Any] | None
    created_at: float
    last_seen: float
    last_nudge: float


@dataclass
class QuizRecord:
    id: int
    user_id: int
    item_key: str
    context: dict[str, Any]
    payload: dict[str, Any]
    chosen: int | None


@dataclass
class ReviewItem:
    item_key: str
    kind: str
    payload: dict[str, Any]
    box: int
    due_at: float


@dataclass
class LaterTask:
    id: int
    lesson_id: str
    title: str
    task: str
    repo_path: str | None
    done: bool


class Store:
    def __init__(self, path: Path):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> Store:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store is not open")
        return self._db

    # users -----------------------------------------------------------------------------------

    async def get_user(self, user_id: int) -> User | None:
        async with self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return User(
            user_id=row["user_id"],
            first_name=row["first_name"],
            lesson_id=row["lesson_id"],
            cursor=row["cursor"],
            awaiting=json.loads(row["awaiting"]) if row["awaiting"] else None,
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            last_nudge=row["last_nudge"],
        )

    async def create_user(self, user_id: int, first_name: str | None, lesson_id: str) -> User:
        now = time.time()
        await self.db.execute(
            "INSERT OR IGNORE INTO users (user_id, first_name, lesson_id, cursor, created_at,"
            " last_seen) VALUES (?, ?, ?, 0, ?, ?)",
            (user_id, first_name, lesson_id, now, now),
        )
        await self.db.commit()
        user = await self.get_user(user_id)
        assert user is not None
        return user

    async def list_users(self) -> list[User]:
        async with self.db.execute("SELECT user_id FROM users") as cur:
            ids = [row["user_id"] for row in await cur.fetchall()]
        return [user for uid in ids if (user := await self.get_user(uid)) is not None]

    async def set_position(self, user_id: int, lesson_id: str, cursor: int) -> None:
        await self.db.execute(
            "UPDATE users SET lesson_id = ?, cursor = ? WHERE user_id = ?",
            (lesson_id, cursor, user_id),
        )
        await self.db.commit()

    async def set_awaiting(self, user_id: int, awaiting: dict[str, Any] | None) -> None:
        value = json.dumps(awaiting) if awaiting is not None else None
        await self.db.execute("UPDATE users SET awaiting = ? WHERE user_id = ?", (value, user_id))
        await self.db.commit()

    async def touch(self, user_id: int) -> None:
        await self.db.execute(
            "UPDATE users SET last_seen = ? WHERE user_id = ?", (time.time(), user_id)
        )
        await self.db.commit()

    async def mark_nudged(self, user_id: int) -> None:
        await self.db.execute(
            "UPDATE users SET last_nudge = ? WHERE user_id = ?", (time.time(), user_id)
        )
        await self.db.commit()

    # completions -----------------------------------------------------------------------------

    async def complete_lesson(self, user_id: int, lesson_id: str) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO completions VALUES (?, ?, ?)", (user_id, lesson_id, time.time())
        )
        await self.db.commit()

    async def completed_lessons(self, user_id: int) -> set[str]:
        async with self.db.execute(
            "SELECT lesson_id FROM completions WHERE user_id = ?", (user_id,)
        ) as cur:
            return {row["lesson_id"] for row in await cur.fetchall()}

    # quizzes ---------------------------------------------------------------------------------

    async def log_quiz(
        self, user_id: int, item_key: str, context: dict[str, Any], payload: dict[str, Any]
    ) -> int:
        cur = await self.db.execute(
            "INSERT INTO quiz_log (user_id, item_key, context, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, item_key, json.dumps(context), json.dumps(payload), time.time()),
        )
        await self.db.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    async def get_quiz(self, quiz_id: int) -> QuizRecord | None:
        async with self.db.execute("SELECT * FROM quiz_log WHERE id = ?", (quiz_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return QuizRecord(
            id=row["id"],
            user_id=row["user_id"],
            item_key=row["item_key"],
            context=json.loads(row["context"]),
            payload=json.loads(row["payload"]),
            chosen=row["chosen"],
        )

    async def answer_quiz(self, quiz_id: int, chosen: int, correct: bool) -> None:
        await self.db.execute(
            "UPDATE quiz_log SET chosen = ?, correct = ?, answered_at = ? WHERE id = ?",
            (chosen, int(correct), time.time(), quiz_id),
        )
        await self.db.commit()

    async def quiz_stats(self, user_id: int) -> tuple[int, int]:
        async with self.db.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS k FROM quiz_log"
            " WHERE user_id = ? AND chosen IS NOT NULL",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        return row["n"], row["k"]

    async def lesson_quiz_stats(self, user_id: int, lesson_id: str) -> tuple[int, int]:
        async with self.db.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS k FROM quiz_log"
            " WHERE user_id = ? AND chosen IS NOT NULL AND item_key LIKE ?",
            (user_id, f"{lesson_id}#%"),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        return row["n"], row["k"]

    async def log_think(self, user_id: int, item_key: str, answer: str | None, score: int) -> None:
        await self.db.execute(
            "INSERT INTO think_log (user_id, item_key, answer, score, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, item_key, answer, score, time.time()),
        )
        await self.db.commit()

    async def answered_think_keys(self, user_id: int) -> set[str]:
        async with self.db.execute(
            "SELECT DISTINCT item_key FROM think_log WHERE user_id = ?", (user_id,)
        ) as cur:
            return {row["item_key"] for row in await cur.fetchall()}

    # spaced review ---------------------------------------------------------------------------

    async def schedule_review(
        self, user_id: int, item_key: str, kind: str, payload: dict[str, Any], box: int
    ) -> None:
        due = time.time() + REVIEW_INTERVALS_DAYS[box] * DAY
        await self.db.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (user_id, item_key) DO UPDATE SET box = excluded.box,"
            " due_at = excluded.due_at, payload = excluded.payload",
            (user_id, item_key, kind, json.dumps(payload), box, due),
        )
        await self.db.commit()

    async def get_review(self, user_id: int, item_key: str) -> ReviewItem | None:
        async with self.db.execute(
            "SELECT * FROM reviews WHERE user_id = ? AND item_key = ?", (user_id, item_key)
        ) as cur:
            row = await cur.fetchone()
        return _review(row) if row else None

    async def record_review(self, user_id: int, item_key: str, success: bool) -> None:
        """Move a review item up a box on success (retiring it after the last) or back to box 0."""
        item = await self.get_review(user_id, item_key)
        if item is None:
            return
        if not success:
            await self.schedule_review(user_id, item_key, item.kind, item.payload, 0)
            return
        if item.box + 1 >= len(REVIEW_INTERVALS_DAYS):
            await self.db.execute(
                "DELETE FROM reviews WHERE user_id = ? AND item_key = ?", (user_id, item_key)
            )
            await self.db.commit()
            return
        await self.schedule_review(user_id, item_key, item.kind, item.payload, item.box + 1)

    async def next_due_review(self, user_id: int) -> ReviewItem | None:
        async with self.db.execute(
            "SELECT * FROM reviews WHERE user_id = ? AND due_at <= ? ORDER BY due_at LIMIT 1",
            (user_id, time.time()),
        ) as cur:
            row = await cur.fetchone()
        return _review(row) if row else None

    async def review_counts(self, user_id: int) -> tuple[int, int, float | None]:
        """Return (due now, total scheduled, next due timestamp)."""
        async with self.db.execute(
            "SELECT COALESCE(SUM(due_at <= ?), 0) AS due, COUNT(*) AS total, MIN(due_at) AS nxt"
            " FROM reviews WHERE user_id = ?",
            (time.time(), user_id),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        return row["due"], row["total"], row["nxt"]

    # laptop queue ----------------------------------------------------------------------------

    async def add_later(
        self, user_id: int, lesson_id: str, title: str, task: str, repo_path: str | None
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO later (user_id, lesson_id, title, task, repo_path, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, lesson_id, title, task, repo_path, time.time()),
        )
        await self.db.commit()

    async def list_later(self, user_id: int, include_done: bool = False) -> list[LaterTask]:
        query = "SELECT * FROM later WHERE user_id = ?"
        if not include_done:
            query += " AND done = 0"
        async with self.db.execute(query + " ORDER BY id", (user_id,)) as cur:
            rows = await cur.fetchall()
        return [
            LaterTask(
                id=row["id"],
                lesson_id=row["lesson_id"],
                title=row["title"],
                task=row["task"],
                repo_path=row["repo_path"],
                done=bool(row["done"]),
            )
            for row in rows
        ]

    async def finish_later(self, user_id: int, task_id: int) -> None:
        await self.db.execute(
            "UPDATE later SET done = 1 WHERE user_id = ? AND id = ?", (user_id, task_id)
        )
        await self.db.commit()

    # tutor chat ------------------------------------------------------------------------------

    async def add_chat(self, user_id: int, role: str, content: str) -> None:
        await self.db.execute(
            "INSERT INTO chat (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time()),
        )
        await self.db.commit()

    async def recent_chat(self, user_id: int, limit: int = 12) -> list[dict[str, str]]:
        async with self.db.execute(
            "SELECT role, content FROM chat WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def _review(row: aiosqlite.Row) -> ReviewItem:
    return ReviewItem(
        item_key=row["item_key"],
        kind=row["kind"],
        payload=json.loads(row["payload"]),
        box=row["box"],
        due_at=row["due_at"],
    )
