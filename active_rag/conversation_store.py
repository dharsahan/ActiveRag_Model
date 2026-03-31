"""SQLite-based persistent conversation storage."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime


class ConversationStore:
    """Persists conversations to SQLite for cross-session memory."""

    def __init__(self, db_path: str = "conversations.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
        """)

    def create_conversation(self, title: str = "New Chat") -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        self._conn.commit()
        return conv_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        self._conn.commit()

    def get_messages(self, conversation_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()
