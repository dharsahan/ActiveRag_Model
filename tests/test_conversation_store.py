"""Tests for persistent conversation storage."""

import tempfile
import os

from active_rag.conversation_store import ConversationStore


def test_save_and_load_conversation():
    """Messages persist across store instances."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ConversationStore(db_path)
        conv_id = store.create_conversation("Test Chat")
        store.add_message(conv_id, "user", "Hello")
        store.add_message(conv_id, "assistant", "Hi there!")

        # New instance should see the same messages
        store2 = ConversationStore(db_path)
        messages = store2.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Hi there!"
    finally:
        os.unlink(db_path)


def test_list_conversations():
    """Can list all conversations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ConversationStore(db_path)
        store.create_conversation("Chat 1")
        store.create_conversation("Chat 2")
        convs = store.list_conversations()
        assert len(convs) == 2
    finally:
        os.unlink(db_path)


def test_delete_conversation():
    """Can delete conversations and messages cascade."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ConversationStore(db_path)
        c1 = store.create_conversation("Delete Me")
        c2 = store.create_conversation("Keep Me")
        store.add_message(c1, "user", "to be deleted")
        
        store.delete_conversation(c1)
        
        assert len(store.list_conversations()) == 1
        assert store.get_messages(c1) == []
    finally:
        os.unlink(db_path)
