"""Conversation memory for follow-up questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """A single message in the conversation."""
    
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class ConversationMemory:
    """Maintains conversation history for context-aware responses."""
    
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 20  # Keep last N messages to avoid token limits
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._add_message(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self._add_message(Message(role="assistant", content=content))
    
    def _add_message(self, message: Message) -> None:
        """Add a message and trim if necessary."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # Keep system messages, trim oldest user/assistant messages
            self.messages = self.messages[-self.max_messages:]
    
    def get_context_messages(self) -> list[dict[str, str]]:
        """Get messages formatted for OpenAI API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context."""
        if not self.messages:
            return ""
        
        recent = self.messages[-6:]  # Last 3 exchanges
        lines = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{prefix}: {content}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def is_followup_question(self, query: str) -> bool:
        """Detect if query is likely a follow-up question."""
        followup_indicators = [
            "what about", "how about", "and ", "also ", "what else",
            "tell me more", "explain", "why", "can you", "could you",
            "it", "that", "this", "they", "them", "those", "these",
            "the same", "similar", "related", "more details",
        ]
        query_lower = query.lower().strip()
        
        # Short queries after conversation are likely follow-ups
        if len(self.messages) > 0 and len(query.split()) < 5:
            return True
        
        return any(indicator in query_lower for indicator in followup_indicators)
    
    def enhance_query_with_context(self, query: str) -> str:
        """Enhance a follow-up query with conversation context."""
        if not self.messages or not self.is_followup_question(query):
            return query
        
        summary = self.get_conversation_summary()
        return (
            f"Given this conversation context:\n{summary}\n\n"
            f"Answer this follow-up question: {query}"
        )
