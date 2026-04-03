"""Conversation memory for follow-up questions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI
from active_rag.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    
    role: Literal["user", "assistant", "system"]
    content: str


class ConversationMemory:
    """Maintains conversation history for context-aware responses with summarization."""
    
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.messages: list[Message] = []
        self.max_window: int = 10  # Keep last 10 messages, summarize earlier ones
        self.summary: str = ""
        
        # client for summarization
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.ollama_base_url,
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._add_message(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self._add_message(Message(role="assistant", content=content))
    
    def _add_message(self, message: Message) -> None:
        """Add a message and trigger summarization if window is exceeded."""
        self.messages.append(message)
        if len(self.messages) > self.max_window:
            self._summarize_old_messages()

    def _summarize_old_messages(self) -> None:
        """Summarize the oldest messages and keep only the recent window."""
        # messages to summarize: everything except the last N
        to_summarize = self.messages[:-self.max_window]
        self.messages = self.messages[-self.max_window:]
        
        history_str = "\n".join([f"{m.role}: {m.content}" for m in to_summarize])
        
        prompt = (
            f"Existing Summary: {self.summary or 'None'}\n\n"
            f"New conversation segment to include:\n{history_str}\n\n"
            "Update the summary to capture all key facts and context from the new segment. "
            "Keep it concise but detailed regarding entities and findings. Return ONLY the new summary text."
        )
        
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a memory summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            self.summary = response.choices[0].message.content or ""
            logger.info("Memory summarized successfully.")
        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")

    def get_context_messages(self) -> list[dict[str, str]]:
        """Get messages formatted for OpenAI API, including the summary."""
        history = []
        if self.summary:
            history.append({
                "role": "system", 
                "content": f"Summary of previous conversation: {self.summary}"
            })
            
        for msg in self.messages:
            history.append({"role": msg.role, "content": msg.content})
            
        return history
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context."""
        if not self.messages and not self.summary:
            return ""
        
        recent = self.messages[-6:]  # Last 3 exchanges
        lines = []
        if self.summary:
            lines.append(f"Summary: {self.summary[:200]}...")
            
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{prefix}: {content}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.summary = ""
    
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
        if (len(self.messages) > 0 or self.summary) and len(query.split()) < 5:
            return True
        
        return any(indicator in query_lower for indicator in followup_indicators)
    
    def enhance_query_with_context(self, query: str) -> str:
        """Enhance a follow-up query with conversation context."""
        if not (self.messages or self.summary) or not self.is_followup_question(query):
            return query
        
        summary = self.get_conversation_summary()
        return (
            f"Given this conversation context:\n{summary}\n\n"
            f"Answer this follow-up question: {query}"
        )
