"""Autonomous Agent Loop implementing ReAct / Tool Calling."""

from __future__ import annotations

import json
import logging
from typing import Callable, Generator

from openai import OpenAI

from active_rag.config import Config
from active_rag.tools.calculator import TOOL_SCHEMA as CALC_SCHEMA, execute as execute_calc
from active_rag.tools.web_browser import WebBrowserTool
from active_rag.tools.vector_database import VectorDatabaseTool

logger = logging.getLogger(__name__)

class AgenticOrchestrator:
    """An autonomous LLM agent capable of invoking tools dynamically."""

    def __init__(self, config: Config | None = None, progress_callback: Callable[[str], None] | None = None) -> None:
        self._config = config or Config()
        self._progress_callback = progress_callback or (lambda _: None)
        
        self._client = OpenAI(
            base_url=self._config.ollama_base_url,
            api_key=self._config.api_key,
        )
        
        # Initialize tools
        self._web_tool = WebBrowserTool(self._config)
        self._vector_tool = VectorDatabaseTool(self._config)
        
        self.tools_schema = [
            CALC_SCHEMA,
            self._web_tool.schema,
            self._vector_tool.schema,
        ]
        
    def _execute_tool(self, name: str, args: str) -> str:
        """Execute a tool by name with JSON string arguments."""
        self._progress_callback(f"Executing tool: {name}...")
        try:
            parsed_args = json.loads(args)
        except json.JSONDecodeError:
            return "Error: invalid JSON arguments"
            
        if name == "calculator":
            return execute_calc(parsed_args)
        elif name == "web_browser":
            return self._web_tool.execute(parsed_args)
        elif name == "query_memory":
            return self._vector_tool.execute(parsed_args)
        else:
            return f"Error: Unknown tool '{name}'"

    def run(self, sys_prompt: str, user_prompt: str, max_steps: int = 5) -> str:
        """Run the agent loop synchronously until a final answer is produced."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        for step in range(max_steps):
            self._progress_callback(f"Agent reasoning (step {step+1})...")
            
            # Note: Provider must support tools! (e.g., GPT-3.5+, Llama-3.1 via Groq/Ollama)
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.2,
            )
            
            message = response.choices[0].message
            # OpenAI python client wraps it in a model object, so we dump to dict:
            msg_dict = message.model_dump(exclude_unset=True)
            messages.append(msg_dict)
            
            # If no tool calls, it is the final answer!
            if not getattr(message, "tool_calls", None) or len(message.tool_calls) == 0:
                return message.content or ""
                
            # Execute tool calls
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                result_str = self._execute_tool(func_name, func_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result_str,
                })
                
        return "Error: Agent reached maximum steps without finding a final answer."
