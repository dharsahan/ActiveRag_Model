"""Autonomous Agent Loop implementing ReAct / Tool Calling."""

from __future__ import annotations

import json
import logging
from typing import Callable, Generator, AsyncGenerator, Any

from openai import OpenAI

from active_rag.config import Config
from active_rag.pipeline import PipelineResult
from active_rag.answer_generator import Answer
from active_rag.memory import ConversationMemory
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
        self._memory = ConversationMemory(self._config)
        
        self.tools_schema = [
            CALC_SCHEMA,
            self._web_tool.schema,
            self._vector_tool.schema,
        ]
        
    def _execute_tool(self, name: str, args: str) -> str:
        """Execute a tool by name with JSON string arguments (Sync)."""
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

    async def _execute_tool_async(self, name: str, args: str) -> str:
        """Execute a tool by name with JSON string arguments (Async)."""
        self._progress_callback(f"Executing tool: {name}...")
        try:
            parsed_args = json.loads(args)
        except json.JSONDecodeError:
            return "Error: invalid JSON arguments"
            
        if name == "calculator":
            return execute_calc(parsed_args)
        elif name == "web_browser":
            return await self._web_tool.execute_async(parsed_args)
        elif name == "query_memory":
            return await self._vector_tool.execute_async(parsed_args)
        else:
            return f"Error: Unknown tool '{name}'"

    def run(self, query: str, max_steps: int = 5) -> PipelineResult:
        """Run the agent loop synchronously until a final answer is produced."""
        # Use windowed memory history
        messages = self._memory.get_context_messages()
        # Add current query
        messages.append({"role": "user", "content": query})
        
        final_answer = ""
        citations = []
        
        for step in range(max_steps):
            self._progress_callback(f"Agent reasoning (step {step+1})...")
            
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.2,
            )
            
            message = response.choices[0].message
            msg_dict = message.model_dump(exclude_unset=True)
            messages.append(msg_dict)
            
            if not getattr(message, "tool_calls", None) or len(message.tool_calls) == 0:
                final_answer = message.content or ""
                break
                
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                result_str = self._execute_tool(func_name, func_args)
                
                # If web search was used, we might want to collect URLs as citations
                if func_name == "web_browser":
                    try:
                        args_data = json.loads(func_args)
                        if "url" in args_data:
                            citations.append(args_data["url"])
                    except:
                        pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result_str,
                })
        
        if not final_answer:
            final_answer = "Error: Agent reached maximum steps without finding a final answer."
            
        # Update memory with the exchange
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(final_answer)
                
        return PipelineResult(
            answer=Answer(text=final_answer, citations=list(set(citations)), source="agent"),
            path="agent",
        )

    async def run_stream(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming agent implementation yielding tokens and metadata."""
        # Initial metadata
        yield {
            "type": "metadata",
            "path": "agent",
            "confidence": 1.0, # Agents are "confident" by default in this flow
            "reasoning": "Autonomous agent loop started."
        }

        messages = self._memory.get_context_messages()
        messages.append({"role": "user", "content": query})
        
        max_steps = 5
        citations = []
        final_answer = ""

        for step in range(max_steps):
            self._progress_callback(f"Agent reasoning (step {step+1})...")
            
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.2,
                stream=True 
            )
            
            collected_content = ""
            tool_calls = []
            
            for chunk in response:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Handle content streaming
                if hasattr(delta, "content") and delta.content:
                    collected_content += delta.content
                    yield {"type": "token", "content": delta.content}
                
                # Handle tool call streaming (accumulate)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        # Correctly grow tool_calls list to accommodate tc.index
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

            # If no tool calls were found in the stream, we are done
            if not tool_calls:
                final_answer = collected_content
                break
            
            # Prepare messages for next turn (including the tool call from assistant)
            messages.append({
                "role": "assistant",
                "content": collected_content,
                "tool_calls": tool_calls
            })
            
            # Execute tools
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                
                yield {"type": "token", "content": f"\n[Invoking {func_name}...]\n"}
                # AWAIT THE ASYNC TOOL EXECUTION
                result_str = await self._execute_tool_async(func_name, func_args)
                
                if func_name == "web_browser":
                    try:
                        args_data = json.loads(func_args)
                        if "url" in args_data:
                            citations.append(args_data["url"])
                            yield {"type": "citations", "content": list(set(citations))}
                    except:
                        pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": func_name,
                    "content": result_str,
                })

        if not final_answer:
            final_answer = collected_content
            
        # Update memory
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(final_answer)
        
        # Final yield of metadata/citations just in case
        yield {"type": "citations", "content": list(set(citations))}

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._memory.clear()

    def clear_cache(self) -> None:
        """Clear cache adapter for compatibility."""
        pass
