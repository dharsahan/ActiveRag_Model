"""Autonomous Agent Loop implementing ReAct / Tool Calling."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, Generator, AsyncGenerator, Any

from openai import OpenAI

from active_rag.config import Config
from active_rag.pipeline import PipelineResult
from active_rag.answer_generator import Answer
from active_rag.memory import ConversationMemory
from active_rag.tools.calculator import TOOL_SCHEMA as CALC_SCHEMA, execute as execute_calc
from active_rag.tools.web_browser import WebBrowserTool
from active_rag.tools.vector_database import VectorDatabaseTool
from active_rag.tools.graph_query import GraphQueryTool
from active_rag.tools.store_memory import StoreMemoryTool
from active_rag.tools.list_memory import ListMemoryTool

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
        
        # Initialize tools with shared resources
        from active_rag.vector_store import VectorStore
        shared_vector_store = VectorStore(self._config)
        
        self._vector_tool = VectorDatabaseTool(self._config)
        self._vector_tool._store = shared_vector_store # Inject shared store
        
        self._web_tool = WebBrowserTool(self._config, vector_store=shared_vector_store)
        self._graph_tool = GraphQueryTool(self._config)
        self._store_memory_tool = StoreMemoryTool(self._config, vector_store=shared_vector_store)
        self._list_memory_tool = ListMemoryTool(self._config, vector_store=shared_vector_store)
        self._memory = ConversationMemory(self._config)
        self._cache = None # Agent doesn't use internal cache yet
        
        self.tools_schema = [
            CALC_SCHEMA,
            self._web_tool.schema,
            self._vector_tool.schema,
            self._graph_tool.schema,
            self._store_memory_tool.schema,
            self._list_memory_tool.schema,
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
        elif name == "graph_query":
            return self._graph_tool.execute(parsed_args)
        elif name == "store_memory":
            return self._store_memory_tool.execute(parsed_args)
        elif name == "list_memory":
            return self._list_memory_tool.execute(parsed_args)
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
        elif name == "graph_query":
            return await self._graph_tool.execute_async(parsed_args)
        elif name == "store_memory":
            # store_memory is currently sync in its implementation but we'll call it async-ready
            return self._store_memory_tool.execute(parsed_args)
        elif name == "list_memory":
            return self._list_memory_tool.execute(parsed_args)
        else:
            return f"Error: Unknown tool '{name}'"

    def run(self, query: str, max_steps: int = 5) -> PipelineResult:
        """Run the agent loop synchronously until a final answer is produced."""
        # Use windowed memory history
        messages = self._memory.get_context_messages()

        # Enhanced system message for better formatting
        enhanced_system_msg = {
            "role": "system",
            "content": (
                "You are an advanced 'Refined Active RAG' Agent. You have access to a suite of tools "
                "to provide accurate, up-to-date, and well-reasoned answers.\n\n"
                "YOUR TOOLSET:\n"
                "1. **web_browser**: Search and scrape the live internet. Use this for current events, fresh facts, or when your internal knowledge is insufficient. Everything you browse is automatically indexed.\n"
                "2. **query_memory**: Search your long-term Vector Database (Neo4j) for previously learned facts and documents.\n"
                "3. **graph_query**: Perform multi-hop reasoning over your Knowledge Graph (Neo4j). Use this to explore complex relationships between entities.\n"
                "4. **store_memory**: Explicitly memorize a fact if the user asks you to 'remember' something or if you find a critical piece of info you want to save permanently.\n"
                "5. **list_memory**: List everything currently in your memory if requested.\n"
                "6. **calculator**: Perform precise mathematical calculations.\n\n"
                "STRATEGY:\n"
                "- Always prioritize factual accuracy. If unsure, use a tool.\n"
                "- If a query involves multiple steps or complex relationships, use the graph_query.\n"
                "- Format your responses beautifully using markdown (headers, bolding, bullet points).\n"
                "- Be transparent: if you use a tool, you can briefly mention it if helpful."
            )
        }

        # Insert enhanced system message if not already present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, enhanced_system_msg)
        else:
            messages[0] = enhanced_system_msg

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

        # Post-process the final answer to ensure good formatting
        final_answer = self._post_process_response(final_answer)

        # Update memory with the exchange
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(final_answer)

        # Continuous Learning: Index the interaction
        self._index_interaction(query, final_answer)

        return PipelineResult(
            answer=Answer(text=final_answer, citations=list(set(citations)), source="agent"),
            path="agent",
        )

    async def run_async(self, query: str, max_steps: int = 5) -> PipelineResult:
        """Run the agent loop asynchronously until a final answer is produced."""
        # Use windowed memory history
        messages = self._memory.get_context_messages()

        # Enhanced system message for better formatting
        enhanced_system_msg = {
            "role": "system",
            "content": (
                "You are an advanced 'Refined Active RAG' Agent. You have access to a suite of tools "
                "to provide accurate, up-to-date, and well-reasoned answers.\n\n"
                "YOUR TOOLSET:\n"
                "1. **web_browser**: Search and scrape the live internet. Use this for current events, fresh facts, or when your internal knowledge is insufficient. Everything you browse is automatically indexed.\n"
                "2. **query_memory**: Search your long-term Vector Database (Neo4j) for previously learned facts and documents.\n"
                "3. **graph_query**: Perform multi-hop reasoning over your Knowledge Graph (Neo4j). Use this to explore complex relationships between entities.\n"
                "4. **store_memory**: Explicitly memorize a fact if the user asks you to 'remember' something or if you find a critical piece of info you want to save permanently.\n"
                "5. **list_memory**: List everything currently in your memory if requested.\n"
                "6. **calculator**: Perform precise mathematical calculations.\n\n"
                "STRATEGY:\n"
                "- Always prioritize factual accuracy. If unsure, use a tool.\n"
                "- If a query involves multiple steps or complex relationships, use the graph_query.\n"
                "- Format your responses beautifully using markdown (headers, bolding, bullet points).\n"
                "- Be transparent: if you use a tool, you can briefly mention it if helpful."
            )
        }

        # Insert enhanced system message if not already present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, enhanced_system_msg)
        else:
            messages[0] = enhanced_system_msg

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

                # Use async tool execution
                result_str = await self._execute_tool_async(func_name, func_args)

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

        # Post-process the final answer to ensure good formatting
        final_answer = self._post_process_response(final_answer)

        # Update memory with the exchange
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(final_answer)

        # Continuous Learning: Index the interaction
        self._index_interaction(query, final_answer)

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
        # Enhanced system message for better formatting
        enhanced_system_msg = {
            "role": "system",
            "content": (
                "You are an advanced 'Refined Active RAG' Agent. You have access to a suite of tools "
                "to provide accurate, up-to-date, and well-reasoned answers.\n\n"
                "YOUR TOOLSET:\n"
                "1. **web_browser**: Search and scrape the live internet. Use this for current events, fresh facts, or when your internal knowledge is insufficient. Everything you browse is automatically indexed.\n"
                "2. **query_memory**: Search your long-term Vector Database (Neo4j) for previously learned facts and documents.\n"
                "3. **graph_query**: Perform multi-hop reasoning over your Knowledge Graph (Neo4j). Use this to explore complex relationships between entities.\n"
                "4. **store_memory**: Explicitly memorize a fact if the user asks you to 'remember' something or if you find a critical piece of info you want to save permanently.\n"
                "5. **list_memory**: List everything currently in your memory if requested.\n"
                "6. **calculator**: Perform precise mathematical calculations.\n\n"
                "STRATEGY:\n"
                "- Always prioritize factual accuracy. If unsure, use a tool.\n"
                "- If a query involves multiple steps or complex relationships, use the graph_query.\n"
                "- Format your responses beautifully using markdown (headers, bolding, bullet points).\n"
                "- Be transparent: if you use a tool, you can briefly mention it if helpful."
            )
        }

        # Insert enhanced system message if not already present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, enhanced_system_msg)
        else:
            messages[0] = enhanced_system_msg

        messages.append({"role": "user", "content": query})

        max_steps = 5
        citations = []
        final_answer = ""
        accumulated_tokens = []  # Buffer for token accumulation

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

                # Handle content streaming with buffering for better formatting
                if hasattr(delta, "content") and delta.content:
                    collected_content += delta.content
                    accumulated_tokens.append(delta.content)

                    # Yield tokens in chunks to preserve formatting
                    combined_buffer = ''.join(accumulated_tokens)

                    # Check if we have complete formatting elements
                    if any(marker in combined_buffer for marker in ['\n\n', '**', '##', '- ', '* ', '1. ']):
                        # Yield the buffered content and clear buffer
                        yield {"type": "token", "content": ''.join(accumulated_tokens)}
                        accumulated_tokens = []
                    elif len(accumulated_tokens) > 10:  # Fallback: yield after 10 tokens
                        yield {"type": "token", "content": ''.join(accumulated_tokens)}
                        accumulated_tokens = []

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

            # Yield any remaining tokens in buffer
            if accumulated_tokens:
                yield {"type": "token", "content": ''.join(accumulated_tokens)}
                accumulated_tokens = []

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

            # Execute tools with better formatting
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]

                yield {"type": "token", "content": f"\n\n**[Invoking {func_name}...]**\n\n"}
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

        # Post-process the final answer to ensure good formatting
        final_answer = self._post_process_response(final_answer)

        # Update memory
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(final_answer)

        # Continuous Learning: Index the interaction
        self._index_interaction(query, final_answer)

        # Final yield of metadata/citations just in case
        yield {"type": "citations", "content": list(set(citations))}

    def _post_process_response(self, text: str) -> str:
        """Post-process response text to ensure good formatting."""
        if not text:
            return text

        # Fix common formatting issues
        text = text.strip()

        # Ensure proper spacing around headers
        text = re.sub(r'(?<!^)(\n#{1,6}\s)', r'\n\n\1', text, flags=re.MULTILINE)
        text = re.sub(r'(#{1,6}.*?)(\n)(?!\n)', r'\1\n\n', text)

        # Fix bullet points formatting
        text = re.sub(r'\n([*-])\s*([^\n]+)', r'\n\1 \2', text)

        # Fix numbered lists
        text = re.sub(r'\n(\d+\.)\s*([^\n]+)', r'\n\1 \2', text)

        # Clean up multiple newlines but preserve intentional breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Ensure proper spacing after periods in lists
        text = re.sub(r'\.([A-Z])', r'. \1', text)

        return text.strip()

    def _index_interaction(self, query: str, answer: str):
        """Index the current interaction into the knowledge base (Continuous Learning)."""
        if not query or not answer or len(answer) < 10:
            return

        interaction_text = f"User Query: {query}\nAssistant Answer: {answer}"
        
        # Use the shared neo4j client to avoid creating new connections
        neo4j_client = self._vector_tool._store._neo4j

        # 1. Index into Vector Store (Neo4j)
        chunk_ids = []
        try:
            chunk_ids = self._vector_tool._store.add_documents(
                contents=[interaction_text],
                source_urls=["Chat History"]
            )
            # Add metadata if possible (VectorStore needs support for more meta)
            logger.info(f"Interaction indexed into vector store (ID: {chunk_ids}).")
        except Exception as e:
            logger.warning(f"Failed to index interaction: {e}")

        # 2. Extract entities and update Graph (Neo4j)
        extracted_entities = []
        try:
            from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
            from active_rag.schemas.entities import ContentDomain
            
            extractor = EntityExtractor()
            extracted_entities = extractor.extract_entities(interaction_text, ContentDomain.MIXED_WEB)
            
            for entity in extracted_entities[:5]: # Limit to top 5 entities per interaction
                try:
                    props = entity["properties"].copy()
                    props["source"] = "chat_interaction"
                    props["source_type"] = "chat"
                    props["timestamp"] = time.time()
                    neo4j_client.create_entity(entity["label"], props)
                    
                    # Link Chunk to Entity (MENTIONS)
                    if chunk_ids:
                        for cid in chunk_ids:
                            neo4j_client.create_relationship(
                                subject_id=cid,
                                subject_label="Chunk",
                                predicate="MENTIONS",
                                object_id=props["id"],
                                object_label=entity["label"],
                                properties={"context": "chat_history"}
                            )
                except Exception:
                    continue
            logger.info(f"Knowledge graph enriched with {len(extracted_entities[:5])} entities from chat.")
        except Exception as e:
            logger.warning(f"Failed to update entities from chat: {e}")

        # 3. Extract and create relationships between entities
        if extracted_entities and len(extracted_entities) >= 2:
            try:
                from active_rag.nlp_pipeline.relation_extractor import RelationExtractor
                rel_extractor = RelationExtractor(self._config)
                relations = rel_extractor.extract_relations(interaction_text, extracted_entities)
                
                for rel in relations:
                    try:
                        neo4j_client.create_relationship(
                            subject_id=rel["subject_id"],
                            subject_label=rel["subject_label"],
                            predicate=rel["predicate"],
                            object_id=rel["object_id"],
                            object_label=rel["object_label"],
                            properties=rel.get("properties", {})
                        )
                    except Exception:
                        continue
                logger.info(f"Extracted {len(relations)} new relationships from chat.")
            except Exception as e:
                logger.warning(f"Failed to extract relationships from chat: {e}")

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._memory.clear()

    def clear_cache(self) -> None:
        """Clear cache adapter for compatibility."""
        pass

    def clear_database(self) -> bool:
        """Completely wipe the Neo4j knowledge base (Chunks, Entities, Relations)."""
        return self._vector_tool._store._neo4j.clear_all_data()

    def get_knowledge_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge systems."""
        graph_stats = self._graph_tool.get_stats()
        
        # Format for display in main.py loop
        return {
            "vector_store": f"{self._vector_tool.count()} chunks",
            "knowledge_graph": f"{graph_stats.get('total_nodes', 0)} nodes, {graph_stats.get('total_relationships', 0)} relations"
        }
