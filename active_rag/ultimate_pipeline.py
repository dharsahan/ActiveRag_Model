"""
Ultimate ActiveRAG Pipeline - Unified Knowledge System

This pipeline automatically escalates through all available knowledge sources:
1. Check LLM confidence → 2. Search vector store → 3. Query knowledge graph →
4. Search web + scrape → 5. Update knowledge graph → 6. Re-query enhanced graph → 7. Generate final answer

The system learns and grows with every query, building a richer knowledge base over time.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Generator, Dict, Any, List
from dataclasses import dataclass

from openai import OpenAI

from active_rag.config import Config
from active_rag.pipeline import PipelineResult
from active_rag.answer_generator import Answer
from active_rag.memory import ConversationMemory
from active_rag.confidence_checker import ConfidenceChecker
from active_rag.vector_store import VectorStore
from active_rag.web_search import WebSearcher
from active_rag.tools.web_browser import WebBrowserTool

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUpdateResult:
    """Result of knowledge update operations."""
    new_documents_added: int = 0
    new_entities_created: int = 0
    new_relationships_added: int = 0
    sources_discovered: List[str] = None

    def __post_init__(self):
        if self.sources_discovered is None:
            self.sources_discovered = []


class UltimateActiveRAGPipeline:
    """
    The ultimate RAG pipeline that automatically combines all knowledge sources
    and continuously learns from new information.
    """

    def __init__(
        self,
        config: Config | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config or Config()
        self._progress_callback = progress_callback or (lambda _: None)

        # Core LLM client
        self._client = OpenAI(
            base_url=self._config.ollama_base_url,
            api_key=self._config.api_key,
        )

        # Knowledge components
        self._confidence_checker = ConfidenceChecker(self._config)
        self._vector_store = VectorStore(self._config)
        self._web_searcher = WebSearcher(self._config)
        self._web_browser = WebBrowserTool(self._config)
        self._memory = ConversationMemory(self._config)

        # Graph operations (with fallback)
        self._graph_ops = None
        self._entity_extractor = None

        if self._config.enable_graph_features:
            try:
                from active_rag.knowledge_graph.neo4j_client import Neo4jClient
                from active_rag.knowledge_graph.graph_operations import GraphOperations
                from active_rag.nlp_pipeline.entity_extractor import EntityExtractor

                client = Neo4jClient(
                    self._config.neo4j_uri,
                    self._config.neo4j_username,
                    self._config.neo4j_password,
                )
                self._graph_ops = GraphOperations(client)
                self._entity_extractor = EntityExtractor()
                self._progress_callback("🕸️ Knowledge graph connected")
            except Exception as e:
                logger.warning(f"Graph backend unavailable: {e}")

    def run(self, query: str, memory: ConversationMemory | None = None) -> PipelineResult:
        """Execute the ultimate RAG pipeline with automatic knowledge expansion."""
        self._progress_callback("🧠 Starting ultimate knowledge search...")

        active_memory = memory or self._memory
        # Track what we discover
        knowledge_updates = KnowledgeUpdateResult()
        search_path = []
        final_context = []
        citations = []

        # Step 1: Check LLM Confidence
        self._progress_callback("📊 Checking AI confidence...")
        confidence = self._confidence_checker.check(query)
        search_path.append(f"confidence_check: {confidence.confidence:.0%}")

        # Step 2: Search Vector Store (always check local knowledge first)
        self._progress_callback("🗃️ Searching local knowledge...")
        vector_result = self._vector_store.search(query)

        vector_context = []
        if vector_result.found:
            search_path.append(f"vector_found: {len(vector_result.results)} docs")
            vector_context = [r.content for r in vector_result.results]
            citations.extend([r.source_url for r in vector_result.results if r.source_url])
            final_context.extend(vector_context)
        else:
            search_path.append("vector_empty")

        # Step 3: Query Knowledge Graph
        graph_context = []
        if self._graph_ops and self._entity_extractor:
            self._progress_callback("🕸️ Querying knowledge graph...")
            try:
                graph_result = self._graph_ops.multi_hop_query(query, max_hops=self._config.max_graph_hops)

                if graph_result.get("entities"):
                    search_path.append(f"graph_found: {len(graph_result['entities'])} entities")

                    # Extract meaningful context from graph
                    for entity in graph_result.get("entities", []):
                        if entity.get("name"):
                            entity_info = f"Entity: {entity['name']}"
                            if entity.get("labels"):
                                entity_info += f" (Type: {', '.join(entity['labels'])})"
                            graph_context.append(entity_info)

                    # Add reasoning paths
                    for path in graph_result.get("paths", []):
                        if path.get("reasoning_path"):
                            graph_context.append(f"Connection: {path['reasoning_path']}")

                    final_context.extend(graph_context)
                else:
                    search_path.append("graph_empty")
            except Exception as e:
                logger.warning(f"Graph query failed: {e}")
                search_path.append("graph_error")

        # Step 4: Determine if we need web search
        is_current_events = any(word in query.lower() for word in [
            'today', 'latest', 'current', 'news', 'recent', 'now', 'headlines', 'breaking',
            'todays', "today's", 'this morning', 'tonight', 'yesterday', 'update'
        ])

        is_simple_greeting = any(word in query.lower().strip() for word in [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'
        ]) and len(query.split()) <= 3

        # High confidence threshold for simple queries
        high_confidence = confidence.confidence >= 0.9
        has_some_context = len(final_context) >= 1
        has_sufficient_context = len(final_context) >= 2 and confidence.confidence >= 0.7

        # Smart web search logic:
        # - Always search for current events
        # - Skip web search for simple greetings with high confidence
        # - Skip web search if high confidence + some context
        # - Search if low confidence and insufficient context
        need_web_search = (
            is_current_events or
            (not is_simple_greeting and not high_confidence and not has_sufficient_context)
        )

        self._progress_callback(f"🔍 Analysis: context={len(final_context)}, confidence={confidence.confidence:.0%}, current_events={is_current_events}, greeting={is_simple_greeting}, need_web={need_web_search}")

        # Force web search for current events or insufficient context
        if need_web_search:
            self._progress_callback("🌐 Searching web for fresh information...")

            try:
                # Use direct web searcher instead of browser tool
                pages = self._web_searcher.search_and_scrape(query)

                if pages:
                    search_path.append(f"web_found: {len(pages)} pages")

                    # Extract content and sources
                    web_content = []
                    web_sources = []

                    for page in pages[:3]:  # Limit to top 3 results
                        if page.content and len(page.content.strip()) > 50:
                            web_content.append(page.content)
                            web_sources.append(page.url)

                    if web_content:
                        citations.extend(web_sources)

                        # Step 5: Update Knowledge Systems
                        self._progress_callback("💾 Updating knowledge systems...")

                        # Add to vector store
                        for i, content in enumerate(web_content):
                            source_url = web_sources[i] if i < len(web_sources) else f"web_search_{i}"

                            try:
                                self._vector_store.add_documents([content], [source_url])
                                knowledge_updates.new_documents_added += 1
                            except Exception as e:
                                logger.warning(f"Failed to add document to vector store: {e}")

                        # Extract entities and update graph
                        if self._graph_ops and self._entity_extractor and web_content:
                            for content in web_content[:2]:  # Process top 2 for entity extraction
                                try:
                                    from active_rag.schemas.entities import ContentDomain
                                    entities = self._entity_extractor.extract_entities(content, ContentDomain.MIXED_WEB)
                                    for entity in entities[:5]:  # Limit entities per document
                                        try:
                                            # Create entity in graph
                                            entity_props = entity["properties"].copy()
                                            # Add source information
                                            entity_props["discovered_from"] = "web_search"
                                            entity_props["query"] = query

                                            self._graph_ops.client.create_entity(
                                                entity["label"],
                                                entity_props
                                            )
                                            knowledge_updates.new_entities_created += 1
                                        except Exception as entity_e:
                                            logger.debug(f"Failed to create entity: {entity_e}")
                                except Exception as e:
                                    logger.warning(f"Failed to extract entities: {e}")

                        # Add web context
                        final_context.extend(web_content)
                        knowledge_updates.sources_discovered = web_sources

                    else:
                        search_path.append("web_content_empty")
                else:
                    search_path.append("web_no_results")

            except Exception as e:
                logger.warning(f"Web search failed: {e}")
                search_path.append(f"web_error: {str(e)[:50]}")

        # Step 6: Re-query enhanced systems (if we added new data)
        if knowledge_updates.new_documents_added > 0:
            self._progress_callback("🔄 Re-querying enhanced knowledge...")

            # Re-search vector store with new data
            enhanced_vector_result = self._vector_store.search(query)
            if enhanced_vector_result.found:
                new_results = [r.content for r in enhanced_vector_result.results
                              if r.content not in final_context]
                final_context.extend(new_results)
                search_path.append(f"enhanced_vector: +{len(new_results)} new")

        # Step 7: Generate Final Answer
        self._progress_callback("✨ Generating comprehensive answer...")

        # Prepare context
        context_text = "\n\n".join(final_context) if final_context else ""

        # Build conversation messages
        messages = active_memory.get_context_messages()

        system_message = {
            "role": "system",
            "content": (
                "You are an advanced AI assistant with access to multiple knowledge sources. "
                "Use the provided context to give accurate, comprehensive answers. "
                "If you used web sources, mention that the information is current. "
                "Format your response clearly with markdown. "
                "Be transparent about your sources and reasoning."
            )
        }

        if not messages or messages[0].get("role") != "system":
            messages.insert(0, system_message)
        else:
            messages[0] = system_message

        # Prepare user message with context
        user_content = query
        if context_text.strip():
            user_content = f"Context from knowledge sources:\n{context_text}\n\nQuestion: {query}"

        messages.append({"role": "user", "content": user_content})

        # Generate answer
        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                temperature=0.2,
            )
            if response.choices:
                answer_text = response.choices[0].message.content or "Unable to generate answer."
            else:
                answer_text = "No response generated from LLM."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer_text = f"Error generating answer: {e}"

        # Update memory
        active_memory.add_user_message(query)
        active_memory.add_assistant_message(answer_text)

        # Determine final path
        path_components = []
        if "confidence_check" in search_path:
            path_components.append("confidence")
        if any("vector" in step for step in search_path):
            path_components.append("vector")
        if any("graph" in step for step in search_path):
            path_components.append("graph")
        if any("web" in step for step in search_path):
            path_components.append("web")

        final_path = "ultimate_" + "_".join(path_components)

        # Create result with diagnostics
        diagnostics = {
            "search_path": search_path,
            "knowledge_updates": {
                "documents_added": knowledge_updates.new_documents_added,
                "entities_created": knowledge_updates.new_entities_created,
                "sources_discovered": knowledge_updates.sources_discovered,
            },
            "context_sources": len(final_context),
            "total_citations": len(set(citations)),
        }

        return PipelineResult(
            answer=Answer(text=answer_text, citations=list(set(citations)), source=final_path),
            path=final_path,
            diagnostics=diagnostics,
        )

    def run_stream(self, query: str, memory: ConversationMemory | None = None) -> Generator[str | PipelineResult, None, None]:
        """Streaming version of the ultimate pipeline."""
        yield "🧠 Starting ultimate knowledge search..."

        active_memory = memory or self._memory
        # Track what we discover
        knowledge_updates = KnowledgeUpdateResult()
        search_path = []
        final_context = []
        citations = []

        try:
            # Step 1: Check LLM Confidence
            yield "📊 Checking AI confidence..."
            confidence = self._confidence_checker.check(query)
            search_path.append(f"confidence_check: {confidence.confidence:.0%}")
            yield f"Confidence: {confidence.confidence:.0%}"

            # Step 2: Search Vector Store
            yield "🗃️ Searching local knowledge..."
            vector_result = self._vector_store.search(query)

            vector_context = []
            if vector_result.found:
                search_path.append(f"vector_found: {len(vector_result.results)} docs")
                vector_context = [r.content for r in vector_result.results]
                citations.extend([r.source_url for r in vector_result.results if r.source_url])
                final_context.extend(vector_context)
                yield f"Found {len(vector_result.results)} local documents"
            else:
                search_path.append("vector_empty")
                yield "No local documents found"

            # Step 3: Query Knowledge Graph
            graph_context = []
            if self._graph_ops and self._entity_extractor:
                yield "🕸️ Querying knowledge graph..."
                try:
                    graph_result = self._graph_ops.multi_hop_query(query, max_hops=self._config.max_graph_hops)

                    if graph_result.get("entities"):
                        search_path.append(f"graph_found: {len(graph_result['entities'])} entities")
                        yield f"Found {len(graph_result['entities'])} graph entities"

                        # Extract meaningful context from graph
                        for entity in graph_result.get("entities", []):
                            if entity.get("name"):
                                entity_info = f"Entity: {entity['name']}"
                                if entity.get("labels"):
                                    entity_info += f" (Type: {', '.join(entity['labels'])})"
                                graph_context.append(entity_info)

                        # Add reasoning paths
                        for path in graph_result.get("paths", []):
                            if path.get("reasoning_path"):
                                graph_context.append(f"Connection: {path['reasoning_path']}")

                        final_context.extend(graph_context)
                    else:
                        search_path.append("graph_empty")
                        yield "No graph connections found"
                except Exception as e:
                    logger.warning(f"Graph query failed: {e}")
                    search_path.append("graph_error")
                    yield f"Graph query error: {str(e)[:50]}"

            # Step 4: Determine if we need web search
            is_current_events = any(word in query.lower() for word in [
                'today', 'latest', 'current', 'news', 'recent', 'now', 'headlines', 'breaking',
                'todays', "today's", 'this morning', 'tonight', 'yesterday', 'update'
            ])

            is_simple_greeting = any(word in query.lower().strip() for word in [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'
            ]) and len(query.split()) <= 3

            # High confidence threshold for simple queries
            high_confidence = confidence.confidence >= 0.9
            has_some_context = len(final_context) >= 1
            has_sufficient_context = len(final_context) >= 2 and confidence.confidence >= 0.7

            # Smart web search logic:
            need_web_search = (
                is_current_events or
                (not is_simple_greeting and not high_confidence and not has_sufficient_context)
            )

            yield f"Analysis: context={len(final_context)}, confidence={confidence.confidence:.0%}, current_events={is_current_events}, greeting={is_simple_greeting}, need_web={need_web_search}"

            # Force web search for current events or insufficient context
            if need_web_search:
                yield "🌐 Searching web for fresh information..."

                try:
                    # Use direct web searcher
                    pages = self._web_searcher.search_and_scrape(query)

                    if pages:
                        search_path.append(f"web_found: {len(pages)} pages")
                        yield f"Found {len(pages)} web pages"

                        # Extract content and sources
                        web_content = []
                        web_sources = []

                        for page in pages[:3]:  # Limit to top 3 results
                            if page.content and len(page.content.strip()) > 50:
                                web_content.append(page.content)
                                web_sources.append(page.url)

                        if web_content:
                            citations.extend(web_sources)
                            yield f"💾 Adding {len(web_content)} documents to knowledge base..."

                            # Add to vector store
                            for i, content in enumerate(web_content):
                                source_url = web_sources[i] if i < len(web_sources) else f"web_search_{i}"
                                try:
                                    self._vector_store.add_documents([content], [source_url])
                                    knowledge_updates.new_documents_added += 1
                                except Exception as e:
                                    logger.warning(f"Failed to add document to vector store: {e}")

                            # Add web context
                            final_context.extend(web_content)
                            knowledge_updates.sources_discovered = web_sources
                            yield f"Knowledge base expanded with {knowledge_updates.new_documents_added} new documents"

                        else:
                            search_path.append("web_content_empty")
                            yield "Web content was empty or too short"
                    else:
                        search_path.append("web_no_results")
                        yield "No web results found"

                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    search_path.append(f"web_error: {str(e)[:50]}")
                    yield f"Web search error: {str(e)[:100]}"

            # Generate final answer
            yield "✨ Generating comprehensive answer..."

            # Prepare context
            context_text = "\n\n".join(final_context) if final_context else ""

            # Build conversation messages
            messages = active_memory.get_context_messages()

            system_message = {
                "role": "system",
                "content": (
                    "You are an advanced AI assistant with access to multiple knowledge sources. "
                    "Use the provided context to give accurate, comprehensive answers. "
                    "If you used web sources, mention that the information is current. "
                    "Format your response clearly with markdown. "
                    "Be transparent about your sources and reasoning."
                )
            }

            if not messages or messages[0].get("role") != "system":
                messages.insert(0, system_message)
            else:
                messages[0] = system_message

            # Prepare user message with context
            user_content = query
            if context_text.strip():
                user_content = f"Context from knowledge sources:\n{context_text}\n\nQuestion: {query}"

            messages.append({"role": "user", "content": user_content})

            # Generate answer
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model_name,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                )

                answer_text = ""
                for chunk in response:
                    if not chunk.choices:
                        continue
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        answer_text += token
                        yield token

            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer_text = f"Error generating answer: {e}"
                yield answer_text

            # Update memory
            active_memory.add_user_message(query)
            active_memory.add_assistant_message(answer_text)

            # Determine final path
            path_components = []
            if any("confidence" in step for step in search_path):
                path_components.append("confidence")
            if any("vector" in step for step in search_path):
                path_components.append("vector")
            if any("graph" in step for step in search_path):
                path_components.append("graph")
            if any("web" in step for step in search_path):
                path_components.append("web")

            final_path = "ultimate_" + "_".join(path_components)

            # Create result with diagnostics
            diagnostics = {
                "search_path": search_path,
                "knowledge_updates": {
                    "documents_added": knowledge_updates.new_documents_added,
                    "entities_created": knowledge_updates.new_entities_created,
                    "sources_discovered": knowledge_updates.sources_discovered,
                },
                "context_sources": len(final_context),
                "total_citations": len(set(citations)),
            }

            # Yield final result
            yield PipelineResult(
                answer=Answer(text=answer_text, citations=list(set(citations)), source=final_path),
                path=final_path,
                diagnostics=diagnostics,
            )

        except Exception as e:
            yield f"Error in ultimate pipeline: {e}"
            # Fallback to basic answer
            yield PipelineResult(
                answer=Answer(text=f"Error occurred: {e}", citations=[], source="error"),
                path="error",
                diagnostics={"error": str(e)},
            )

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._memory.clear()

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge systems."""
        stats = {
            "vector_store": {
                "documents": self._vector_store.count()
            }
        }

        if self._graph_ops:
            try:
                with self._graph_ops.client._driver.session() as session:
                    # Count nodes
                    node_result = session.run("MATCH (n) RETURN count(n) as count")
                    node_count = node_result.single()["count"]

                    # Count relationships
                    rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                    rel_count = rel_result.single()["count"]

                    stats["knowledge_graph"] = {
                        "nodes": node_count,
                        "relationships": rel_count
                    }
            except Exception as e:
                stats["knowledge_graph"] = {"error": str(e)}

        return stats