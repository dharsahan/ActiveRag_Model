#!/usr/bin/env python3
"""Enhanced command-line entry point for the Active RAG system.

Features:
- Beautiful Rich console output with colors and spinners
- Streaming responses for real-time output
- Conversation memory for follow-up questions
- Response caching for faster repeated queries
- Retry logic with exponential backoff
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import NoReturn

import httpx
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown

from active_rag.config import Config
from active_rag.agent import AgenticOrchestrator
from active_rag.pipeline import ActiveRAGPipeline, PipelineResult
from active_rag.console import (
    console,
    print_banner,
    print_result,
    print_error,
    print_success,
    print_info,
    print_warning,
    status_spinner,
)


def _check_llm_backend(config: Config) -> None:
    """Verify that the LLM backend is reachable before running queries."""
    base_url = config.ollama_base_url.rstrip("/")
    
    # Detect if using external API (NVIDIA, OpenAI, etc.) vs local Ollama
    is_external_api = any(
        host in base_url for host in [
            "nvidia.com", "openai.com", "anthropic.com", 
            "api.together.xyz", "api.groq.com"
        ]
    )
    
    if is_external_api:
        # For external APIs, check the /models endpoint which is standard OpenAI-compatible
        try:
            resp = httpx.get(f"{base_url}/models", timeout=10)
            # 401/403 means API is reachable but needs auth - that's okay, auth is in the client
            if resp.status_code in (401, 403):
                return  # API is reachable, auth will be handled by OpenAI client
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException):
            print_error(f"Cannot connect to API at {base_url}")
            print_info("Check your internet connection and API endpoint.")
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            # Some APIs return errors without auth, but the endpoint exists
            if exc.response.status_code in (401, 403, 404):
                return  # Assume API is reachable, let OpenAI client handle auth
            print_error(f"API returned HTTP {exc.response.status_code}")
            sys.exit(1)
    else:
        # Local Ollama health check
        health_url = base_url[:-3] if base_url.endswith("/v1") else base_url
        try:
            resp = httpx.get(f"{health_url}/", timeout=5)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException):
            print_error(f"Cannot connect to Ollama at {config.ollama_base_url}")
            console.print()
            console.print("[dim]Make sure Ollama is installed and running:[/dim]")
            console.print("[dim]  1. Install Ollama   → https://ollama.com[/dim]")
            console.print(f"[dim]  2. Pull a model     → ollama pull {config.model_name}[/dim]")
            console.print("[dim]  3. Start the server → ollama serve[/dim]")
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            print_error(f"Ollama returned HTTP {exc.response.status_code}")
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Active RAG – Refined Retrieval-Augmented Generation"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The question to answer. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming output."
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable response caching."
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Disable conversation memory."
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear the response cache and exit."
    )
    parser.add_argument(
        "--clear-memory", action="store_true", help="Clear conversation memory."
    )
    parser.add_argument(
        "--ingest", "-i", nargs="+",
        help="Ingest local files (PDF, TXT, MD, DOCX) into the vector store.",
    )
    parser.add_argument(
        "--legacy-pipeline", action="store_true",
        help="Use the fast static pipeline instead of the default Autonomous Agent.",
    )
    parser.add_argument(
        "--agent", action="store_true",
        help="Use the Autonomous Agent (ReAct) with tool-calling capabilities.",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use the Hybrid Vector-Graph RAG pipeline with intelligent routing.",
    )
    parser.add_argument(
        "--ultimate", action="store_true",
        help="Use the Ultimate Pipeline that auto-escalates through all knowledge sources and learns continuously.",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start the REST API server (FastAPI).",
    )
    
    db_group = parser.add_argument_group("Vector Store Management")
    db_group.add_argument("--db-stats", action="store_true", help="Show vector store statistics.")
    db_group.add_argument("--db-search", type=str, help="Search the vector store for a query.")
    db_group.add_argument("--db-clear", action="store_true", help="Clear all documents from vector store.")
    db_group.add_argument("--db-export", type=str, help="Export vector store contents to JSON file.")
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    config = Config()
    
    # Handle cache clear
    if args.clear_cache:
        from active_rag.cache import ResponseCache
        cache = ResponseCache(config)
        cache.clear()
        print_success("Cache cleared!")
        return

    # Handle Vector Store Management
    if args.db_stats or args.db_search or args.db_clear or args.db_export:
        from active_rag.vector_store import VectorStore
        store = VectorStore(config)
        
        if args.db_stats:
            count = store.count()
            print_info(f"Chunks in Graph: {count}")
            print_info(f"Vector Index: {config.vector_index_name}")
            return
            
        if args.db_search:
            from active_rag.console import console
            result = store.search(args.db_search)
            if result.found:
                for r in result.results:
                    console.print(f"\n[bold]{r.source_url}[/bold] (score: {r.score:.2f})")
                    console.print(f"[dim]{r.content[:200]}...[/dim]")
            else:
                print_warning("No matching documents found.")
            return
            
        if args.db_clear:
            store.clear()
            print_success("Vector store (Chunks) cleared!")
            return
            
        if args.db_export:
            import json
            all_data = store.get_all_documents()
            if all_data:
                with open(args.db_export, "w") as f:
                    json.dump(all_data, f, indent=2)
                print_success(f"Exported {len(all_data)} documents to {args.db_export}")
            else:
                print_warning("No documents to export.")
            return

    # Handle document ingestion (no LLM needed for vectors, but uses LLM for graph)
    if args.ingest:
        from active_rag.document_loader import DocumentLoader
        from active_rag.vector_store import VectorStore
        from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
        from active_rag.nlp_pipeline.relation_extractor import RelationExtractor
        from active_rag.schemas.entities import ContentDomain
        
        loader = DocumentLoader()
        store = VectorStore(config)
        extractor = EntityExtractor()
        rel_extractor = RelationExtractor(config)
        
        for filepath in args.ingest:
            try:
                with status_spinner(f"Ingesting {filepath}..."):
                    docs = loader.load(filepath)
                    for d in docs:
                        # 1. Add to Vector Store
                        chunk_ids = store.add_documents(
                            contents=[d.content],
                            source_urls=[d.source],
                        )
                        
                        # 2. Extract Entities
                        entities = extractor.extract_entities(d.content, ContentDomain.TECHNICAL)
                        for entity in entities[:10]:
                            try:
                                props = entity["properties"].copy()
                                props["source_file"] = d.source
                                props["source_type"] = "ingestion"
                                store._neo4j.create_entity(entity["label"], props)
                            except Exception:
                                continue
                        
                        # 3. Extract Dynamic Relationships (including Chunk -> Entity)
                        if entities:
                            cid = chunk_ids[0] if chunk_ids else None
                            relations = rel_extractor.extract_relations(d.content, entities, chunk_id=cid)
                            for rel in relations:
                                try:
                                    store._neo4j.create_relationship(
                                        subject_id=rel["subject_id"],
                                        subject_label=rel["subject_label"],
                                        predicate=rel["predicate"],
                                        object_id=rel["object_id"],
                                        object_label=rel["object_label"],
                                        properties=rel.get("properties", {})
                                    )
                                except Exception:
                                    continue
                                
                print_success(f"Ingested and dynamically enriched {len(docs)} chunk(s) from {filepath}")
            except (FileNotFoundError, ValueError) as e:
                print_error(str(e))
        return

    # Handle API server
    if args.serve:
        try:
            import uvicorn
            from active_rag.api import create_app
            print_info("Starting API server at http://0.0.0.0:8000")
            print_info("Docs at http://0.0.0.0:8000/docs")
            uvicorn.run(create_app(config), host="0.0.0.0", port=8000)
        except ImportError:
            print_error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    # Check backend connectivity
    with status_spinner("Checking LLM connection..."):
        _check_llm_backend(config)

    def progress_callback(msg: str) -> None:
        if args.verbose:
            print_info(msg)

    # Choose pipeline with streaming compatibility
    if args.legacy_pipeline:
        pipeline = ActiveRAGPipeline(
            config,
            enable_cache=not args.no_cache,
            enable_memory=not args.no_memory,
            progress_callback=progress_callback,
        )
    elif args.hybrid:
        from active_rag.hybrid_pipeline import HybridRAGPipeline
        pipeline = HybridRAGPipeline(config, progress_callback=progress_callback)
    elif args.ultimate:
        from active_rag.ultimate_pipeline import UltimateActiveRAGPipeline
        pipeline = UltimateActiveRAGPipeline(config, progress_callback=progress_callback)
        if not args.query:  # Only print this message for interactive mode
            print_info("🚀 Using Ultimate Pipeline - Auto-learning knowledge system!")
    elif args.agent:
        pipeline = AgenticOrchestrator(config, progress_callback=progress_callback)
        if not args.query:
            print_info("🤖 Using Autonomous Agent (ReAct)!")
    else:
        # Default to agentic orchestrator for the most interactive experience
        pipeline = AgenticOrchestrator(config, progress_callback=progress_callback)
        if not args.query:
            print_info("🤖 Using Autonomous Agent (ReAct)!")

    if args.clear_memory:
        pipeline.clear_memory()
        print_success("Conversation memory cleared!")
        if not args.query:
            return

    if args.query:
        explain = getattr(args, 'explain', False)
        _process_query(pipeline, args.query, stream=not args.no_stream, explain=explain)
    else:
        explain = getattr(args, 'explain', False)
        _interactive_loop(pipeline, stream=not args.no_stream, explain=explain)


def _process_query(
    pipeline: ActiveRAGPipeline,
    query: str,
    stream: bool = True,
    explain: bool = False,
) -> None:
    """Process a single query with optional streaming."""
    # Handle internal commands
    cmd = query.lower().strip()
    if cmd == "/stats":
        if hasattr(pipeline, 'get_knowledge_stats'):
            stats = pipeline.get_knowledge_stats()
            print_info("📊 Knowledge System Statistics:")
            for system, data in stats.items():
                console.print(f"  {system.title()}: {data}")
        else:
            print_warning("Stats not available for this pipeline")
        return
    
    if cmd == "/cache":
        if hasattr(pipeline, '_cache') and pipeline._cache:
            stats = pipeline._cache.stats()
            print_info(f"Cache: {stats['size']} entries, {stats['volume']} bytes")
        else:
            print_warning("Caching is disabled for this pipeline")
        return

    if cmd == "/clear":
        pipeline.clear_memory()
        print_success("Conversation memory cleared!")
        return

    if cmd == "/reset":
        if hasattr(pipeline, 'clear_database'):
            if pipeline.clear_database():
                print_success("Knowledge base completely wiped!")
            else:
                print_error("Failed to wipe knowledge base.")
        else:
            print_warning("Reset not available for this pipeline")
        return

    if cmd == "/health":
        import subprocess
        subprocess.run([sys.executable, "scripts/health_check.py"])
        return

    if cmd == "/dump":
        # Dump memory directly
        if isinstance(pipeline, AgenticOrchestrator):
            print_info("📂 Memory Dump:")
            result = pipeline._list_memory_tool.execute({})
            console.print(result, markup=False)
        else:
            print_warning("/dump is only available for the Agentic pipeline")
        return

    if stream:
        _process_query_stream(pipeline, query)
    else:
        _process_query_sync(pipeline, query, explain=explain)


def _process_query_sync(pipeline: ActiveRAGPipeline, query: str, explain: bool = False) -> None:
    """Process query without streaming."""
    with status_spinner("Thinking..."):
        if explain and hasattr(pipeline, 'run') and 'explain' in pipeline.run.__code__.co_varnames:
            result = pipeline.run(query, explain=True)
        else:
            result = pipeline.run(query)
    
    _display_result(result)

    # Display explanation if available
    if explain and result.diagnostics.get("explanation"):
        exp = result.diagnostics["explanation"]
        console.print("\n[bold cyan]─── Reasoning Explanation ───[/bold cyan]")
        console.print(exp.get("reasoning_text", ""))
        console.print(f"\n[dim]{exp.get('confidence_explanation', '')}[/dim]")
        if exp.get("path_visualization") and "No graph" not in exp["path_visualization"]:
            console.print(f"\n[bold]Path Diagram:[/bold]\n{exp['path_visualization']}")


def _process_query_stream(pipeline: ActiveRAGPipeline, query: str) -> None:
    """Process query with streaming output."""
    console.print()

    # Check if pipeline has async streaming (AgenticOrchestrator)
    if isinstance(pipeline, AgenticOrchestrator):
        # Handle async agent streaming
        _process_async_stream(pipeline, query)
        return

    # Handle regular streaming (Legacy/Hybrid pipelines)
    _process_sync_stream(pipeline, query)


def _process_async_stream(pipeline, query: str) -> None:
    """Handle async streaming from agentic pipeline."""
    import asyncio

    async def async_stream_handler():
        console.print()
        answer_text = ""

        with Live(Panel("", title="[bold green]Answer[/bold green]", border_style="green"),
                  console=console, refresh_per_second=10) as live:

            async for item in pipeline.run_stream(query):
                if isinstance(item, dict):
                    # Handle metadata and tokens from agent
                    if item.get("type") == "metadata":
                        path_labels = {
                            "agent": "[red]🤖 Autonomous Agent[/red]",
                            "direct": "[green]⚡ Direct Answer[/green]",
                        }
                        path = item.get("path", "agent")
                        console.print(f"[dim]Path: {path_labels.get(path, path)}[/dim]")
                    elif item.get("type") == "token":
                        content = item.get("content", "")
                        answer_text += content
                        live.update(Panel(
                            answer_text,
                            title="[bold green]Answer[/bold green]",
                            border_style="green",
                            padding=(1, 2),
                        ))
                    elif item.get("type") == "citations":
                        # We'll display these at the end
                        pass
                elif isinstance(item, str):
                    # Fallback for plain string tokens
                    answer_text += item
                    live.update(Panel(
                        answer_text,
                        title="[bold green]Answer[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    ))

        console.print()

    # Run the async handler
    try:
        asyncio.run(async_stream_handler())
    except Exception as e:
        console.print(f"[red]Streaming error: {e}[/red]")
        # Fall back to sync mode
        _process_query_sync(pipeline, query)


def _process_sync_stream(pipeline: ActiveRAGPipeline, query: str) -> None:
    """Handle sync streaming from legacy/hybrid pipelines."""
    # Track metadata from stream
    confidence = None
    path = "direct"
    indexed = 0
    answer_text = ""
    final_result = None

    # Stream tokens with live display
    with Live(Panel("", title="[bold green]Answer[/bold green]", border_style="green"),
              console=console, refresh_per_second=10) as live:

        for item in pipeline.run_stream(query):
            if isinstance(item, str):
                if item.startswith("__confidence__:"):
                    confidence = float(item.split(":")[1])
                    conf_style = "green" if confidence >= 0.7 else "yellow"
                    console.print(f"[dim]Confidence: [{conf_style}]{confidence:.0%}[/{conf_style}][/dim]")
                elif item.startswith("__path__:"):
                    path = item.split(":")[1]
                    path_labels = {
                        "direct": "[green]⚡ Direct Answer[/green]",
                        "rag_memory": "[blue]🧠 From Memory[/blue]",
                        "rag_web": "[magenta]🌐 Web Search[/magenta]",
                    }
                    console.print(f"[dim]Path: {path_labels.get(path, path)}[/dim]")
                elif item.startswith("__indexed__:"):
                    indexed = int(item.split(":")[1])
                    console.print(f"[dim]Indexed {indexed} web pages[/dim]")
                else:
                    # Token
                    answer_text += item
                    live.update(Panel(
                        Markdown(answer_text),
                        title="[bold green]Answer[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    ))
            elif isinstance(item, PipelineResult):
                final_result = item

    # Show citations if any
    if final_result and final_result.answer.citations:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for url in final_result.answer.citations:
            console.print(f"  [cyan]•[/cyan] [dim]{url}[/dim]")

    console.print()


def _display_result(result: PipelineResult) -> None:
    """Display a pipeline result with beautiful formatting."""
    print_result(
        answer_text=result.answer.text,
        path=result.path,
        confidence=result.confidence.confidence if result.confidence else None,
        citations=result.answer.citations,
        web_pages_indexed=result.web_pages_indexed,
        reasoning=result.confidence.reasoning if result.confidence else "",
    )
    
    if result.from_cache:
        print_info("(from cache)")


def _interactive_loop(pipeline: ActiveRAGPipeline, stream: bool = True, explain: bool = False) -> None:
    """Interactive REPL mode with conversation memory."""
    print_banner()
    console.print()
    console.print("[dim]Type your questions below. Commands:[/dim]")
    console.print("[dim]  /clear  - Clear conversation memory[/dim]")
    console.print("[dim]  /cache  - Show cache stats[/dim]")
    console.print("[dim]  /stats  - Show knowledge system statistics[/dim]")
    console.print("[dim]  /quit   - Exit[/dim]")
    console.print()
    
    while True:
        try:
            query = console.input("[bold cyan]>>> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            print_success("Goodbye!")
            break
        
        if not query:
            continue
        
        # Handle commands
        if query.lower() in {"/quit", "/exit", "/q", "quit", "exit", "q"}:
            print_success("Goodbye!")
            break
        
        if query.lower() == "/clear":
            pipeline.clear_memory()
            print_success("Conversation memory cleared!")
            continue

        if query.lower() == "/reset":
            if hasattr(pipeline, 'clear_database'):
                if pipeline.clear_database():
                    print_success("Knowledge base completely wiped!")
                else:
                    print_error("Failed to wipe knowledge base.")
            else:
                print_warning("Reset not available for this pipeline")
            continue

        if query.lower() == "/health":
            import subprocess
            subprocess.run([sys.executable, "scripts/health_check.py"])
            continue
        
        if query.lower() == "/cache":
            if pipeline._cache:
                stats = pipeline._cache.stats()
                print_info(f"Cache: {stats['size']} entries, {stats['volume']} bytes")
            else:
                print_warning("Caching is disabled")
            continue
        
        if query.lower() == "/help":
            console.print("[dim]Commands:[/dim]")
            console.print("[dim]  /clear  - Clear conversation memory[/dim]")
            console.print("[dim]  /reset  - Wipe entire knowledge base (Neo4j)[/dim]")
            console.print("[dim]  /health - Run system health check[/dim]")
            console.print("[dim]  /cache  - Show cache stats[/dim]")
            console.print("[dim]  /dump   - Dump all learned chunks[/dim]")
            console.print("[dim]  /stats  - Show knowledge system statistics[/dim]")
            console.print("[dim]  /quit   - Exit[/dim]")
            continue

        if query.lower() == "/stats":
            if hasattr(pipeline, 'get_knowledge_stats'):
                stats = pipeline.get_knowledge_stats()
                print_info("📊 Knowledge System Statistics:")
                for system, data in stats.items():
                    console.print(f"  {system.title()}: {data}")
            else:
                print_warning("Stats not available for this pipeline")
            continue
        
        # Process the query
        try:
            _process_query(pipeline, query, stream=stream)
        except KeyboardInterrupt:
            console.print()
            print_warning("Query cancelled")
            continue
        except Exception as e:
            print_error(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
