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
        "--agent", action="store_true",
        help="Launch the Autonomous Agent (Tool-Calling ReAct loop) instead of the static pipeline.",
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
            count = store._collection.count()
            print_info(f"Documents: {count}")
            print_info(f"Collection: {config.collection_name}")
            print_info(f"Persist dir: {config.chroma_persist_dir}")
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
            store._client.delete_collection(config.collection_name)
            print_success("Vector store cleared!")
            return
            
        if args.db_export:
            import json
            all_data = store._collection.get(include=["documents", "metadatas"])
            export = []
            if all_data and all_data.get("documents"):
                for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                    export.append({"content": doc, "metadata": meta})
            with open(args.db_export, "w") as f:
                json.dump(export, f, indent=2)
            print_success(f"Exported {len(export)} documents to {args.db_export}")
            return

    # Handle document ingestion (no LLM needed)
    if args.ingest:
        from active_rag.document_loader import DocumentLoader
        from active_rag.vector_store import VectorStore
        loader = DocumentLoader()
        store = VectorStore(config)
        for filepath in args.ingest:
            try:
                with status_spinner(f"Ingesting {filepath}..."):
                    docs = loader.load(filepath)
                    store.add_documents(
                        contents=[d.content for d in docs],
                        source_urls=[d.source for d in docs],
                    )
                print_success(f"Ingested {len(docs)} chunk(s) from {filepath}")
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

    # Agent Execution Mode
    if args.agent:
        from active_rag.agent import AgenticOrchestrator
        from active_rag.console import console
        agent = AgenticOrchestrator(config, progress_callback=progress_callback)
        
        if args.query:
            result = agent.run("You are a helpful AI Assistant with access to tools.", args.query)
            console.print(f"\n[green]Agent final answer:[/green] {result}")
        else:
            print_info("Starting Agent Interactive Mode! (Type 'exit' or 'quit' to end)")
            from rich.prompt import Prompt
            while True:
                try:
                    user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                    if not user_input.strip():
                        continue
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        print_success("Goodbye!")
                        break
                    
                    result = agent.run("You are a helpful AI Assistant with access to tools.", user_input)
                    console.print(f"\n[bold magenta]Agent:[/bold magenta] {result}")
                    
                except KeyboardInterrupt:
                    break
        return

    # Static Pipeline Execution Mode
    pipeline = ActiveRAGPipeline(
        config,
        enable_cache=not args.no_cache,
        enable_memory=not args.no_memory,
        progress_callback=progress_callback,
    )

    if args.clear_memory:
        pipeline.clear_memory()
        print_success("Conversation memory cleared!")
        if not args.query:
            return

    if args.query:
        _process_query(pipeline, args.query, stream=not args.no_stream)
    else:
        _interactive_loop(pipeline, stream=not args.no_stream)


def _process_query(
    pipeline: ActiveRAGPipeline,
    query: str,
    stream: bool = True,
) -> None:
    """Process a single query with optional streaming."""
    if stream:
        _process_query_stream(pipeline, query)
    else:
        _process_query_sync(pipeline, query)


def _process_query_sync(pipeline: ActiveRAGPipeline, query: str) -> None:
    """Process query without streaming."""
    with status_spinner("Thinking..."):
        result = pipeline.run(query)
    
    _display_result(result)


def _process_query_stream(pipeline: ActiveRAGPipeline, query: str) -> None:
    """Process query with streaming output."""
    console.print()
    
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


def _interactive_loop(pipeline: ActiveRAGPipeline, stream: bool = True) -> None:
    """Interactive REPL mode with conversation memory."""
    print_banner()
    console.print()
    console.print("[dim]Type your questions below. Commands:[/dim]")
    console.print("[dim]  /clear  - Clear conversation memory[/dim]")
    console.print("[dim]  /cache  - Show cache stats[/dim]")
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
            console.print("[dim]  /cache  - Show cache stats[/dim]")
            console.print("[dim]  /quit   - Exit[/dim]")
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
