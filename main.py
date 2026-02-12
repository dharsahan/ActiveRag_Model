#!/usr/bin/env python3
"""Command-line entry point for the Active RAG system."""

from __future__ import annotations

import argparse
import logging
import sys

import httpx

from active_rag.config import Config
from active_rag.pipeline import ActiveRAGPipeline


def _check_ollama(config: Config) -> None:
    """Verify that the Ollama server is reachable before running queries."""
    # Strip the /v1 suffix to hit the Ollama root health endpoint.
    base = config.ollama_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        resp = httpx.get(f"{base}/", timeout=5)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        print(
            "Error: Cannot connect to Ollama at "
            f"{config.ollama_base_url}\n\n"
            "Make sure Ollama is installed and running:\n"
            "  1. Install Ollama   → https://ollama.com\n"
            f"  2. Pull a model     → ollama pull {config.model_name}\n"
            "  3. Start the server → ollama serve\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(
            f"Error: Ollama returned HTTP {exc.response.status_code} "
            f"at {config.ollama_base_url}",
            file=sys.stderr,
        )
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    config = Config()
    _check_ollama(config)

    pipeline = ActiveRAGPipeline(config)

    if args.query:
        _process_query(pipeline, args.query)
    else:
        _interactive_loop(pipeline)


def _process_query(pipeline: ActiveRAGPipeline, query: str) -> None:
    result = pipeline.run(query)
    print(f"\n{'=' * 60}")
    print(f"Path taken : {result.path}")
    if result.confidence:
        print(f"Confidence : {result.confidence.confidence:.2f}")
    if result.web_pages_indexed:
        print(f"Pages indexed: {result.web_pages_indexed}")
    print(f"{'=' * 60}")
    print(f"\n{result.answer.text}\n")
    if result.answer.citations:
        print("Citations:")
        for url in result.answer.citations:
            print(f"  • {url}")
    print()


def _interactive_loop(pipeline: ActiveRAGPipeline) -> None:
    print("Active RAG – interactive mode (type 'quit' to exit)\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        _process_query(pipeline, query)


if __name__ == "__main__":
    main()
