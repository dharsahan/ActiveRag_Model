#!/usr/bin/env python3
"""Command-line entry point for the Active RAG system."""

from __future__ import annotations

import argparse
import logging
import sys

from active_rag.config import Config
from active_rag.pipeline import ActiveRAGPipeline


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
    if not config.openai_api_key:
        print(
            "Error: OPENAI_API_KEY is not set. "
            "Export it as an environment variable or add it to a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

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
