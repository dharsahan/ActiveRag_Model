#!/usr/bin/env python3
"""Test script to verify formatting improvements."""

import asyncio
from active_rag.config import Config
from active_rag.agent import AgenticOrchestrator
from active_rag.answer_generator import AnswerGenerator

def test_post_processing():
    """Test the post-processing functionality."""
    config = Config()
    generator = AnswerGenerator(config)

    # Test text with various formatting issues
    test_text = """
This is a test response with formatting issues.


Here are some bullet points:
-Point one
- Point two
*Another bullet
*   Spaced bullet

Here are numbered items:
1.First item
2.  Second item

## This is a header
This text follows a header.

Another paragraph.Multiple sentences.No spacing.

***Bold text*** and **more bold**

    """

    processed = generator._post_process_response(test_text)
    print("Original text:")
    print(repr(test_text))
    print("\nProcessed text:")
    print(repr(processed))
    print("\nFormatted output:")
    print(processed)

async def test_agent_formatting():
    """Test agent with a simple query."""
    config = Config()

    # Create a progress callback
    def progress_callback(msg: str):
        print(f"Progress: {msg}")

    try:
        agent = AgenticOrchestrator(config, progress_callback)

        # Test a simple query that should produce formatted output
        test_query = "Create a simple list of 3 benefits of using Python for data science."

        print("Testing agent formatting with query:", test_query)

        # Use the synchronous version for testing
        result = agent.run(test_query)

        print("\nAgent response:")
        print(result.answer.text)

        print("\nPath:", result.path)
        print("Citations:", result.answer.citations)

    except Exception as e:
        print(f"Agent test failed: {e}")
        print("This is expected if LLM backend is not configured")

if __name__ == "__main__":
    print("=== Testing Post-Processing ===")
    test_post_processing()

    print("\n=== Testing Agent Formatting ===")
    # Comment this out if you don't have a working LLM backend
    # asyncio.run(test_agent_formatting())
    print("Agent test skipped - uncomment to run if LLM backend is available")