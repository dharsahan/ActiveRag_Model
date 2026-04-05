#!/usr/bin/env python3
"""Test script to verify the API performance fix works correctly."""

import asyncio
import time
from active_rag.config import Config
from active_rag.agent import AgenticOrchestrator

async def test_async_performance():
    """Test that the new async method works and is performant."""
    print("Testing AgenticOrchestrator.run_async() method...")

    config = Config()
    orchestrator = AgenticOrchestrator(config)

    # Test a simple query that shouldn't require web search
    start = time.time()
    try:
        result = await orchestrator.run_async("What is 2 + 2?")
        elapsed = time.time() - start

        print(f"✅ Async query completed in {elapsed:.2f}s")
        print(f"   Answer: {result.answer.text[:100]}...")
        print(f"   Path: {result.path}")

        if elapsed < 5:
            print("✅ Performance looks good!")
        else:
            print("⚠️ Still slow - may need more investigation")

    except Exception as e:
        print(f"❌ Error during async query: {e}")

def test_sync_vs_async():
    """Compare sync vs async method signatures."""
    config = Config()
    orchestrator = AgenticOrchestrator(config)

    # Check that both methods exist
    assert hasattr(orchestrator, 'run'), "Sync run() method missing"
    assert hasattr(orchestrator, 'run_async'), "Async run_async() method missing"

    print("✅ Both run() and run_async() methods exist")
    print("✅ API can now choose between sync and async execution")

if __name__ == "__main__":
    print("API Performance Fix Verification")
    print("=================================")

    # Test method existence
    test_sync_vs_async()
    print()

    # Test async performance
    asyncio.run(test_async_performance())