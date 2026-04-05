#!/usr/bin/env python3
"""Test script to demonstrate the API performance issue."""

import asyncio
import time
import httpx

async def test_api_performance():
    """Test the API response times to confirm the performance issue."""
    base_url = "http://localhost:8000"

    # Simple health check should be fast
    print("Testing /health endpoint...")
    start = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        health_time = time.time() - start
        print(f"Health check: {health_time:.2f}s - {response.json()}")

    # Query endpoint should be slow (demonstrating the issue)
    print("\nTesting /query endpoint (expecting ~10+ seconds)...")
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{base_url}/query",
                json={"query": "What is Python?"},
                timeout=30
            )
            query_time = time.time() - start
            print(f"Query endpoint: {query_time:.2f}s")
            if query_time > 5:
                print("❌ SLOW RESPONSE CONFIRMED - API is blocking!")
            else:
                print("✅ Response is fast")
    except httpx.TimeoutException:
        print("❌ Request timed out - API is too slow!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("API Performance Test")
    print("===================")
    print("Make sure to start the API server first:")
    print("python main.py --serve")
    print()

    asyncio.run(test_api_performance())