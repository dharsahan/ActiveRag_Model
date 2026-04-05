# API Performance Fix Summary

## Root Cause Analysis ✅

**Issue:** All API requests taking 10+ seconds since today's commit (c40cb53)

**Root Cause:** The `/query` endpoint was calling synchronous `pipeline.run()` which blocks FastAPI's event loop by using `run_until_complete()` for async web scraping operations.

**Evidence:**
- Timeline: Issue started with today's commit that added async web scraping
- Scope: All API endpoints affected due to shared pipeline architecture
- Technical: Synchronous wrapper around async Playwright operations blocking event loop

## Solution Implemented ✅

**Fix:** Added `async def run_async()` method to AgenticOrchestrator and updated API endpoint

**Changes Made:**

1. **`active_rag/agent.py`**:
   - Added `async def run_async()` method (copy of `run()` but uses `await self._execute_tool_async()`)
   - Kept original `run()` method for backward compatibility

2. **`active_rag/api.py`**:
   - Changed `/query` endpoint from `def query()` to `async def query()`
   - Updated call from `pipeline.run()` to `await pipeline.run_async()`

## Expected Performance Improvement

**Before Fix:**
- `/query` endpoint: 10+ seconds (blocking)
- All requests queued behind slow operations

**After Fix:**
- `/query` endpoint: ~1-3 seconds (non-blocking async)
- Concurrent requests handled properly by FastAPI

## Testing

To test the fix:

1. **Start the API server:**
   ```bash
   python main.py --serve
   ```

2. **Test the endpoints:**
   ```bash
   # Health check (should be fast)
   curl http://localhost:8000/health

   # Query endpoint (should now be fast and async)
   curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"query": "What is Python?"}'
   ```

3. **Concurrent request test:**
   - Make multiple requests simultaneously
   - Should no longer queue behind each other

## Rollback Plan

If issues arise, revert the API endpoint change:

```python
# In active_rag/api.py, line 57-58:
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):  # Remove 'async'
    result = pipeline.run(req.query)  # Remove 'await'
```

## Architecture Notes

- The `/query/stream` endpoint was already async and performing well
- This fix brings the non-streaming endpoint to the same performance level
- Both sync and async methods are available for different use cases