#!/usr/bin/env python3
"""Debug script to test MCP connection step by step."""
import asyncio
import httpx


async def test_connection():
    """Test MCP connection step by step."""
    base_url = "https://localhost:22000"
    service = "weather"
    url = f"{base_url}/{service}/mcp"

    client = httpx.AsyncClient(verify=False, timeout=30.0)

    print(f"Testing connection to: {url}\n")

    # Step 1: Try noop to get session
    print("Step 1: Sending noop request...")
    noop_payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "noop",
        "params": {}
    }

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }

    try:
        response = await client.post(url, json=noop_payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content: {response.text[:500]}")

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            print(f"\nSession ID obtained: {session_id}")
        else:
            print("\nNo session ID in response!")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Response status: {getattr(e, 'response', None)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(test_connection())
