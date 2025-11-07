#!/usr/bin/env python3
"""
Test script for MCP integration with chat-agent.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.mcp_client import MCPClient


async def test_weather_service():
    """Test weather service connection."""
    print("Testing Weather Service...")
    client = MCPClient("https://localhost:22000", verify_ssl=False)

    try:
        # List available tools
        print("  Listing weather tools...")
        tools = await client.list_tools("weather")
        print(f"  Available tools: {len(tools)}")
        for tool in tools:
            print(f"    - {tool.get('name', 'unknown')}")

        # Test weather query
        print("\n  Testing weather query for Seoul...")
        result = await client.call_tool(
            service="weather",
            tool_name="get_current_weather",
            arguments={"location": "Seoul"}
        )
        print(f"  Weather result: {result}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        await client.close()


async def test_web_search_service():
    """Test web search service connection."""
    print("\nTesting Web Search Service...")
    client = MCPClient("https://localhost:22000", verify_ssl=False)

    try:
        # List available tools
        print("  Listing search tools...")
        tools = await client.list_tools("search")
        print(f"  Available tools: {len(tools)}")
        for tool in tools:
            print(f"    - {tool.get('name', 'unknown')}")

        # Test search query
        print("\n  Testing web search for 'Python programming'...")
        result = await client.call_tool(
            service="search",
            tool_name="search_web",
            arguments={"query": "Python programming", "count": 3}
        )
        print(f"  Search result: {result}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        await client.close()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("MCP Integration Test")
    print("=" * 60)
    print()

    # Check if MCP servers are running
    print("Checking MCP server connectivity...")
    print("Expected: https://localhost:22000")
    print()

    weather_ok = await test_weather_service()
    search_ok = await test_web_search_service()

    print()
    print("=" * 60)
    print("Test Results:")
    print(f"  Weather Service: {'✓ PASS' if weather_ok else '✗ FAIL'}")
    print(f"  Web Search Service: {'✓ PASS' if search_ok else '✗ FAIL'}")
    print("=" * 60)

    if weather_ok and search_ok:
        print("\n✓ All tests passed! MCP integration is working.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check:")
        print("  1. MCP servers are running (./run-mcp-servers.sh)")
        print("  2. Ports are correct (22000 for HTTPS)")
        print("  3. TLS certificates are generated")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
