#!/usr/bin/env python3
"""
Test tool calling with Gemma 3
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from tools.mcp_tools import create_all_mcp_tools

def test_tool_calling():
    """Test tool calling."""
    print("=" * 60)
    print("Testing Tool Calling with Gemma 3")
    print("=" * 60)
    print()

    # Load environment
    load_dotenv()

    # Get configuration
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    mcp_base_url = os.getenv("MCP_BASE_URL", "https://localhost:22000")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    # Create MCP tools
    print("1. Creating MCP tools...")
    tools = create_all_mcp_tools(mcp_base_url, mcp_verify_ssl)
    print(f"   ✓ Created {len(tools)} tools: {[t.name for t in tools]}")

    # Create agent
    print("\n2. Creating agent with tools...")
    agent = ChatAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key="ollama",
        system_prompt="""You are a helpful assistant with access to tools.

Available tools:
- get_weather(location: str): Get weather for a location
- search_web(query: str): Search the web

When you need to use a tool, respond with this exact format:

[get_weather(location='Seoul')]

or for multiple tools:

[get_weather(location='Seoul'), search_web(query='weather')]

Always use keyword arguments and wrap in square brackets [].""",
        tools=tools,
        use_agent=True
    )
    print("   ✓ Agent created successfully")

    # Test weather query
    print("\n3. Testing weather query...")
    print("   Query: What's the weather in Seoul?")
    response = agent.chat("What's the weather in Seoul?")
    print(f"\n   Response:\n   {response}\n")

    return True


if __name__ == "__main__":
    success = test_tool_calling()
    sys.exit(0 if success else 1)
