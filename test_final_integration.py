#!/usr/bin/env python3
"""
Final integration test for chat-agent with MCP tools.
Tests the agent's ability to use MCP tools through LangChain.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from tools.mcp_tools import create_all_mcp_tools

def test_agent_with_tools():
    """Test the agent with MCP tools."""
    print("=" * 60)
    print("Chat Agent MCP Integration Test")
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
    try:
        tools = create_all_mcp_tools(mcp_base_url, mcp_verify_ssl)
        print(f"   ✓ Created {len(tools)} tools: {[t.name for t in tools]}")
    except Exception as e:
        print(f"   ✗ Failed to create tools: {e}")
        return False

    # Create agent
    print("\n2. Creating agent with tools...")
    try:
        agent = ChatAgent(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key="ollama",
            system_prompt="You are a helpful assistant with access to weather and web search tools.",
            tools=tools,
            use_agent=True
        )
        print("   ✓ Agent created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test simple query (no tools)
    print("\n3. Testing simple query (no tools needed)...")
    try:
        response = agent.chat("Hello, how are you?")
        print(f"   Response: {response[:100]}...")
        print("   ✓ Simple query works")
    except Exception as e:
        print(f"   ✗ Simple query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    print("The agent is ready to use MCP tools.")
    print("You can now run: uv run python src/main.py")
    return True


if __name__ == "__main__":
    success = test_agent_with_tools()
    sys.exit(0 if success else 1)
