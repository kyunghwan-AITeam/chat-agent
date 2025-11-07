#!/usr/bin/env python3
"""
Test streaming with tool calls
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from tools.mcp_tools import create_all_mcp_tools

def test_streaming():
    """Test streaming tool calling."""
    print("=" * 60)
    print("Testing Streaming Tool Calling")
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
    print("Creating MCP tools...")
    tools = create_all_mcp_tools(mcp_base_url, mcp_verify_ssl)
    print(f"Created {len(tools)} tools\n")

    # Create agent
    print("Creating agent...")
    from prompts import HOME_ASSISTANT_PROMPT
    agent = ChatAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key="ollama",
        system_prompt=HOME_ASSISTANT_PROMPT,
        tools=tools,
        use_agent=True
    )
    print("Agent created\n")

    # Test streaming
    print("Query: 서울 날씨 알려줘")
    print("\nStreaming response:")
    print("-" * 60)
    for chunk in agent.chat_stream("서울 날씨 알려줘"):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 60)

    return True


if __name__ == "__main__":
    success = test_streaming()
    sys.exit(0 if success else 1)
