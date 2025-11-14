"""
Test prompt building with mock MCP instructions and tools
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prompts.system_prompt_builder import build_home_assistant_prompt
from unittest.mock import Mock

def test_prompt_with_mcp_servers():
    """Test prompt building with mock MCP server information"""

    # Mock MCP server instructions
    mock_instructions = {
        "weather": "Use this server to get weather information. Always provide location in English.",
        "search": "Use this server to search the web. Provide clear and specific search queries."
    }

    # Mock MCP tools by server
    mock_weather_tool = Mock()
    mock_weather_tool.name = "get_weather"
    mock_weather_tool.description = "Get current weather and forecast for a location"

    mock_search_tool = Mock()
    mock_search_tool.name = "search_web"
    mock_search_tool.description = "Search the web for information"

    mock_tools = {
        "weather": [mock_weather_tool],
        "search": [mock_search_tool]
    }

    print("="*80)
    print("Testing Prompt Building with MCP Server Information")
    print("="*80)
    print()

    # Build prompt without MCP
    print("1. Prompt WITHOUT MCP servers:")
    print("-" * 80)
    prompt_without = build_home_assistant_prompt()
    print(prompt_without)
    print()
    print(f"Length: {len(prompt_without)} characters")
    print()

    # Build prompt with MCP instructions only
    print("\n" + "="*80)
    print("2. Prompt WITH MCP instructions only:")
    print("-" * 80)
    prompt_instructions = build_home_assistant_prompt(mcp_instructions=mock_instructions)
    print(prompt_instructions)
    print()
    print(f"Length: {len(prompt_instructions)} characters")
    print()

    # Build prompt with MCP instructions and tools
    print("\n" + "="*80)
    print("3. Prompt WITH MCP instructions and tools:")
    print("-" * 80)
    prompt_full = build_home_assistant_prompt(
        mcp_instructions=mock_instructions,
        mcp_tools=mock_tools
    )
    print(prompt_full)
    print()
    print(f"Length: {len(prompt_full)} characters")
    print()

    # Verify new format
    print("\n" + "="*80)
    print("4. Verification:")
    print("-" * 80)
    if "<MCP_CALL>" in prompt_full:
        print("✓ MCP_CALL format found in prompt")
    else:
        print("✗ MCP_CALL format NOT found")

    if "server:" in prompt_full and "tool:" in prompt_full and "params:" in prompt_full:
        print("✓ MCP call structure (server/tool/params) found")
    else:
        print("✗ MCP call structure NOT found")

    if "AVAILABLE MCP SERVERS" in prompt_full:
        print("✓ AVAILABLE MCP SERVERS section found")
    else:
        print("✗ AVAILABLE MCP SERVERS section NOT found")

    if "WEATHER Server" in prompt_full:
        print("✓ Weather server found")
    else:
        print("✗ Weather server NOT found")

    if "get_weather" in prompt_full:
        print("✓ Weather tool (get_weather) found")
    else:
        print("✗ Weather tool NOT found")

    if "search_web" in prompt_full:
        print("✓ Search tool (search_web) found")
    else:
        print("✗ Search tool NOT found")

if __name__ == "__main__":
    test_prompt_with_mcp_servers()
