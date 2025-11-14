"""
Test MCP server instructions and tools retrieval
"""
import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.mcp_tools_v2 import get_mcp_server_instructions, get_mcp_tools_by_server
from prompts.system_prompt_builder import build_home_assistant_prompt

# Load environment variables
load_dotenv()

def test_mcp_integration():
    """Test fetching MCP server instructions and tools"""

    # Get MCP configuration
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:22001")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    print(f"Testing MCP integration from: {mcp_base_url}")
    print(f"SSL Verification: {mcp_verify_ssl}\n")

    # Get instructions
    print("="*80)
    print("1. Fetching MCP Server Instructions")
    print("="*80)
    instructions = get_mcp_server_instructions(mcp_base_url, mcp_verify_ssl)

    if instructions:
        print(f"✓ Successfully retrieved instructions from {len(instructions)} servers:\n")
        for server_name, instruction in instructions.items():
            print(f"--- {server_name.upper()} Server ---")
            print(instruction)
            print()
    else:
        print("✗ No instructions retrieved or MCP servers are not available")
        print("This is normal if MCP servers are not running\n")

    # Get tools by server
    print("="*80)
    print("2. Fetching MCP Tools by Server")
    print("="*80)
    mcp_tools = get_mcp_tools_by_server(mcp_base_url, mcp_verify_ssl)

    if mcp_tools:
        print(f"✓ Successfully retrieved tools from {len(mcp_tools)} servers:\n")
        for server_name, tools in mcp_tools.items():
            print(f"--- {server_name.upper()} Server ({len(tools)} tools) ---")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:60] if hasattr(tool, 'description') and tool.description else 'No description'}...")
            print()
    else:
        print("✗ No tools retrieved or MCP servers are not available")
        print("This is normal if MCP servers are not running\n")

    # Test prompt building with MCP server information
    print("="*80)
    print("3. Building System Prompt with MCP Information")
    print("="*80)

    prompt = build_home_assistant_prompt(
        mcp_instructions=instructions if instructions else None,
        mcp_tools=mcp_tools if mcp_tools else None
    )

    print("\nGenerated System Prompt:")
    print("-"*80)
    print(prompt)
    print("-"*80)

    if instructions and mcp_tools:
        print(f"\n✓ Prompt includes MCP server instructions and tools")
    elif instructions:
        print(f"\n✓ Prompt includes MCP server instructions (no tools)")
    else:
        print(f"\n⚠ Prompt built without MCP information (servers not available)")

    print(f"\nTotal prompt length: {len(prompt)} characters")

    # Verify MCP_CALL format
    print("\n" + "="*80)
    print("4. Verification")
    print("="*80)
    if "<MCP_CALL>" in prompt:
        print("✓ MCP_CALL format present in prompt")
    else:
        print("✗ MCP_CALL format NOT found")

    if "AVAILABLE MCP SERVERS" in prompt:
        print("✓ AVAILABLE MCP SERVERS section present")
    else:
        print("✗ AVAILABLE MCP SERVERS section NOT found")

if __name__ == "__main__":
    test_mcp_integration()
