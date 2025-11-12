"""
Test the new MCP integration using langchain-mcp-adapters
"""
import os
import asyncio
from dotenv import load_dotenv

async def test_mcp():
    # Load environment variables
    load_dotenv()

    # Get MCP configuration
    mcp_base_url = os.getenv("MCP_BASE_URL", "https://localhost:22000")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    print(f"Testing MCP integration with langchain-mcp-adapters...")
    print(f"MCP Base URL: {mcp_base_url}")
    print(f"MCP Verify SSL: {mcp_verify_ssl}")
    print()

    try:
        from src.tools.mcp_tools_v2 import create_mcp_tools

        print("Loading MCP tools...")
        tools = create_mcp_tools(mcp_base_url, mcp_verify_ssl)

        print(f"\nSuccessfully loaded {len(tools)} tools:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                print(f"   Description: {tool.description[:100]}...")
            if hasattr(tool, 'args_schema') and tool.args_schema:
                # Check if args_schema is already a dict or needs to be converted
                if hasattr(tool.args_schema, 'model_json_schema'):
                    print(f"   Schema: {tool.args_schema.model_json_schema()}")
                else:
                    print(f"   Schema: {tool.args_schema}")
            print()

        # Test weather tool if available
        weather_tool = next((t for t in tools if 'weather' in t.name.lower()), None)
        if weather_tool:
            print(f"\nTesting weather tool: {weather_tool.name}")
            try:
                result = await weather_tool.ainvoke({"location": "Seoul"})
                print(f"Result: {result[:200]}...")
            except Exception as e:
                print(f"Error: {e}")

        print("\n✓ MCP integration test completed successfully!")

    except Exception as e:
        import traceback
        print(f"\n✗ Error testing MCP integration:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_mcp())
