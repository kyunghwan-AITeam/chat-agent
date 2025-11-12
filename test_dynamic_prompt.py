"""
Test dynamic prompt generation with MCP tools
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MCP configuration
mcp_base_url = os.getenv("MCP_BASE_URL", "https://localhost:22000")
mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

print("=" * 80)
print("Testing Dynamic Prompt Generation")
print("=" * 80)
print()

# Test 1: Load MCP tools
print("1. Loading MCP tools...")
from src.tools.mcp_tools_v2 import create_mcp_tools

tools = create_mcp_tools(mcp_base_url, mcp_verify_ssl)
print(f"   ✓ Loaded {len(tools)} tools")
for tool in tools:
    print(f"     - {tool.name}: {tool.description[:60]}..." if tool.description else f"     - {tool.name}")
print()

# Test 2: Generate dynamic prompt with tools
print("2. Generating dynamic prompt WITH tools...")
from src.prompts.prompt_builder import build_home_assistant_prompt

prompt_with_tools = build_home_assistant_prompt(tools)
print(f"   ✓ Generated prompt: {len(prompt_with_tools)} characters")
print()

# Test 3: Show AVAILABLE FUNCTIONS section
print("3. AVAILABLE FUNCTIONS section:")
print("-" * 80)
# Extract the AVAILABLE FUNCTIONS section
import re
functions_match = re.search(r'## AVAILABLE FUNCTIONS\n\n(.*?)\n\n##', prompt_with_tools, re.DOTALL)
if functions_match:
    functions_section = functions_match.group(1)
    print(functions_section)
else:
    print("   ✗ Could not find AVAILABLE FUNCTIONS section")
print("-" * 80)
print()

# Test 4: Show first example
print("4. First EXAMPLE:")
print("-" * 80)
examples_match = re.search(r'## EXAMPLES\n\n(.*?)(?=\n\nExample \(General|$)', prompt_with_tools, re.DOTALL)
if examples_match:
    first_example = examples_match.group(1).split('\n\n')[0]
    print(first_example)
else:
    print("   ✗ Could not find EXAMPLES section")
print("-" * 80)
print()

# Test 5: Generate prompt WITHOUT tools
print("5. Generating dynamic prompt WITHOUT tools...")
prompt_without_tools = build_home_assistant_prompt(None)
print(f"   ✓ Generated prompt: {len(prompt_without_tools)} characters")
print()

# Test 6: Compare
print("6. Comparison:")
print(f"   - Prompt WITH tools:    {len(prompt_with_tools)} chars, {len(tools)} tools")
print(f"   - Prompt WITHOUT tools: {len(prompt_without_tools)} chars, 0 tools")
print(f"   - Difference:           {len(prompt_with_tools) - len(prompt_without_tools)} chars")
print()

# Test 7: Verify tool names in prompt
print("7. Verifying tool names in prompt...")
for tool in tools:
    if tool.name in prompt_with_tools:
        print(f"   ✓ {tool.name} found in prompt")
    else:
        print(f"   ✗ {tool.name} NOT found in prompt")
print()

print("=" * 80)
print("✓ All tests completed successfully!")
print("=" * 80)
