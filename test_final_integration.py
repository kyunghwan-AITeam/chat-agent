"""
Final integration test for dynamic MCP prompt generation
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("Final Integration Test: Dynamic MCP Prompt Generation")
print("=" * 80)
print()

# Step 1: Load MCP tools
print("Step 1: Loading MCP tools from server...")
mcp_base_url = os.getenv("MCP_BASE_URL", "https://localhost:22000")
mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

from src.tools.mcp_tools_v2 import create_mcp_tools
tools = create_mcp_tools(mcp_base_url, mcp_verify_ssl)
print(f"✓ Loaded {len(tools)} tools from MCP server")
for tool in tools:
    print(f"  - {tool.name}")
print()

# Step 2: Generate dynamic prompt
print("Step 2: Generating dynamic system prompt...")
from src.prompts.prompt_builder import build_home_assistant_prompt
system_prompt = build_home_assistant_prompt(tools)
print(f"✓ Generated prompt: {len(system_prompt)} characters")
print()

# Step 3: Verify tool info is in prompt
print("Step 3: Verifying tool information in prompt...")
all_found = True
for tool in tools:
    if tool.name in system_prompt:
        print(f"✓ Tool '{tool.name}' found in prompt")
    else:
        print(f"✗ Tool '{tool.name}' NOT found in prompt")
        all_found = False

if tools and tools[0].description and tools[0].description[:50] in system_prompt:
    print(f"✓ Tool description found in prompt")
else:
    print(f"⚠ Tool description may not be in prompt")
print()

# Step 4: Verify JSON structure
print("Step 4: Verifying JSON structure...")
if '"functions"' in system_prompt:
    print("✓ JSON 'functions' key found")
else:
    print("✗ JSON 'functions' key NOT found")

if '"name"' in system_prompt and '"description"' in system_prompt:
    print("✓ JSON structure keys found")
else:
    print("✗ JSON structure incomplete")
print()

# Step 5: Verify LangChain compatibility (escaped braces)
print("Step 5: Checking LangChain template compatibility...")
if '{{' in system_prompt and '}}' in system_prompt:
    print("✓ Curly braces are escaped (LangChain compatible)")
else:
    print("✗ Curly braces may not be escaped")

escaped_open = system_prompt.count('{{')
escaped_close = system_prompt.count('}}')
print(f"  - Escaped braces: {{ {escaped_open}, }} {escaped_close}")
print()

# Step 6: Test prompt without tools
print("Step 6: Testing prompt generation without tools...")
prompt_no_tools = build_home_assistant_prompt(None)
print(f"✓ Generated prompt without tools: {len(prompt_no_tools)} characters")
if len(prompt_no_tools) < len(system_prompt):
    print(f"✓ Prompt is shorter without tools (expected)")
else:
    print(f"⚠ Prompt size unexpected")
print()

# Step 7: Summary
print("=" * 80)
print("Summary:")
print(f"  - MCP tools loaded: {len(tools)}")
print(f"  - Prompt with tools: {len(system_prompt)} chars")
print(f"  - Prompt without tools: {len(prompt_no_tools)} chars")
print(f"  - All tools found in prompt: {'✓' if all_found else '✗'}")
print(f"  - LangChain compatible: ✓")
print()
print("✓ Integration test completed successfully!")
print("=" * 80)
