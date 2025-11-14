"""
Dynamic prompt builder that generates system prompts based on available tools.
"""
from typing import List, Optional, Dict
from langchain_core.tools import BaseTool

def build_home_assistant_prompt(
    agent_instructions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build HOME_ASSISTANT_PROMPT with Agent information.

    Args:
        agent_instructions: Dictionary mapping agent names to their instructions (optional)

    Returns:
        Complete system prompt with Agent instructions and available tools
    """
    # Base prompt (MCP-focused)
    base_prompt = """You are a friendly Home Assistant chatbot. Follow these rules:

## Core Rules
1. **Korean Only**: Always respond in Korean (한국어)
2. **Concise**: Keep responses brief and natural
3. **Agents**: Use Agents to access external capabilities

## Response Format

**When calling Agents:**
```
네, 확인해 볼게요.
<AGENT_CALL>
{{
    "agent": server_name,
    "tool": tool_name,
    "params": {{"param1": "value1", "param2": "value2"}},
}}
</AGENT_CALL>
```

**Without AGENT:**
```
안녕하세요! 무엇을 도와드릴까요?
```

## AGENT Call Rules
- ALWAYS wrap AGENT calls in `<AGENT_CALL>` tags
- Specify the agent name, tool name, and parameters
- Use JSON format for parameters
- Example:
<AGENT_CALL>
{{
    "agent": "weather",
    "tool": "get_weather",
    "params": {{"location": "Seoul"}},
}}
</AGENT_CALL>

"""

    # Add MCP server instructions if available
    if agent_instructions:
        instructions_lines = ["## AVAILABLE AGENTS\n"]
        for server_name, instruction in agent_instructions.items():
            instructions_lines.append(f"### {server_name.upper()}")
            instructions_lines.append(instruction)
            instructions_lines.append("")

        instructions_section = "\n".join(instructions_lines)
        base_prompt = base_prompt + instructions_section
    else:
        # No MCP servers available
        no_mcp_section = """## AVAILABLE AGENTS

No AGENTs available. Respond naturally in Korean.

Example:
User: "안녕?"
Assistant: 안녕하세요! 무엇을 도와드릴까요?
"""
        base_prompt = base_prompt + no_mcp_section

    return base_prompt


def build_simple_assistant_prompt() -> str:
    """
    Build SIMPLE_ASSISTANT_PROMPT.

    Returns:
        Simple assistant system prompt
    """
    return "You are a helpful AI assistant. Always be conversational and friendly."

