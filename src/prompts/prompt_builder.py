"""
Dynamic prompt builder that generates system prompts based on available tools.
"""
from typing import List, Optional
from langchain_core.tools import BaseTool


def build_tool_description(tool: BaseTool) -> str:
    """
    Build a clearly formatted tool description.

    Args:
        tool: LangChain BaseTool instance

    Returns:
        Formatted string with tool name, description, and parameters
    """
    lines = [
        f"### {tool.name}",
        f"Description: {tool.description or 'No description'}"
    ]

    # Extract parameters
    if hasattr(tool, 'args_schema') and tool.args_schema:
        try:
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            elif hasattr(tool.args_schema, 'model_json_schema'):
                schema = tool.args_schema.model_json_schema()
            else:
                schema = {}

            properties = schema.get('properties', {})
            required = schema.get('required', [])

            if properties:
                lines.append("Parameters:")
                for param_name, param_info in properties.items():
                    param_desc = param_info.get('description', '')
                    is_required = "✓" if param_name in required else "○"
                    lines.append(f"  {is_required} {param_name}: {param_desc}")
        except Exception:
            pass

    return "\n".join(lines)


def generate_tools_json(tools: List[BaseTool]) -> str:
    """
    Generate clearly formatted text representation of available tools.

    Args:
        tools: List of LangChain tools

    Returns:
        Formatted string describing all tools
    """
    if not tools:
        return "No tools available."

    tool_descriptions = [build_tool_description(tool) for tool in tools]
    return "\n\n".join(tool_descriptions)


# def generate_tool_examples(tools: List[BaseTool]) -> str:
#     """
#     Generate example usage for each tool.

#     Args:
#         tools: List of LangChain tools

#     Returns:
#         Formatted examples string
#     """
#     if not tools:
#         return ""

#     examples = []

#     for i, tool in enumerate(tools, 1):
#         tool_name = tool.name
#         description = tool.description or "No description"

#         # Extract first required parameter for example
#         example_param = None
#         if hasattr(tool, 'args_schema') and tool.args_schema:
#             if isinstance(tool.args_schema, dict):
#                 schema = tool.args_schema
#             elif hasattr(tool.args_schema, 'model_json_schema'):
#                 schema = tool.args_schema.model_json_schema()
#             else:
#                 schema = {}

#             properties = schema.get('properties', {})
#             required = schema.get('required', [])

#             if required and required[0] in properties:
#                 param_name = required[0]
#                 param_info = properties[param_name]
#                 param_desc = param_info.get('description', 'value')

#                 # Generate example value based on parameter name
#                 if 'location' in param_name.lower():
#                     example_value = 'Seoul'
#                 elif 'query' in param_name.lower():
#                     example_value = 'example query'
#                 elif 'city' in param_name.lower():
#                     example_value = 'Seoul'
#                 else:
#                     example_value = 'example'

#                 example_param = f'{param_name}="{example_value}"'

#         if not example_param:
#             example_param = 'param="value"'

#         example = f"""Example {i} - {tool_name}:
# User: "Use {tool_name}"
# Assistant: <THOUGHT>
# Executing {tool_name}.
# </THOUGHT>

# <TOOL_CALL>
# [{tool_name}({example_param})]
# </TOOL_CALL>"""

#         examples.append(example)

#     return "\n\n".join(examples)


def build_home_assistant_prompt(tools: Optional[List[BaseTool]] = None) -> str:
    """
    Build HOME_ASSISTANT_PROMPT with dynamic tool information.

    Args:
        tools: List of available LangChain tools (optional)

    Returns:
        Complete system prompt with tool descriptions
    """
    # Base prompt (tool-agnostic)
    base_prompt = """You are a friendly Home Assistant chatbot. Follow these rules:

## Core Rules
1. **Korean Only**: Always respond in Korean (한국어)
2. **Concise**: Keep responses brief and natural
3. **Limitations**: Only use functions listed below. If impossible, politely decline in Korean.

## Response Format

**With Tool:**
```
네, 확인해 볼게요.
<TOOL_CALL>
[function_name(param="value")]
</TOOL_CALL>
```

**Without Tool:**
```
안녕하세요! 무엇을 도와드릴까요?
```

## Tool Call Rules
- ALWAYS wrap in `<TOOL_CALL>` tags
- ALWAYS use list format: `[function_name(param="value")]`
- Use double quotes for all string parameters
- Multiple calls: `[func1(param="val1"), func2(param="val2")]`

"""

    # Add tools section if tools are available
    if tools:
        tools_json = generate_tools_json(tools)
        tools_section = f"## AVAILABLE FUNCTIONS\n\n{tools_json}\n"
        return base_prompt + tools_section
    else:
        # No tools available
        no_tools_section = """## AVAILABLE FUNCTIONS

No external tools available. Respond naturally in Korean.

Example:
User: "안녕?"
Assistant: 안녕하세요! 무엇을 도와드릴까요?
"""
        return base_prompt + no_tools_section


def build_simple_assistant_prompt() -> str:
    """
    Build SIMPLE_ASSISTANT_PROMPT.

    Returns:
        Simple assistant system prompt
    """
    return "You are a helpful AI assistant. Always be conversational and friendly."

