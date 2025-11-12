"""
Dynamic prompt builder that generates system prompts based on available tools.
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
import json


def build_tool_description(tool: BaseTool) -> Dict[str, Any]:
    """
    Build a JSON-serializable tool description from a LangChain tool.

    Args:
        tool: LangChain BaseTool instance

    Returns:
        Dictionary with tool name, description, and parameters
    """
    tool_desc = {
        "name": tool.name,
        "description": tool.description or "No description available"
    }

    # Extract parameters from tool's args_schema
    if hasattr(tool, 'args_schema') and tool.args_schema:
        if isinstance(tool.args_schema, dict):
            # Already a dict
            schema = tool.args_schema
        elif hasattr(tool.args_schema, 'model_json_schema'):
            # Pydantic model
            schema = tool.args_schema.model_json_schema()
        elif hasattr(tool.args_schema, 'schema'):
            # Has schema method
            schema = tool.args_schema.schema()
        else:
            schema = {}

        # Extract properties and required fields
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        tool_desc["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required
        }
    else:
        tool_desc["parameters"] = {
            "type": "object",
            "properties": {},
            "required": []
        }

    return tool_desc


def generate_tools_json(tools: List[BaseTool]) -> str:
    """
    Generate JSON representation of available tools.

    Note: Escapes curly braces for LangChain prompt template compatibility.

    Args:
        tools: List of LangChain tools

    Returns:
        Formatted JSON string describing all tools (with escaped braces)
    """
    if not tools:
        return '{{\n  "functions": []\n}}'

    tools_list = [build_tool_description(tool) for tool in tools]
    tools_json = {"functions": tools_list}

    # Format with nice indentation
    json_str = json.dumps(tools_json, indent=2, ensure_ascii=False)

    # Escape curly braces for LangChain prompt template
    # Replace { with {{ and } with }}
    json_str = json_str.replace('{', '{{').replace('}', '}}')

    return json_str


def generate_tool_examples(tools: List[BaseTool]) -> str:
    """
    Generate example usage for each tool.

    Args:
        tools: List of LangChain tools

    Returns:
        Formatted examples string
    """
    if not tools:
        return ""

    examples = []

    for i, tool in enumerate(tools, 1):
        tool_name = tool.name
        description = tool.description or "No description"

        # Extract first required parameter for example
        example_param = None
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            elif hasattr(tool.args_schema, 'model_json_schema'):
                schema = tool.args_schema.model_json_schema()
            else:
                schema = {}

            properties = schema.get('properties', {})
            required = schema.get('required', [])

            if required and required[0] in properties:
                param_name = required[0]
                param_info = properties[param_name]
                param_desc = param_info.get('description', 'value')

                # Generate example value based on parameter name
                if 'location' in param_name.lower():
                    example_value = 'Seoul'
                elif 'query' in param_name.lower():
                    example_value = 'example query'
                elif 'city' in param_name.lower():
                    example_value = 'Seoul'
                else:
                    example_value = 'example'

                example_param = f'{param_name}="{example_value}"'

        if not example_param:
            example_param = 'param="value"'

        example = f"""Example {i} - {tool_name}:
User: "Use {tool_name}"
Assistant: <THOUGHT>
Executing {tool_name}.
</THOUGHT>

<TOOL_CALL>
[{tool_name}({example_param})]
</TOOL_CALL>"""

        examples.append(example)

    return "\n\n".join(examples)


def build_home_assistant_prompt(tools: Optional[List[BaseTool]] = None) -> str:
    """
    Build HOME_ASSISTANT_PROMPT with dynamic tool information.

    Args:
        tools: List of available LangChain tools (optional)

    Returns:
        Complete system prompt with tool descriptions
    """
    # Base prompt (tool-agnostic)
    base_prompt = """You are a friendly Home Assistant chatbot that helps users control their smart home.

IMPORTANT INSTRUCTIONS:
- Always respond in Korean (한국어로 응답하세요)
- Do NOT repeat your response
- Keep responses concise and clear
- You can include both tool calls AND conversational text in your response
- After receiving tool results, provide a natural Korean response explaining the information

## RESPONSE FORMAT

You can respond in TWO ways:

### 1. Tool Call WITH Conversational Response
When you need to call a tool, use this format:

<THOUGHT>
Brief explanation of what you're doing (optional, in Korean)
</THOUGHT>

<TOOL_CALL>
[function_name(param="value")]
</TOOL_CALL>

### 2. Conversational Response ONLY
When no tool is needed, just respond naturally in Korean without any tags.

## FUNCTION CALLING FORMAT

Tool calls must be wrapped in <TOOL_CALL></TOOL_CALL> tags and use Python-style syntax:

<TOOL_CALL>
[function_name(param1="value1", param2="value2")]
</TOOL_CALL>

For multiple tools at once:

<TOOL_CALL>
[function1(param="value"), function2(param="value")]
</TOOL_CALL>
"""

    # Add tools section if tools are available
    if tools:
        tools_json = generate_tools_json(tools)
        tools_section = f"\n## AVAILABLE FUNCTIONS\n\n{tools_json}\n"

        # Generate function names list
        function_names = [tool.name for tool in tools]
        function_names_str = ", ".join(function_names)

        # Add examples section
        examples_section = f"\n## EXAMPLES\n\n{generate_tool_examples(tools)}\n"

        # Add general conversation examples
        general_examples = """
Example (General conversation - no tool needed):
User: "안녕?"
Assistant: 안녕하세요! 무엇을 도와드릴까요?

Example (General question - no tool needed):
User: "오늘 뭐 할까?"
Assistant: 오늘은 날씨를 확인하거나 관심 있는 주제를 검색해보시는 건 어떨까요?
"""

        # Critical rules section
        critical_rules = f"""
CRITICAL RULES:
- When calling tools: ALWAYS wrap function calls in <TOOL_CALL></TOOL_CALL> tags
- <THOUGHT> tags are optional but recommended for clarity
- Use ONLY the function names defined above: {function_names_str}
- Always use keyword arguments with double quotes (e.g., param="value")
- For general conversation (no tool needed), respond naturally in Korean without any tags
- ALL conversational responses must be in Korean
"""

        return base_prompt + tools_section + examples_section + general_examples + critical_rules
    else:
        # No tools available
        no_tools_section = """
## AVAILABLE FUNCTIONS

No external tools are currently available.

## EXAMPLES

Example:
User: "안녕?"
Assistant: 안녕하세요! 무엇을 도와드릴까요?

CRITICAL RULES:
- Respond naturally in Korean
- Be helpful and conversational
- ALL responses must be in Korean
"""
        return base_prompt + no_tools_section


def build_simple_assistant_prompt() -> str:
    """
    Build SIMPLE_ASSISTANT_PROMPT.

    Returns:
        Simple assistant system prompt
    """
    return "You are a helpful AI assistant. Always be conversational and friendly."
