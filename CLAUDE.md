# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain-based conversational AI agent that uses Ollama (local LLM) for smart home control and general assistance. The project integrates with external MCP (Model Context Protocol) servers for weather information and web search capabilities.

## Development Commands

### Setup and Installation
```bash
# Install dependencies (recommended)
uv sync

# Alternative: using pip
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
```

### Running the Application
```bash
# Run the chat agent
uv run python src/main.py

# Alternative: using activated venv
python src/main.py
```

### Testing
```bash
# Test MCP integration
uv run python test_mcp_integration.py

# Test streaming responses
uv run python test_streaming.py

# Test tool calling functionality
uv run python test_tool_call.py
```

### Ollama Setup
```bash
# Pull a model
ollama pull llama3.2

# Start Ollama server (runs on port 11434 by default)
ollama serve

# List available models
ollama list
```

## Architecture

### Core Components

1. **ChatAgent** ([src/agents/chat_agent.py](src/agents/chat_agent.py))
   - Main conversational agent using LangChain
   - Supports both streaming and non-streaming responses
   - Handles tool calling in two formats:
     - Standard LangChain tool calls (JSON-RPC style)
     - Pythonic format for Gemma 3 model: `[function_name(param="value")]`
   - Manages conversation history and memory
   - Extracts and processes `<THOUGHT>` and `<TOOL_CALL>` tags from model responses

2. **System Prompts** ([src/prompts/](src/prompts/))
   - **Dynamic Prompt Builder** ([prompt_builder.py](src/prompts/prompt_builder.py)): Generates system prompts dynamically from MCP tools
   - `build_home_assistant_prompt(tools)`: Creates prompt with tool information automatically extracted from MCP
   - `system_prompts.py`: Legacy static prompts (kept for reference)
   - Automatically syncs with MCP server tool definitions

3. **MCP Integration** ([src/tools/](src/tools/))
   - **Official Implementation** ([mcp_tools_v2.py](src/tools/mcp_tools_v2.py)): Uses `langchain-mcp-adapters` library
   - `MultiServerMCPClient`: Supports multiple MCP servers simultaneously
   - `create_mcp_tools()`: Synchronous wrapper for tool loading
   - `create_mcp_tools_async()`: Async tool loading with SSL configuration
   - Legacy files moved to `src/tools/legacy/`

4. **Langfuse Integration** ([src/utils/langfuse_config.py](src/utils/langfuse_config.py))
   - LLM observability and tracing with Langfuse
   - `LangfuseConfig`: Configuration manager for Langfuse client
   - `init_langfuse()`: Initialize Langfuse with environment variables
   - Conditionally enabled via `LANGFUSE_ENABLED` environment variable
   - Uses LangChain's `CallbackHandler` for automatic tracing

### MCP Protocol Flow

The MCP client follows a session-based protocol:
1. Acquire session ID via `noop` method
2. Initialize session with `initialize` method
3. Call tools with session ID in headers
4. Responses are in Server-Sent Events (SSE) format

### Tool Calling Mechanism

The agent supports a custom tool calling format for models that don't natively support function calling:

```
<THOUGHT>
Brief explanation in Korean (optional)
</THOUGHT>

<TOOL_CALL>
[function_name(param1="value1", param2="value2")]
</TOOL_CALL>
```

The `extract_tool_calls_and_text()` method parses these tags and the `_parse_pythonic_tool_calls()` method converts the pythonic syntax into standard tool calls.

### Response Streaming

The agent uses two response modes:
- `chat()`: Returns complete response at once
- `chat_stream()`: Yields response chunks in real-time

When tools are called, the agent:
1. Streams initial response until tool call is detected
2. Executes tool(s) synchronously
3. Streams LLM interpretation of tool results

## Configuration

### Environment Variables (.env)

Required variables:
- `OLLAMA_MODEL`: Model name (e.g., `llama3.2`, `mistral`, `qwen2.5`)
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: `http://localhost:11434/v1`)
- `TEMPERATURE`: Response randomness 0.0-2.0 (default: `0.7`)

MCP-specific variables:
- `USE_MCP_TOOLS`: Enable/disable MCP tools (`true`/`false`)
- `MCP_BASE_URL`: MCP server base URL (default: `https://localhost:22000`)
- `MCP_VERIFY_SSL`: SSL verification for MCP (`true`/`false`)

Langfuse-specific variables:
- `LANGFUSE_ENABLED`: Enable/disable Langfuse tracing (`true`/`false`)
- `LANGFUSE_BASE_URL` or `LANGFUSE_HOST`: Langfuse server URL (e.g., `http://192.168.3.20:3000`)
- `LANGFUSE_PUBLIC_KEY`: Public API key from Langfuse (optional)
- `LANGFUSE_SECRET_KEY`: Secret API key from Langfuse (optional)
- `LANGFUSE_TRACING_ENVIRONMENT`: Environment tag (e.g., `development`, `production`)

### Switching System Prompts

To change the agent's behavior, modify the system prompt in [src/main.py](src/main.py):

```python
from prompts import SIMPLE_ASSISTANT_PROMPT  # or HOME_ASSISTANT_PROMPT

agent = ChatAgent(
    # ...
    system_prompt=SIMPLE_ASSISTANT_PROMPT
)
```

## Key Technical Details

### Ollama Integration

The project uses OpenAI-compatible API to connect to Ollama:
- Uses `langchain-openai` package with custom `base_url`
- No real API key needed (uses dummy value `"ollama"`)
- Supports any model available in Ollama

### Korean Language Support

The project uses `prompt-toolkit` for better Korean input handling:
- Proper backspace behavior with multibyte characters
- Input history management
- Session-based prompting in the main loop

### Tool Execution with Streaming

When streaming is enabled and tools are called:
1. The full response is accumulated during streaming
2. After streaming completes, tool calls are parsed
3. Tools are executed synchronously (async MCP calls wrapped)
4. A second LLM call interprets tool results
5. The interpretation is streamed to the user

### MCP Async/Sync Bridge

MCP client is async but LangChain tools are sync. The bridge is handled in `MCPToolWrapper.run()`:
- Detects if running in async context
- Uses thread pool executor if needed
- Creates new event loop for synchronous execution

## Testing Strategy

Test files demonstrate different aspects:
- `test_mcp_integration.py`: MCP connection and tool listing
- `test_streaming.py`: Streaming response behavior
- `test_tool_call.py`: Tool calling functionality
- `test_final.py`: End-to-end integration

Run tests to verify functionality after making changes.

## External Dependencies

### MCP Servers (Optional)

The project can connect to external MCP servers from the `mcp-servers` project:
- Weather server: `https://localhost:22000/weather/mcp`
- Search server: `https://localhost:22000/search/mcp`

See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) for setup instructions.

### Required Packages

Core dependencies from [pyproject.toml](pyproject.toml):
- `langchain>=0.3.0`: Agent framework
- `langchain-openai>=0.2.0`: OpenAI-compatible LLM integration
- `python-dotenv>=1.0.0`: Environment variable management
- `prompt-toolkit>=3.0.0`: Enhanced terminal input
- `mcp>=1.0.0`: MCP protocol support
- `httpx>=0.27.0`: HTTP client for MCP

## Model Compatibility

Not all Ollama models support function calling equally:
- **Recommended**: `llama3.2`, `mistral`, `qwen2.5`, `gemma3`
- **For tool calling**: Models must understand the pythonic format or support native function calling
- **Gemma 3**: Uses pythonic format `[func(param="value")]` parsed by the agent

Test with your chosen model to ensure tool calling works correctly.
