"""
Main entry point for LangChain Chat Agent (LLM)
"""
import os
import logging
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from agents.chat_agent import ChatAgent
from prompts.system_prompt_builder import build_home_assistant_prompt
from mem0 import Memory


# Load environment variables
load_dotenv()

# Suppress OpenTelemetry error messages when Langfuse server is unavailable
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "chat_memories",
            "embedding_model_dims": 1536,
            "on_disk": True,
            "path": "/tmp/qdrant_mem0"
        },
    },
    # "llm": {
    #     "provider": "ollama",
    #     "config": {
    #         "model": "qwen3:32b",
    #         "ollama_base_url": "http://172.168.0.201:11434",
    #         "temperature": 0.1
    #     },
    # },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    # "embedder": {
    #     "provider": "ollama",
    #     "config": {
    #         "model": "qwen3-embedding:0.6b",
    #         "ollama_base_url": "http://172.168.0.201:11434",
    #         "embedding_dims": 1024,
    #     },
    # },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "embedding_dims": 1536,
        }
    },
}

memory = Memory.from_config(config)


def main():
    """Main function to run the chat agent"""


    # Get LLM configuration from environment
    model = os.getenv("LLM_MODEL", "llama3.2")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "local")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))

    # Get MCP configuration
    use_mcp_tools = os.getenv("USE_MCP_TOOLS", "false").lower() == "true"
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:22001")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    # Get Langfuse configuration
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

    # Initialize MCP tools and instructions if enabled
    tools = []
    mcp_instructions = {}
    mcp_tools_by_server = {}
    if use_mcp_tools:
        try:
            from tools.mcp_tools_v2 import (
                create_mcp_tools,
                get_mcp_server_instructions,
                get_mcp_tools_by_server
            )

            # Get MCP tools grouped by server
            mcp_tools_by_server = get_mcp_tools_by_server(mcp_base_url, mcp_verify_ssl)

            # Flatten tools for agent use (backward compatibility)
            for server_tools in mcp_tools_by_server.values():
                tools.extend(server_tools)

            print(f"MCP Tools Loaded: {len(tools)} tools from {len(mcp_tools_by_server)} servers")
            for server_name, server_tools in mcp_tools_by_server.items():
                print(f"  [{server_name}] {len(server_tools)} tools")

            # Get MCP server instructions
            agent_instructions = get_mcp_server_instructions(mcp_base_url, mcp_verify_ssl)
            if agent_instructions:
                print(f"AGENT Instructions Loaded from {len(agent_instructions)} servers")
        except Exception as e:
            print(f"Warning: Could not load AGENT tools: {e}")
            print("Continuing without AGENT tools...")
            tools = []
            mcp_instructions = {}
            mcp_tools_by_server = {}

    # Add memory search tools
    try:
        from tools.memory_tools import create_memory_tools
        memory_tools = create_memory_tools(memory, user_id="alex")
        tools.extend(memory_tools)
        print(f"Memory Tools Loaded: {len(memory_tools)} tools available")
        for tool in memory_tools:
            print(f"  - {tool.name}: {tool.description[:80]}...")
    except Exception as e:
        print(f"Warning: Could not load memory tools: {e}")

    # Initialize Langfuse if enabled
    if langfuse_enabled:
        try:
            from utils.langfuse_config import init_langfuse
            init_langfuse()
            langfuse_host = os.getenv('LANGFUSE_BASE_URL') or os.getenv('LANGFUSE_HOST', 'not configured')
            print(f"Langfuse: Enabled ({langfuse_host})")
        except Exception as e:
            print(f"Warning: Could not initialize Langfuse: {e}")
            langfuse_enabled = False

    # Build dynamic system prompt based on MCP servers
    system_prompt = build_home_assistant_prompt(
        agent_instructions=agent_instructions if agent_instructions else None,
    )

    # Initialize agent with Home Assistant prompt
    print(f"\nInitializing Home Assistant Chat Agent with LLM...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Temperature: {temperature}")
    print(f"MCP Tools: {'Enabled' if use_mcp_tools and tools else 'Disabled'}")
    print(f"Langfuse: {'Enabled' if langfuse_enabled else 'Disabled'}")

    # Debug: Check if prompt is loaded
    print(f"\nSystem Prompt Generated: {len(system_prompt)} characters")
    print(f"First 100 chars: {system_prompt[:100]}...")

    agent = ChatAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        agents=tools,
        use_agent=len(tools) > 0,  # Enable agent if any tools are available
        enable_langfuse=langfuse_enabled
    )

    # Verify prompt is set correctly
    actual_prompt = agent.get_system_prompt()
    print(f"Actual System Prompt Set: {len(actual_prompt)} characters\n")

    print("Home Assistant Agent ready! Type 'quit' or 'exit' to end the conversation.\n")
    print("Commands:")
    print("  - 'reset': Clear conversation history")
    print("  - 'quit', 'exit', 'bye': Exit the program")
    print("  - 'prompt': Show current system prompt")
    print("\nI can help you control your smart home devices!")
    if use_mcp_tools and any(tool.name not in ["search_memory", "get_all_memories"] for tool in tools):
        print("I also have access to weather information and web search capabilities via MCP!")
    if any(tool.name in ["search_memory", "get_all_memories"] for tool in tools):
        print("I can search through our previous conversations and remember past discussions!")
    print("Try asking me to turn on lights, check temperature, or activate scenes.\n")

    # Create prompt session for better input handling (especially Korean)
    session = PromptSession(history=InMemoryHistory())

    # Chat loop
    while True:
        try:
            user_input = session.prompt("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset_memory()
                print("Conversation history cleared.\n")
                continue

            if user_input.lower() == "prompt":
                print(f"\n=== Current System Prompt ===")
                print(agent.get_system_prompt())
                print("=" * 50 + "\n")
                continue

            # Get response from agent (streaming)
            print("\nAgent: ", end="", flush=True)
            memory.add([{"role": "user", "content": user_input}], user_id="alex")
            full_content = ""
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
                full_content += chunk
            memory.add([{"role": "assistant", "content": full_content}], user_id="alex")
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
