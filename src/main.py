"""
Main entry point for LangChain Chat Agent (LLM)
"""
import os
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from agents.chat_agent import ChatAgent
from prompts.prompt_builder import build_home_assistant_prompt
from langchain_ollama import OllamaEmbeddings
from mem0 import Memory

# ollama_embeddings = OllamaEmbeddings(
#     base_url="http://172.168.0.201:11434",
#     model="bge-m3:latest"
# )
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "chat_memories",
            "embedding_model_dims": 1024,
            "on_disk": True,
            "path": "/tmp/qdrant_mem0"
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen3:32b",
            "ollama_base_url": "http://172.168.0.201:11434",
            "temperature": 0.1
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:0.6b",
            "ollama_base_url": "http://172.168.0.201:11434",
            "embedding_dims": 1024,
        },
    },
}

memory = Memory.from_config(config)


def main():
    """Main function to run the chat agent"""
    # Load environment variables
    load_dotenv()

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

    # Initialize MCP tools if enabled
    tools = []
    if use_mcp_tools:
        try:
            from tools.mcp_tools_v2 import create_mcp_tools
            tools = create_mcp_tools(mcp_base_url, mcp_verify_ssl)
            print(f"MCP Tools Loaded: {len(tools)} tools available")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:80] if hasattr(tool, 'description') and tool.description else 'No description'}...")
        except Exception as e:
            print(f"Warning: Could not load MCP tools: {e}")
            print("Continuing without MCP tools...")
            tools = []

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
            print(f"Langfuse: Enabled ({os.getenv('LANGFUSE_HOST', 'not configured')})")
        except Exception as e:
            print(f"Warning: Could not initialize Langfuse: {e}")
            langfuse_enabled = False

    # Build dynamic system prompt based on available tools
    system_prompt = build_home_assistant_prompt(tools if use_mcp_tools and tools else None)

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
        tools=tools,
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
