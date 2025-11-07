"""
Main entry point for LangChain Chat Agent (Ollama)
"""
import os
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from agents.chat_agent import ChatAgent
from prompts import HOME_ASSISTANT_PROMPT


def main():
    """Main function to run the chat agent"""
    # Load environment variables
    load_dotenv()

    # Get Ollama configuration from environment
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))

    # Get MCP configuration
    use_mcp_tools = os.getenv("USE_MCP_TOOLS", "false").lower() == "true"
    mcp_base_url = os.getenv("MCP_BASE_URL", "https://localhost:22000")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    # Initialize MCP tools if enabled
    tools = []
    if use_mcp_tools:
        try:
            from tools.mcp_tools import create_all_mcp_tools
            tools = create_all_mcp_tools(mcp_base_url, mcp_verify_ssl)
            print(f"MCP Tools Loaded: {len(tools)} tools available")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")
        except Exception as e:
            print(f"Warning: Could not load MCP tools: {e}")
            print("Continuing without MCP tools...")
            tools = []

    # Initialize agent with Home Assistant prompt
    print(f"\nInitializing Home Assistant Chat Agent with Ollama...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Temperature: {temperature}")
    print(f"MCP Tools: {'Enabled' if use_mcp_tools and tools else 'Disabled'}")

    # Debug: Check if prompt is loaded
    print(f"\nSystem Prompt Loaded: {len(HOME_ASSISTANT_PROMPT)} characters")
    print(f"First 100 chars: {HOME_ASSISTANT_PROMPT[:100]}...")

    agent = ChatAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key="ollama",
        system_prompt=HOME_ASSISTANT_PROMPT,
        tools=tools,
        use_agent=use_mcp_tools and len(tools) > 0
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
    if use_mcp_tools and tools:
        print("I also have access to weather information and web search capabilities via MCP!")
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
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
