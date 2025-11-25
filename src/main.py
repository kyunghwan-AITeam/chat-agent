"""
Main entry point for LangChain Chat Agent (CLI Interface)
"""
import os
import sys
import logging
import asyncio
import uuid
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from utils.agent_factory import setup_agent


# Load environment variables
load_dotenv()

# Suppress OpenTelemetry error messages when Langfuse server is unavailable
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)

# CLI session directory
CLI_SESSION_DIR = ".sessions"


def get_session_file(session_name: str = "default") -> str:
    """Get session file path for a given session name"""
    os.makedirs(CLI_SESSION_DIR, exist_ok=True)
    return os.path.join(CLI_SESSION_DIR, f"{session_name}.session")


def get_or_create_session_id(session_name: str = "default") -> str:
    """
    Get existing session ID or create a new one.
    Session ID is saved to file for persistence across restarts.

    Args:
        session_name: Name of the session (e.g., "work", "personal", "project1")
    """
    session_file = get_session_file(session_name)

    # Try to load existing session ID
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                session_id = f.read().strip()
                if session_id:
                    return session_id
        except Exception as e:
            print(f"Warning: Could not read session file: {e}")

    # Create new session ID
    session_id = f"cli-{session_name}-{uuid.uuid4()}"

    # Save to file
    try:
        with open(session_file, 'w') as f:
            f.write(session_id)
    except Exception as e:
        print(f"Warning: Could not save session file: {e}")

    return session_id


def create_new_session(session_name: str = "default") -> str:
    """Create a new session ID and save it"""
    session_file = get_session_file(session_name)
    session_id = f"cli-{session_name}-{uuid.uuid4()}"
    try:
        with open(session_file, 'w') as f:
            f.write(session_id)
        print(f"New session created: {session_id[:30]}...")
    except Exception as e:
        print(f"Warning: Could not save session file: {e}")
    return session_id


def list_sessions() -> list:
    """List all available sessions"""
    if not os.path.exists(CLI_SESSION_DIR):
        return []

    sessions = []
    for file in os.listdir(CLI_SESSION_DIR):
        if file.endswith('.session'):
            session_name = file[:-8]  # Remove .session extension
            sessions.append(session_name)
    return sorted(sessions)


async def async_chat_stream(agent, user_input):
    """Helper to stream chat responses asynchronously"""
    full_content = ""
    async for chunk in agent.chat_stream(user_input):
        print(chunk, end="", flush=True)
        full_content += chunk
    return full_content


def main():
    """Main function to run the chat agent CLI"""

    # Parse command line arguments
    session_name = "default"
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Usage: python src/main.py [session_name]")
            print("\nArguments:")
            print("  session_name    Name of the session (default: 'default')")
            print("                  Examples: 'work', 'personal', 'project1'")
            print("\nExamples:")
            print("  python src/main.py              # Use default session")
            print("  python src/main.py work         # Use 'work' session")
            print("  python src/main.py project1     # Use 'project1' session")
            print("\nEach session maintains separate conversation history.")
            sys.exit(0)
        session_name = sys.argv[1]

    # Get or create session ID
    session_id = get_or_create_session_id(session_name)
    session_store_type = os.getenv("SESSION_STORE_TYPE", "memory")

    # Setup agent with all components (MCP, Memory, Langfuse, etc.)
    agent, memory = setup_agent(
        user_id="cli-user",
        session_id=session_id,
        verbose=True
    )

    # Get tools for welcome message
    tools = agent.tools if hasattr(agent, 'tools') else []
    use_mcp_tools = os.getenv("USE_MCP_TOOLS", "false").lower() == "true"

    # Check if session has history
    history_count = len(agent.chat_history)

    # Welcome message
    print("=" * 60)
    print("Chat Agent CLI")
    print("=" * 60)
    print(f"Session Name: {session_name}")
    print(f"Session ID: {session_id[:30]}...")
    print(f"Session Store: {session_store_type}")
    if history_count > 0:
        print(f"Loaded History: {history_count} messages")
    print("=" * 60)
    print("\nCommands:")
    print("  - 'reset': Clear conversation history")
    print("  - 'new': Start a new session (keep same name)")
    print("  - 'session': Show current session info")
    print("  - 'sessions': List all available sessions")
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
                print("✓ Conversation history cleared.\n")
                continue

            if user_input.lower() == "new":
                # Create new session and recreate agent
                print("\nCreating new session...")
                session_id = create_new_session(session_name)
                agent, memory = setup_agent(
                    user_id="cli-user",
                    session_id=session_id,
                    verbose=False
                )
                print(f"✓ New session started: {session_id[:30]}...")
                print(f"✓ History: {len(agent.chat_history)} messages\n")
                continue

            if user_input.lower() == "session":
                from utils.session_store import session_store
                print(f"\n=== Session Information ===")
                print(f"Session Name: {session_name}")
                print(f"Session ID: {session_id}")
                print(f"Store Type: {session_store_type}")
                print(f"History Size: {len(agent.chat_history)} messages")
                if session_store_type == "memory":
                    active_count = session_store.get_active_session_count()
                    if active_count >= 0:
                        print(f"Active Sessions: {active_count}")
                print("=" * 50 + "\n")
                continue

            if user_input.lower() == "sessions":
                print(f"\n=== Available Sessions ===")
                sessions = list_sessions()
                if sessions:
                    for i, sess_name in enumerate(sessions, 1):
                        current = " (current)" if sess_name == session_name else ""
                        print(f"{i}. {sess_name}{current}")
                else:
                    print("No sessions found")
                print(f"\nTo switch sessions, restart with:")
                print(f"  python src/main.py <session_name>")
                print("=" * 50 + "\n")
                continue

            if user_input.lower() == "prompt":
                print(f"\n=== Current System Prompt ===")
                print(agent.get_system_prompt())
                print("=" * 50 + "\n")
                continue

            # Get response from agent (streaming)
            print("\nAgent: ", end="", flush=True)
            full_content = asyncio.run(async_chat_stream(agent, user_input))
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
