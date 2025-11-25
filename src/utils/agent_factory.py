"""
Agent Factory - Common setup logic for ChatAgent
Shared between CLI (main.py) and API (api_server.py)
"""
import os
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from prompts.system_prompt_builder import build_home_assistant_prompt


def load_mcp_tools(
    mcp_base_url: str,
    mcp_verify_ssl: bool,
    verbose: bool = True
) -> Tuple[List, Dict[str, str], Dict[str, List]]:
    """
    Load MCP tools from servers

    Returns:
        Tuple of (tools, agent_instructions, mcp_tools_by_server)
    """
    tools = []
    agent_instructions = {}
    mcp_tools_by_server = {}

    try:
        from tools.mcp_tools_v2 import (
            get_mcp_tools_by_server,
            get_mcp_server_instructions,
        )

        # Get MCP tools grouped by server
        mcp_tools_by_server = get_mcp_tools_by_server(mcp_base_url, mcp_verify_ssl)

        # Flatten tools for agent use
        for server_tools in mcp_tools_by_server.values():
            tools.extend(server_tools)

        if verbose:
            print(f"MCP Tools Loaded: {len(tools)} tools from {len(mcp_tools_by_server)} servers")
            for server_name, server_tools in mcp_tools_by_server.items():
                print(f"  [{server_name}] {len(server_tools)} tools")

        # Get MCP server instructions
        agent_instructions = get_mcp_server_instructions(mcp_base_url, mcp_verify_ssl)
        if agent_instructions and verbose:
            print(f"Agent Instructions Loaded from {len(agent_instructions)} servers")

    except Exception as e:
        if verbose:
            print(f"Warning: Could not load MCP tools: {e}")
            print("Continuing without MCP tools...")

    return tools, agent_instructions, mcp_tools_by_server


def load_memory_tools(
    memory: Any,
    user_id: str,
    verbose: bool = True
) -> List:
    """
    Load memory search tools

    Returns:
        List of memory tools
    """
    memory_tools = []

    try:
        from tools.memory_tools import create_memory_tools
        memory_tools = create_memory_tools(memory, user_id=user_id)

        if verbose:
            print(f"Memory Tools Loaded: {len(memory_tools)} tools available")
            for tool in memory_tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")

    except Exception as e:
        if verbose:
            print(f"Warning: Could not load memory tools: {e}")

    return memory_tools


def initialize_langfuse(verbose: bool = True) -> bool:
    """
    Initialize Langfuse if enabled

    Returns:
        True if Langfuse was successfully initialized, False otherwise
    """
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

    if not langfuse_enabled:
        return False

    try:
        from utils.langfuse_config import init_langfuse
        init_langfuse()

        if verbose:
            langfuse_host = os.getenv('LANGFUSE_BASE_URL') or os.getenv('LANGFUSE_HOST', 'not configured')
            print(f"Langfuse: Enabled ({langfuse_host})")

        return True

    except Exception as e:
        if verbose:
            print(f"Warning: Could not initialize Langfuse: {e}")
        return False


def build_system_prompt_with_tools(
    agent_instructions: Optional[Dict[str, str]] = None,
    memory_tools: Optional[List] = None
) -> str:
    """
    Build dynamic system prompt with tool instructions

    Args:
        agent_instructions: MCP server instructions
        memory_tools: Memory tools to add to instructions

    Returns:
        Generated system prompt
    """
    # Add memory tool descriptions to agent instructions
    if memory_tools and agent_instructions is not None:
        for tool in memory_tools:
            agent_instructions[tool.name] = tool.description

    # Build dynamic system prompt
    system_prompt = build_home_assistant_prompt(
        agent_instructions=agent_instructions if agent_instructions else None,
    )

    return system_prompt


def create_chat_agent(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    enable_langfuse: bool = False,
    session_id: Optional[str] = None,
    verbose: bool = True
) -> ChatAgent:
    """
    Create ChatAgent with provided or environment configuration

    Args:
        model: LLM model name (defaults to LLM_MODEL env)
        base_url: LLM API base URL (defaults to LLM_BASE_URL env)
        api_key: LLM API key (defaults to LLM_API_KEY env)
        temperature: Temperature (defaults to TEMPERATURE env)
        system_prompt: System prompt (required)
        tools: List of tools to use
        enable_langfuse: Enable Langfuse tracing
        session_id: Session ID for multi-session scenarios
        verbose: Print configuration info

    Returns:
        Configured ChatAgent instance
    """
    # Get configuration from environment if not provided
    model = model or os.getenv("LLM_MODEL", "llama3.2")
    base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = api_key or os.getenv("LLM_API_KEY", "local")
    temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.7"))
    tools = tools or []

    if verbose:
        print(f"\nInitializing Chat Agent...")
        print(f"Model: {model}")
        print(f"Base URL: {base_url}")
        print(f"Temperature: {temperature}")
        print(f"Tools: {len(tools)} available")
        print(f"Langfuse: {'Enabled' if enable_langfuse else 'Disabled'}")
        if system_prompt:
            print(f"System Prompt: {len(system_prompt)} characters")

    # Create agent
    agent = ChatAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        agents=tools,
        use_agent=len(tools) > 0,
        enable_langfuse=enable_langfuse,
        session_id=session_id
    )

    if verbose:
        actual_prompt = agent.get_system_prompt()
        print(f"Agent Ready! System Prompt: {len(actual_prompt)} characters\n")

    return agent


def setup_agent(
    user_id: str = "default",
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = True
) -> Tuple[ChatAgent, Any]:
    """
    Complete agent setup with all components

    This is the main entry point that combines all setup steps:
    1. Load environment variables
    2. Create memory
    3. Load MCP tools
    4. Load memory tools
    5. Initialize Langfuse
    6. Build system prompt
    7. Create agent

    Args:
        user_id: User ID for memory storage
        session_id: Session ID for multi-session scenarios
        model: Override LLM model from environment
        temperature: Override temperature from environment
        verbose: Print setup information

    Returns:
        Tuple of (ChatAgent, Memory instance)
    """
    # Load environment
    load_dotenv()

    # Get configuration
    use_mcp_tools = os.getenv("USE_MCP_TOOLS", "false").lower() == "true"
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:22001")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

    # Create memory (imported from memory_tools)
    from tools.memory_tools import create_memory
    memory = create_memory()

    # Load tools
    all_tools = []
    agent_instructions = {}

    # Load MCP tools if enabled
    if use_mcp_tools:
        mcp_tools, agent_instructions, _ = load_mcp_tools(
            mcp_base_url, mcp_verify_ssl, verbose
        )
        all_tools.extend(mcp_tools)

    # Load memory tools
    memory_tools = load_memory_tools(memory, user_id, verbose)
    all_tools.extend(memory_tools)

    # Initialize Langfuse
    langfuse_initialized = False
    if langfuse_enabled:
        langfuse_initialized = initialize_langfuse(verbose)

    # Build system prompt
    system_prompt = build_system_prompt_with_tools(agent_instructions, memory_tools)

    # Create agent
    agent = create_chat_agent(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        tools=all_tools,
        enable_langfuse=langfuse_initialized,
        session_id=session_id,
        verbose=verbose
    )

    return agent, memory
