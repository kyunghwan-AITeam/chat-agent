"""
LangChain tools powered by MCP servers using official langchain-mcp-adapters.
"""
from typing import List, Optional, Dict
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import httpx
import ssl


async def get_mcp_server_instructions_async(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> Dict[str, str]:
    """
    Get instructions from all MCP servers.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Dictionary mapping server names to their instructions
    """
    # Create custom httpx client factory with SSL verification disabled if needed
    httpx_client_factory = None
    if not verify_ssl:
        def create_httpx_client(**kwargs):
            return httpx.AsyncClient(verify=False, **kwargs)
        httpx_client_factory = create_httpx_client

    # Configure MCP servers
    server_config = {
        "weather": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/weather/mcp",
        },
        "web_search": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/search/mcp",
        }
    }

    # Add httpx_client_factory to each server config if needed
    if httpx_client_factory:
        for server in server_config.values():
            server["httpx_client_factory"] = httpx_client_factory

    instructions = {}
    try:
        # Create multi-server MCP client
        client = MultiServerMCPClient(server_config)

        # Get server info from each server using session
        for server_name in server_config.keys():
            try:
                # Create a session for this server to get server info
                async with client.session(server_name, auto_initialize=False) as session:
                    # Initialize the session to get server info
                    init_result = await session.initialize()

                    # Extract instructions from initialize result
                    if init_result and hasattr(init_result, 'instructions'):
                        if init_result.instructions:
                            instructions[server_name] = init_result.instructions
            except Exception as e:
                print(f"Warning: Could not get instructions from {server_name}: {e}")
                continue

        return instructions
    except Exception as e:
        import traceback
        print(f"Error getting MCP server instructions: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return {}


async def get_mcp_tools_by_server_async(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> Dict[str, List[BaseTool]]:
    """
    Get MCP tools grouped by server name.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Dictionary mapping server names to their tools
    """
    # Create custom httpx client factory with SSL verification disabled if needed
    httpx_client_factory = None
    if not verify_ssl:
        def create_httpx_client(**kwargs):
            return httpx.AsyncClient(verify=False, **kwargs)
        httpx_client_factory = create_httpx_client

    # Configure MCP servers
    server_config = {
        "weather": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/weather/mcp",
        },
        "web_search": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/search/mcp",
        }
    }

    # Add httpx_client_factory to each server config if needed
    if httpx_client_factory:
        for server in server_config.values():
            server["httpx_client_factory"] = httpx_client_factory

    tools_by_server = {}
    try:
        # Create multi-server MCP client
        client = MultiServerMCPClient(server_config)

        # Get tools from each server separately
        for server_name in server_config.keys():
            try:
                tools = await client.get_tools(server_name=server_name)
                tools_by_server[server_name] = tools
            except Exception as e:
                print(f"Warning: Could not get tools from {server_name}: {e}")
                tools_by_server[server_name] = []

        return tools_by_server
    except Exception as e:
        import traceback
        print(f"Error getting MCP tools by server: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return {}


async def create_mcp_tools_async(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> List[BaseTool]:
    """
    Create MCP tools using the official langchain-mcp-adapters library.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        List of LangChain Tools from MCP servers
    """
    # Create custom httpx client factory with SSL verification disabled if needed
    httpx_client_factory = None
    if not verify_ssl:
        # Create a factory function that returns httpx client with SSL verification disabled
        def create_httpx_client(**kwargs):
            # Accept any kwargs that mcp library might pass (like headers)
            return httpx.AsyncClient(verify=False, **kwargs)

        httpx_client_factory = create_httpx_client

    # Configure MCP servers
    server_config = {
        "weather": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/weather/mcp",
        },
        "web_search": {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/search/mcp",
        }
    }

    # Add httpx_client_factory to each server config if needed
    if httpx_client_factory:
        for server in server_config.values():
            server["httpx_client_factory"] = httpx_client_factory

    try:
        # Create multi-server MCP client
        client = MultiServerMCPClient(server_config)

        # Get all tools from all configured servers
        tools = await client.get_tools()

        return tools
    except Exception as e:
        import traceback
        print(f"Error creating MCP tools: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return []


def get_mcp_tools_by_server(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> Dict[str, List[BaseTool]]:
    """
    Synchronous wrapper for getting MCP tools grouped by server.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Dictionary mapping server names to their tools
    """
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    get_mcp_tools_by_server_async(mcp_base_url, verify_ssl)
                )
                return future.result()
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(get_mcp_tools_by_server_async(mcp_base_url, verify_ssl))
    except RuntimeError:
        # If get_event_loop fails, just use asyncio.run
        return asyncio.run(get_mcp_tools_by_server_async(mcp_base_url, verify_ssl))


def get_mcp_server_instructions(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> Dict[str, str]:
    """
    Synchronous wrapper for getting MCP server instructions.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        Dictionary mapping server names to their instructions
    """
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    get_mcp_server_instructions_async(mcp_base_url, verify_ssl)
                )
                return future.result()
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(get_mcp_server_instructions_async(mcp_base_url, verify_ssl))
    except RuntimeError:
        # If get_event_loop fails, just use asyncio.run
        return asyncio.run(get_mcp_server_instructions_async(mcp_base_url, verify_ssl))


def create_mcp_tools(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False
) -> List[BaseTool]:
    """
    Synchronous wrapper for creating MCP tools.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        List of LangChain Tools from MCP servers
    """
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    create_mcp_tools_async(mcp_base_url, verify_ssl)
                )
                return future.result()
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(create_mcp_tools_async(mcp_base_url, verify_ssl))
    except RuntimeError:
        # If get_event_loop fails, just use asyncio.run
        return asyncio.run(create_mcp_tools_async(mcp_base_url, verify_ssl))


async def create_mcp_tools_with_session(
    mcp_base_url: str = "https://localhost:22000",
    verify_ssl: bool = False,
    server_name: str = "weather"
) -> List[BaseTool]:
    """
    Create MCP tools with explicit session management for stateful operations.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates
        server_name: Name of the specific server to connect to

    Returns:
        List of LangChain Tools from the specified MCP server
    """
    server_config = {
        server_name: {
            "transport": "streamable_http",
            "url": f"{mcp_base_url}/{server_name}/mcp",
        }
    }

    try:
        client = MultiServerMCPClient(server_config)

        # Use session context for stateful operations
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)
            return tools
    except Exception as e:
        print(f"Error creating MCP tools with session: {e}")
        return []
