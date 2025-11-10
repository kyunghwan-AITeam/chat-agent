"""
LangChain tools powered by MCP servers.
"""
from langchain_core.tools import StructuredTool
from typing import List, Optional
import asyncio
from .mcp_client import MCPClient, MCPToolWrapper


def create_weather_tool(mcp_base_url: str = "https://localhost:22000", verify_ssl: bool = False) -> StructuredTool:
    """
    Create a weather tool that connects to the MCP weather server.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        LangChain Tool for weather queries
    """
    mcp_client = MCPClient(mcp_base_url, verify_ssl=verify_ssl)
    wrapper = MCPToolWrapper(
        mcp_client=mcp_client,
        service="weather",
        tool_name="get_current_weather",
        tool_description="Get current weather and short-term forecast (up to 7 days) for a location using Open-Meteo API"
    )

    def get_weather(location: str) -> str:
        """Get current weather and short-term forecast for a location.

        Args:
            location: Name of the location in ENGLISH (e.g., 'Seoul', 'New York', 'Tokyo', 'London')

        Returns:
            Weather information including current conditions and daily forecast (up to 3 days by default)
        """
        return wrapper.run(location=location)

    return StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description=(
            "Useful for getting current weather and short-term forecast (up to 7 days) for a specific location. "
            "IMPORTANT: Input location must be in ENGLISH (e.g., 'Seoul', 'New York', 'Tokyo', 'London'). "
            "Returns current weather conditions (temperature, wind, weather code) and daily forecast "
            "(max/min temperature, precipitation probability) for the next few days."
        )
    )


def create_web_search_tool(mcp_base_url: str = "https://localhost:22000", verify_ssl: bool = False) -> StructuredTool:
    """
    Create a web search tool that connects to the MCP web search server.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        LangChain Tool for web searches
    """
    mcp_client = MCPClient(mcp_base_url, verify_ssl=verify_ssl)
    wrapper = MCPToolWrapper(
        mcp_client=mcp_client,
        service="search",
        tool_name="search_web",
        tool_description="Search the web using Brave Search API"
    )

    def search_web(query: str) -> str:
        """Search the web for information.

        Args:
            query: The search query string

        Returns:
            Search results as a string
        """
        return wrapper.run(query=query, count=5)

    return StructuredTool.from_function(
        func=search_web,
        name="search_web",
        description=(
            "Useful for searching the web to find current information, news, or answers to questions. "
            "Input should be a search query string. "
            "Returns relevant search results from the web."
        )
    )


def create_all_mcp_tools(mcp_base_url: str = "https://localhost:22000", verify_ssl: bool = False) -> List[StructuredTool]:
    """
    Create all available MCP tools.

    Args:
        mcp_base_url: Base URL of the MCP server
        verify_ssl: Whether to verify SSL certificates

    Returns:
        List of all LangChain Tools
    """
    tools = []

    try:
        weather_tool = create_weather_tool(mcp_base_url, verify_ssl)
        tools.append(weather_tool)
    except Exception as e:
        print(f"Warning: Could not create weather tool: {e}")

    try:
        search_tool = create_web_search_tool(mcp_base_url, verify_ssl)
        tools.append(search_tool)
    except Exception as e:
        print(f"Warning: Could not create web search tool: {e}")

    return tools
