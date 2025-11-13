"""
Memory search tools powered by mem0.
Allows searching through conversation history and memories.
"""
from typing import List, Optional, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MemorySearchInput(BaseModel):
    """Input schema for memory search tool."""
    query: str = Field(
        description="Search query to find relevant memories and past conversations"
    )
    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of memories to return (default: 5)"
    )


class MemorySearchTool(BaseTool):
    """Tool for searching through conversation memories using mem0."""

    name: str = "search_memory"
    description: str = (
        "Search through past conversations and memories. "
        "Use this tool to recall what was discussed before, find information "
        "from previous interactions, or remember user preferences and context. "
        "Input should be a natural language query describing what you're looking for."
    )
    args_schema: type[BaseModel] = MemorySearchInput
    memory: Any = Field(default=None, exclude=True)
    user_id: str = Field(default="alex", exclude=True)

    def _run(self, query: str, limit: int = 5) -> str:
        """
        Execute memory search.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            Formatted string with search results
        """
        if self.memory is None:
            return "Error: Memory system not initialized"

        try:
            # Search memories using mem0
            response = self.memory.search(
                query=query,
                user_id=self.user_id,
                limit=limit
            )

            # mem0 returns dict with 'results' key
            results = response.get("results", []) if isinstance(response, dict) else []

            if not results or len(results) == 0:
                return f"No memories found for query: '{query}'"

            # Format results
            formatted_results = [f"Found {len(results)} relevant memories:\n"]

            for idx, result in enumerate(results, 1):
                memory_text = result.get("memory", "")
                score = result.get("score", 0.0)
                created_at = result.get("created_at", "")

                formatted_results.append(
                    f"{idx}. [Score: {score:.3f}] {memory_text}"
                )
                if created_at:
                    formatted_results.append(f"   (Created: {created_at})")

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error searching memories: {str(e)}"

    async def _arun(self, query: str, limit: int = 5) -> str:
        """Async version - calls sync implementation."""
        return self._run(query, limit)


class MemoryGetAllInput(BaseModel):
    """Input schema for get all memories tool."""
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of memories to return (default: 10)"
    )


class MemoryGetAllTool(BaseTool):
    """Tool for retrieving all recent memories."""

    name: str = "get_all_memories"
    description: str = (
        "Retrieve recent memories and conversation history. "
        "Use this to get a general overview of past interactions "
        "without a specific search query."
    )
    args_schema: type[BaseModel] = MemoryGetAllInput
    memory: Any = Field(default=None, exclude=True)
    user_id: str = Field(default="alex", exclude=True)

    def _run(self, limit: int = 10) -> str:
        """
        Get all recent memories.

        Args:
            limit: Maximum number of memories to return

        Returns:
            Formatted string with all memories
        """
        if self.memory is None:
            return "Error: Memory system not initialized"

        try:
            # Get all memories for user
            response = self.memory.get_all(
                user_id=self.user_id,
                limit=limit
            )

            # mem0 returns dict with 'results' key
            results = response.get("results", []) if isinstance(response, dict) else []

            if not results or len(results) == 0:
                return "No memories found"

            # Format results
            formatted_results = [f"Retrieved {len(results)} recent memories:\n"]

            for idx, result in enumerate(results, 1):
                memory_text = result.get("memory", "")
                created_at = result.get("created_at", "")

                formatted_results.append(f"{idx}. {memory_text}")
                if created_at:
                    formatted_results.append(f"   (Created: {created_at})")

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    async def _arun(self, limit: int = 10) -> str:
        """Async version - calls sync implementation."""
        return self._run(limit)


def create_memory_tools(memory: Any, user_id: str = "alex") -> List[BaseTool]:
    """
    Create memory search tools for LangChain.

    Args:
        memory: mem0 Memory instance
        user_id: User ID for memory operations (default: "alex")

    Returns:
        List of memory search tools
    """
    return [
        MemorySearchTool(memory=memory, user_id=user_id),
        MemoryGetAllTool(memory=memory, user_id=user_id)
    ]
