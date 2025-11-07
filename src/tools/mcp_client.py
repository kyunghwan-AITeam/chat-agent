"""
MCP Client for connecting to MCP servers via HTTP transport.
"""
import asyncio
import httpx
from typing import Any, Dict, List, Optional


class MCPClient:
    """Client for interacting with MCP servers over HTTP."""

    def __init__(self, base_url: str, verify_ssl: bool = False):
        """
        Initialize MCP client.

        Args:
            base_url: Base URL of the MCP server (e.g., https://localhost:22000)
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.verify_ssl = verify_ssl
        self.client = httpx.AsyncClient(verify=verify_ssl, timeout=30.0)
        self.sessions: Dict[str, str] = {}  # service -> session_id

    async def _ensure_session(self, service: str) -> str:
        """
        Ensure we have a valid session for the service.

        Args:
            service: Service name (e.g., 'weather', 'search')

        Returns:
            Session ID
        """
        if service in self.sessions:
            return self.sessions[service]

        url = f"{self.base_url}/{service}/mcp"

        # Step 1: Acquire session with noop
        noop_payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "noop",
            "params": {}
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }

        try:
            response = await self.client.post(url, json=noop_payload, headers=headers)

            # The server returns session ID in header even if the request fails
            # because we don't have a session ID yet
            session_id = response.headers.get("mcp-session-id")
            if not session_id:
                raise Exception(f"Server at {url} did not return mcp-session-id")

            # Don't raise for 400 if we got a session ID - this is expected for the first request
            if response.status_code != 400:
                response.raise_for_status()

            # Step 2: Initialize session
            init_payload = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "capabilities": {},
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "chat-agent-mcp-client",
                        "version": "0.1.0"
                    }
                }
            }

            init_headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "mcp-session-id": session_id
            }

            init_response = await self.client.post(url, json=init_payload, headers=init_headers)
            init_response.raise_for_status()

            # Parse initialization response (SSE format)
            # Just check that it didn't error; we don't need the response content
            init_content = init_response.text
            if init_content:
                import json
                for line in init_content.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('data:'):
                        json_str = line[5:].strip()
                        try:
                            result = json.loads(json_str)
                            if "error" in result:
                                raise Exception(f"MCP initialization error: {result['error']}")
                        except json.JSONDecodeError:
                            pass

            # Store session
            self.sessions[service] = session_id
            return session_id

        except httpx.HTTPError as e:
            raise Exception(f"HTTP Error during session initialization: {str(e)}")

    async def call_tool(self, service: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.

        Args:
            service: Service name (e.g., 'weather', 'search')
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Result from the tool
        """
        # Ensure we have a session
        session_id = await self._ensure_session(service)

        url = f"{self.base_url}/{service}/mcp"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id
        }

        print(f"\n[MCP] Calling {service}/{tool_name} with args: {arguments}")

        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            print(f"[MCP] Response status: {response.status_code}")
            print(f"[MCP] Response headers: {dict(response.headers)}")

            # Handle Server-Sent Events (SSE) streaming response
            content = response.text
            print(f"[MCP] Response content length: {len(content)}")
            print(f"[MCP] Response content preview: {content[:500]}")

            if content:
                import json
                # Parse SSE format: "event: message\ndata: {...}"
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('data:'):
                        json_str = line[5:].strip()  # Remove "data:" prefix
                        try:
                            result = json.loads(json_str)
                            print(f"[MCP] Parsed result: {result}")
                            if "error" in result:
                                raise Exception(f"MCP Error: {result['error']}")
                            # Extract the actual result content
                            final_result = result.get("result", {})
                            print(f"[MCP] Final result: {final_result}")
                            return final_result
                        except json.JSONDecodeError as e:
                            print(f"[MCP] Warning: Failed to parse JSON: {e}")
                            print(f"[MCP] JSON string was: {json_str[:200]}")
                            continue

            print("[MCP] No content in response, returning empty dict")
            return {}

        except httpx.HTTPError as e:
            print(f"[MCP] HTTP Error: {e}")
            raise Exception(f"HTTP Error calling MCP tool: {str(e)}")

    async def list_tools(self, service: str) -> List[Dict[str, Any]]:
        """
        List available tools from an MCP server.

        Args:
            service: Service name (e.g., 'weather', 'search')

        Returns:
            List of available tools
        """
        # Ensure we have a session
        session_id = await self._ensure_session(service)

        url = f"{self.base_url}/{service}/mcp"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id
        }

        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            # Handle Server-Sent Events (SSE) streaming response
            content = response.text
            if content:
                import json
                # Parse SSE format: "event: message\ndata: {...}"
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('data:'):
                        json_str = line[5:].strip()  # Remove "data:" prefix
                        try:
                            result = json.loads(json_str)
                            if "error" in result:
                                raise Exception(f"MCP Error: {result['error']}")
                            return result.get("result", {}).get("tools", [])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON: {e}")
                            continue

            return []

        except httpx.HTTPError as e:
            raise Exception(f"HTTP Error listing MCP tools: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Only try to close if there's an event loop running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule close for later
                asyncio.create_task(self.close())
            else:
                # Close synchronously
                loop.run_until_complete(self.close())
        except Exception:
            # Silently ignore cleanup errors
            pass


class MCPToolWrapper:
    """Wrapper to make MCP tools compatible with LangChain."""

    def __init__(self, mcp_client: MCPClient, service: str, tool_name: str, tool_description: str):
        """
        Initialize tool wrapper.

        Args:
            mcp_client: MCP client instance
            service: Service name
            tool_name: Name of the tool
            tool_description: Description of the tool
        """
        self.mcp_client = mcp_client
        self.service = service
        self.tool_name = tool_name
        self.description = tool_description

    def run(self, **kwargs) -> str:
        """
        Run the tool synchronously.

        Args:
            **kwargs: Arguments to pass to the tool

        Returns:
            String result from the tool
        """
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()
            # We're in an async context, run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._async_run(**kwargs))
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run
            return asyncio.run(self._async_run(**kwargs))

    async def _async_run(self, **kwargs) -> str:
        """Async implementation of run."""
        try:
            result = await self.mcp_client.call_tool(self.service, self.tool_name, kwargs)
            # Convert result to string for LangChain
            if isinstance(result, dict):
                import json
                return json.dumps(result, indent=2, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
