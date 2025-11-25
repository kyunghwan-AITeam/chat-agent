"""
OpenAI-compatible API Server for Chat Agent
Implements /v1/chat/completions endpoint compatible with OpenAI API
"""
import os
import sys
import uuid
import time
import logging
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from agents.chat_agent import ChatAgent
from utils.agent_factory import (
    load_mcp_tools,
    load_memory_tools,
    initialize_langfuse,
    build_system_prompt_with_tools,
    create_chat_agent
)
from tools.memory_tools import create_memory
import uvicorn

from langfuse import get_client, propagate_attributes

# Load environment variables
load_dotenv()

# Suppress OpenTelemetry error messages when Langfuse server is unavailable
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)


# Validate required environment variables on startup
def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = {
        "LLM_MODEL": "LLM model name (e.g., llama3.2, google/gemma-3-27b-it)",
        "LLM_BASE_URL": "LLM API base URL (e.g., http://localhost:11434/v1)"
    }

    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"  - {var}: {description}")

    if missing_vars:
        print("\n" + "="*60)
        print("ERROR: Missing required environment variables")
        print("="*60)
        print("\nThe following variables must be set in .env file:\n")
        print("\n".join(missing_vars))
        print("\nPlease create a .env file with these variables.")
        print("You can copy from .env.example:\n")
        print("  cp .env.example .env")
        print("\nThen edit .env and set the appropriate values.")
        print("="*60 + "\n")
        sys.exit(1)

    # Validate LLM_BASE_URL format
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url.startswith(("http://", "https://")):
        print("\n" + "="*60)
        print("ERROR: Invalid LLM_BASE_URL")
        print("="*60)
        print(f"\nLLM_BASE_URL must start with http:// or https://")
        print(f"Current value: {base_url}")
        print("="*60 + "\n")
        sys.exit(1)

    # Print configuration to stderr (so it's visible even with uvicorn)
    print("\n" + "="*60, file=sys.stderr)
    print("Configuration Loaded", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"LLM_MODEL: {os.getenv('LLM_MODEL')}", file=sys.stderr)
    print(f"LLM_BASE_URL: {os.getenv('LLM_BASE_URL')}", file=sys.stderr)
    print(f"TEMPERATURE: {os.getenv('TEMPERATURE', '0.7')}", file=sys.stderr)
    print(f"USE_MCP_TOOLS: {os.getenv('USE_MCP_TOOLS', 'false')}", file=sys.stderr)

    # Langfuse configuration
    langfuse_enabled = os.getenv('LANGFUSE_ENABLED', 'false').lower() == 'true'
    if langfuse_enabled and os.getenv('LANGFUSE_HOST'):
        print(f"LANGFUSE_ENABLED: true", file=sys.stderr)
        print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}", file=sys.stderr)
        print(f"LANGFUSE_ENVIRONMENT: {os.getenv('LANGFUSE_ENVIRONMENT', 'development')}", file=sys.stderr)
    else:
        print(f"LANGFUSE_ENABLED: false", file=sys.stderr)

    print(f"API_HOST: {os.getenv('API_HOST', '0.0.0.0')}", file=sys.stderr)
    print(f"API_PORT: {os.getenv('API_PORT', '23000')}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)


# Validate environment on module load
validate_environment()

# Initialize Langfuse on module load
LANGFUSE_ENABLED = initialize_langfuse(verbose=True)

# Create shared memory instance
MEMORY = create_memory()

# FastAPI app
app = FastAPI(
    title="Chat Agent API",
    description="OpenAI-compatible API for LangChain Chat Agent with LLM",
    version="1.0.0"
)


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use (defaults to LLM_MODEL from .env)")
    messages: List[Message]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperature (defaults to TEMPERATURE from .env)")
    stream: Optional[bool] = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=0.9)
    n: Optional[int] = Field(default=1)
    stop: Optional[Union[str, List[str]]] = Field(default=None)


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


# Session store now handles chat history persistence
# No need for global agent_sessions dictionary anymore


def create_agent_for_session(session_id: str, model: Optional[str] = None, temperature: Optional[float] = None) -> ChatAgent:
    """
    Create a new agent for the session.
    Chat history is automatically loaded from session store.
    """
    # Get MCP configuration
    use_mcp_tools = os.getenv("USE_MCP_TOOLS", "false").lower() == "true"
    mcp_base_url = os.getenv("MCP_BASE_URL", "http://localhost:22001")
    mcp_verify_ssl = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"

    # Load tools
    all_tools = []
    agent_instructions = {}

    # Load MCP tools if enabled
    if use_mcp_tools:
        mcp_tools, agent_instructions, _ = load_mcp_tools(
            mcp_base_url, mcp_verify_ssl, verbose=False
        )
        all_tools.extend(mcp_tools)

    # Load memory tools
    memory_tools = load_memory_tools(MEMORY, session_id, verbose=False)
    all_tools.extend(memory_tools)

    # Build system prompt
    system_prompt = build_system_prompt_with_tools(agent_instructions, memory_tools)

    # Create agent (chat history automatically loaded from session store)
    agent = create_chat_agent(
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        tools=all_tools,
        enable_langfuse=LANGFUSE_ENABLED,
        session_id=session_id,
        verbose=False
    )

    return agent


def convert_messages_to_chat_history(messages: List[Message], agent: ChatAgent):
    """Convert OpenAI message format to LangChain chat history"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    agent.chat_history.clear()

    for msg in messages[:-1]:  # Exclude the last message (current user input)
        if msg.role == "user":
            agent.chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            agent.chat_history.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            # System messages are handled by the system prompt
            pass


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chat Agent API Server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    # Get configured LLM model (already validated)
    model = os.getenv("LLM_MODEL")

    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "LLM"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI compatible)

    Supports both streaming and non-streaming responses.
    Use session_id in messages to maintain conversation history.
    """
    try:
        # Extract session ID from messages if provided (custom extension)
        session_id = None
        for msg in request.messages:
            if msg.role == "system" and "session_id:" in msg.content:
                # Format: "session_id: <uuid>"
                parts = msg.content.split("session_id:")
                if len(parts) > 1:
                    session_id = parts[1].strip().split()[0]
                    break

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create agent for session (chat history automatically loaded)
        agent = create_agent_for_session(session_id, request.model, request.temperature)

        # Get the last user message
        user_message = request.messages[-1].content if request.messages else ""

        # Generate completion ID
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        langfuse = get_client()
        with langfuse.start_as_current_span(name="langchain-call"):
            with propagate_attributes(
                session_id=session_id,
                tags=["tag-1", "tag-2"]
            ):
                if request.stream:
                    # Streaming response
                    async def generate_stream():
                        first_chunk = True
                        full_content = ""

                        # Store user message in memory
                        try:
                            MEMORY.add([{"role": "user", "content": user_message}], user_id=session_id)
                        except Exception as e:
                            print(f"Warning: Could not save user message to memory: {e}")

                        # Get model name with fallback
                        model_name = request.model or os.getenv("LLM_MODEL", "unknown")

                        async for chunk_text in agent.chat_stream(user_message):
                            full_content += chunk_text
                            if first_chunk:
                                # First chunk with role
                                chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=created_time,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionResponseStreamChoice(
                                            index=0,
                                            delta={"role": "assistant", "content": chunk_text},
                                            finish_reason=None
                                        )
                                    ]
                                )
                                first_chunk = False
                            else:
                                # Subsequent chunks with content only
                                chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=created_time,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionResponseStreamChoice(
                                            index=0,
                                            delta={"content": chunk_text},
                                            finish_reason=None
                                        )
                                    ]
                                )

                            yield f"data: {chunk.model_dump_json()}\n\n"

                        # Store assistant response in memory
                        try:
                            MEMORY.add([{"role": "assistant", "content": full_content}], user_id=session_id)
                        except Exception as e:
                            print(f"Warning: Could not save assistant response to memory: {e}")

                        # Final chunk with finish_reason
                        final_chunk = ChatCompletionStreamResponse(
                            id=completion_id,
                            created=created_time,
                            model=model_name,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=0,
                                    delta={},
                                    finish_reason="stop"
                                )
                            ]
                        )
                        yield f"data: {final_chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Session-ID": session_id  # Custom header for session tracking
                        }
                    )
                else:
                    # Non-streaming response
                    # Store user message in memory
                    try:
                        MEMORY.add([{"role": "user", "content": user_message}], user_id=session_id)
                    except Exception as e:
                        print(f"Warning: Could not save user message to memory: {e}")

                    response_text = await agent.chat(user_message)

                    # Store assistant response in memory
                    try:
                        MEMORY.add([{"role": "assistant", "content": response_text}], user_id=session_id)
                    except Exception as e:
                        print(f"Warning: Could not save assistant response to memory: {e}")

                    # Get model name with fallback
                    model_name = request.model or os.getenv("LLM_MODEL", "unknown")

                    response = ChatCompletionResponse(
                        id=completion_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            ChatCompletionResponseChoice(
                                index=0,
                                message=Message(role="assistant", content=response_text),
                                finish_reason="stop"
                            )
                        ]
                    )

                    return JSONResponse(
                        content=response.model_dump(),
                        headers={"X-Session-ID": session_id}  # Custom header for session tracking
                    )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its chat history"""
    from utils.session_store import session_store

    success = session_store.delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/v1/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset chat history for a session"""
    from utils.session_store import session_store

    try:
        history = session_store.get_history(session_id)
        history.clear()
        return {"message": f"Session {session_id} reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Background task to cleanup expired sessions"""
    import asyncio
    from utils.session_store import session_store

    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            cleaned = session_store.cleanup_expired_sessions()
            if cleaned > 0:
                print(f"[Background] Cleaned up {cleaned} expired sessions")

    # Only run cleanup for memory-based storage
    if session_store.store_type == "memory":
        asyncio.create_task(cleanup_loop())
        print("[Background] Session cleanup task started (5 minute interval)")


def main():
    """Run the API server"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "21000"))

    print(f"\n{'='*60}")
    print(f"Chat Agent API Server")
    print(f"{'='*60}")
    print(f"Server: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"OpenAI Endpoint: http://{host}:{port}/v1/chat/completions")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
