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
from fastapi import FastAPI, HTTPException, Header
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
import httpx

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

# Store for authorized sessions (session_id -> creation_time)
# 나중에 인증 시스템 추가 시 사용자 정보도 포함
AUTHORIZED_SESSIONS: Dict[str, Dict[str, Any]] = {}

# WebSocket server URL
WEBSOCKET_SERVER_URL = os.getenv("WEBSOCKET_SERVER_URL", "http://localhost:21000")

# HTTP client for WebSocket server communication
http_client = httpx.AsyncClient(timeout=5.0)

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


async def check_websocket_connection(session_id: str) -> bool:
    """Check if session is connected to WebSocket server"""
    try:
        response = await http_client.get(
            f"{WEBSOCKET_SERVER_URL}/api/sessions/{session_id}/status"
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("connected", False)
        return False
    except Exception as e:
        print(f"[WebSocket] Failed to check connection for {session_id}: {e}")
        return False


async def disconnect_websocket_session(session_id: str) -> bool:
    """Disconnect session from WebSocket server"""
    try:
        response = await http_client.delete(
            f"{WEBSOCKET_SERVER_URL}/api/sessions/{session_id}"
        )
        return response.status_code in [200, 404]  # 404는 이미 연결 해제된 경우
    except Exception as e:
        print(f"[WebSocket] Failed to disconnect {session_id}: {e}")
        return False


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


class SessionCreateRequest(BaseModel):
    """세션 생성 요청 (나중에 인증 정보 추가 예정)"""
    api_key: Optional[str] = Field(default=None, description="API 키 (추후 인증용)")


class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    session_id: str
    created_at: int
    expires_at: Optional[int] = None
    message: str


@app.post("/v1/sessions", response_model=SessionCreateResponse)
async def create_session(request: Optional[SessionCreateRequest] = None):
    """
    새 세션 생성

    나중에 인증 시스템 추가 시:
    - API 키 검증
    - 사용자 정보 연결
    - 권한 확인

    현재는 인증 없이 세션 ID 발급
    """
    # TODO: 인증 로직 추가
    # if request and request.api_key:
    #     # 인증 처리
    #     pass

    # 세션 ID 생성
    session_id = str(uuid.uuid4())

    # 세션 정보 저장
    ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
    AUTHORIZED_SESSIONS[session_id] = {
        "created_at": int(time.time()),
        "expires_at": int(time.time()) + ttl_seconds,
        "api_key": request.api_key if request else None,
        # 나중에 사용자 정보 추가
    }

    print(f"[Session] Created: {session_id}")

    return SessionCreateResponse(
        session_id=session_id,
        created_at=AUTHORIZED_SESSIONS[session_id]["created_at"],
        expires_at=AUTHORIZED_SESSIONS[session_id]["expires_at"],
        message="Session created successfully. Connect to WebSocket server with this session_id."
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Create a chat completion (OpenAI compatible)

    Supports both streaming and non-streaming responses.
    Requires X-Session-ID header for authentication and WebSocket connection.
    """
    try:
        # 1. 세션 ID 확인 (헤더에서)
        session_id = x_session_id

        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="X-Session-ID header is required. Create a session first: POST /v1/sessions"
            )

        # 2. 인증된 세션인지 확인
        if session_id not in AUTHORIZED_SESSIONS:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized session. Create a session first: POST /v1/sessions"
            )

        # 3. 세션 만료 확인
        session_info = AUTHORIZED_SESSIONS[session_id]
        if session_info["expires_at"] < time.time():
            # 만료된 세션 제거
            del AUTHORIZED_SESSIONS[session_id]
            raise HTTPException(
                status_code=401,
                detail="Session expired. Create a new session: POST /v1/sessions"
            )

        # 4. WebSocket 연결 확인
        ws_connected = await check_websocket_connection(session_id)
        if not ws_connected:
            raise HTTPException(
                status_code=400,
                detail=f"Session {session_id} is not connected to WebSocket server. Connect first: ws://<server>/ws/{session_id}"
            )

        # 5. 세션 활동 시간 갱신 (TTL 연장)
        ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
        session_info["expires_at"] = int(time.time()) + ttl_seconds
        session_info["last_activity"] = int(time.time())

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


class SessionInfo(BaseModel):
    """세션 정보"""
    session_id: str
    created_at: int
    expires_at: int
    last_activity: Optional[int] = None
    websocket_connected: bool
    chat_history_count: int


@app.get("/v1/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """세션 정보 조회"""
    # 1. 인증된 세션인지 확인
    if session_id not in AUTHORIZED_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. 세션 정보 가져오기
    session_info = AUTHORIZED_SESSIONS[session_id]

    # 3. WebSocket 연결 상태 확인
    ws_connected = await check_websocket_connection(session_id)

    # 4. 대화 히스토리 개수 확인
    from utils.session_store import session_store
    history = session_store.get_history(session_id)
    chat_history_count = len(history.messages)

    return SessionInfo(
        session_id=session_id,
        created_at=session_info["created_at"],
        expires_at=session_info["expires_at"],
        last_activity=session_info.get("last_activity"),
        websocket_connected=ws_connected,
        chat_history_count=chat_history_count
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 (인증, WebSocket 연결, 대화 히스토리 모두 삭제)"""
    from utils.session_store import session_store

    # 1. 인증 세션 삭제
    if session_id in AUTHORIZED_SESSIONS:
        del AUTHORIZED_SESSIONS[session_id]
        print(f"[Session] Deleted from authorized sessions: {session_id}")

    # 2. WebSocket 연결 해제
    ws_disconnected = await disconnect_websocket_session(session_id)
    if ws_disconnected:
        print(f"[Session] Disconnected from WebSocket: {session_id}")

    # 3. 대화 히스토리 삭제
    history_deleted = session_store.delete_session(session_id)
    if history_deleted:
        print(f"[Session] Deleted chat history: {session_id}")

    return {
        "message": f"Session {session_id} deleted completely",
        "session_id": session_id,
        "auth_deleted": True,
        "websocket_disconnected": ws_disconnected,
        "history_deleted": history_deleted
    }


@app.post("/v1/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """대화 히스토리만 초기화 (세션은 유지)"""
    from utils.session_store import session_store

    # 1. 인증된 세션인지 확인
    if session_id not in AUTHORIZED_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        history = session_store.get_history(session_id)
        history.clear()
        print(f"[Session] Reset chat history: {session_id}")
        return {"message": f"Session {session_id} chat history reset"}
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

            # 1. 만료된 인증 세션 정리
            current_time = int(time.time())
            expired_sessions = []

            for session_id, session_info in list(AUTHORIZED_SESSIONS.items()):
                if session_info["expires_at"] < current_time:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                # 인증 세션 삭제
                del AUTHORIZED_SESSIONS[session_id]
                print(f"[Background] Expired session: {session_id}")

                # WebSocket 연결 해제
                ws_disconnected = await disconnect_websocket_session(session_id)
                if ws_disconnected:
                    print(f"[Background] Disconnected WebSocket for expired session: {session_id}")

            # 2. 대화 히스토리 정리 (memory 모드만)
            if session_store.store_type == "memory":
                cleaned = session_store.cleanup_expired_sessions()
                if cleaned > 0:
                    print(f"[Background] Cleaned up {cleaned} expired chat histories")

            if expired_sessions:
                print(f"[Background] Total cleaned: {len(expired_sessions)} sessions")

    # 백그라운드 정리 작업 시작
    asyncio.create_task(cleanup_loop())
    print("[Background] Session cleanup task started (5 minute interval)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Close HTTP client
    await http_client.aclose()
    print("[Shutdown] HTTP client closed")


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
