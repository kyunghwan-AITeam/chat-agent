# MCP Integration Guide

chat-agent는 이제 mcp-servers 프로젝트의 MCP 서버들과 연동할 수 있습니다.

## 연결된 MCP 서버

1. **Weather Server** - Open-Meteo 날씨 정보 제공
2. **Web Search Server** - Brave Search API 기반 웹 검색

## 사전 요구사항

### 1. MCP 서버 실행

먼저 mcp-servers 프로젝트에서 서버를 실행해야 합니다:

```bash
cd /home/khkim/Projects/mcp-servers
./run-mcp-servers.sh
```

서버가 실행되면 다음 엔드포인트에서 접근 가능합니다:
- Weather: `https://localhost:22000/weather/mcp`
- Search: `https://localhost:22000/search/mcp`

### 2. 환경 설정

`.env` 파일에서 MCP 설정을 확인하세요:

```env
# MCP Server Configuration
MCP_BASE_URL=https://localhost:22000
MCP_VERIFY_SSL=false
USE_MCP_TOOLS=true
```

## 사용 방법

### 1. 테스트

MCP 연결이 올바르게 작동하는지 테스트:

```bash
cd /home/khkim/Projects/chat-agent
uv run python test_mcp_integration.py
```

### 2. Chat Agent 실행

```bash
cd /home/khkim/Projects/chat-agent
uv run python src/main.py
```

### 3. MCP 도구 사용 예시

채팅 에이전트가 시작되면 다음과 같은 질문을 할 수 있습니다:

**날씨 정보:**
```
You: 서울 날씨 알려줘
You: What's the weather in Tokyo?
You: 뉴욕 내일 날씨 어때?
```

**웹 검색:**
```
You: Python 최신 뉴스 검색해줘
You: Search for machine learning tutorials
You: 인공지능 트렌드 찾아봐
```

**복합 질문:**
```
You: 서울 날씨 확인하고 비 올 확률 알려줘
You: 파이썬 날씨 API 라이브러리 검색해줘
```

## 아키텍처

```
┌─────────────────────┐
│   Chat Agent        │
│   (LangChain)       │
└──────────┬──────────┘
           │
           │ HTTP/HTTPS
           │
┌──────────▼──────────┐
│   MCP Client        │
│   (Session-based)   │
└──────────┬──────────┘
           │
           │ Port 22000
           │
┌──────────▼──────────┐
│   Nginx Proxy       │
│   (TLS termination) │
└─────┬──────────┬────┘
      │          │
┌─────▼────┐ ┌──▼─────────┐
│ Weather  │ │ Web Search │
│ Server   │ │ Server     │
└──────────┘ └────────────┘
```

## MCP 프로토콜 세부사항

MCP는 세션 기반 프로토콜입니다:

1. **세션 획득**: `noop` 메소드로 세션 ID 획득
2. **초기화**: `initialize` 메소드로 클라이언트 정보 전송
3. **도구 호출**: 세션 ID를 헤더에 포함하여 도구 호출

응답 형식: Server-Sent Events (SSE)
```
event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

## 구현 파일

- **[src/tools/mcp_client.py](src/tools/mcp_client.py)** - 저수준 MCP HTTP 클라이언트
- **[src/tools/mcp_tools.py](src/tools/mcp_tools.py)** - LangChain 도구 래퍼
- **[src/agents/chat_agent.py](src/agents/chat_agent.py)** - MCP 도구 지원이 추가된 Agent
- **[src/main.py](src/main.py)** - MCP 도구 초기화 및 실행

## 문제 해결

### MCP 서버가 응답하지 않음

```bash
# 서버 상태 확인
docker ps --filter "name=mcp"

# 로그 확인
docker-compose -f /home/khkim/Projects/mcp-servers/docker-compose.yml logs
```

### SSL 인증서 오류

개발 환경에서는 `.env`에서 SSL 검증을 비활성화:
```env
MCP_VERIFY_SSL=false
```

### Web Search가 작동하지 않음

Brave API 키가 설정되어 있는지 확인:
```bash
cat /home/khkim/Projects/mcp-servers/web_search_mcp/.env
```

### MCP 도구 비활성화

MCP 도구를 사용하지 않으려면 `.env`에서:
```env
USE_MCP_TOOLS=false
```

## 참고 자료

- [MCP 프로토콜 명세](https://modelcontextprotocol.io/)
- [mcp-servers 프로젝트](/home/khkim/Projects/mcp-servers/README.md)
- [FastMCP 문서](https://github.com/jlowin/fastmcp)
