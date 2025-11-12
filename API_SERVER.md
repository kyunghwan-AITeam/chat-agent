# API Server Guide

Chat Agent를 OpenAI 호환 API 서버로 실행하는 방법을 설명합니다.

## 설치

의존성 설치 (FastAPI와 Uvicorn이 추가됨):

```bash
uv sync
```

## 실행

### 1. 환경 변수 설정 (필수)

**중요**: API 서버를 실행하기 전에 반드시 `.env` 파일을 설정해야 합니다.

```bash
# .env.example을 복사
cp .env.example .env

# .env 파일 편집
# 최소한 OLLAMA_MODEL과 OLLAMA_BASE_URL은 반드시 설정해야 합니다
```

**필수 환경 변수**:
```env
# Ollama 설정 (필수)
OLLAMA_MODEL=google/gemma-3-27b-it           # 사용할 Ollama 모델
OLLAMA_BASE_URL=http://localhost:11434/v1   # Ollama API 주소
```

**선택적 환경 변수**:
```env
# API 서버 설정
API_HOST=0.0.0.0        # 모든 네트워크 인터페이스에서 접근 가능
API_PORT=23000          # 서버 포트

# 모델 파라미터
TEMPERATURE=0.7         # 응답 랜덤성 (0.0~2.0)

# MCP Tools
USE_MCP_TOOLS=true      # MCP 도구 사용 여부
MCP_BASE_URL=https://localhost:22000
MCP_VERIFY_SSL=false
```

**환경 변수가 없으면?**

API 서버는 시작 시 필수 환경 변수를 검증합니다. `OLLAMA_MODEL` 또는 `OLLAMA_BASE_URL`이 설정되지 않으면 다음과 같은 에러 메시지와 함께 종료됩니다:

```
============================================================
ERROR: Missing required environment variables
============================================================

The following variables must be set in .env file:

  - OLLAMA_MODEL: Ollama model name (e.g., llama3.2, google/gemma-3-27b-it)
  - OLLAMA_BASE_URL: Ollama API base URL (e.g., http://localhost:11434/v1)

Please create a .env file with these variables.
You can copy from .env.example:

  cp .env.example .env

Then edit .env and set the appropriate values.
============================================================
```

### 2. API 서버 시작

환경 변수를 설정한 후 서버를 시작합니다:

```bash
# uv 사용
uv run python src/api_server.py

# 또는 직접 실행
python src/api_server.py
```

서버가 정상적으로 시작되면 설정 정보가 표시됩니다:

```
============================================================
Configuration Loaded
============================================================
OLLAMA_MODEL: google/gemma-3-27b-it
OLLAMA_BASE_URL: http://localhost:8002/v1
TEMPERATURE: 0.3
USE_MCP_TOOLS: true
API_HOST: 0.0.0.0
API_PORT: 23000
============================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23000 (Press CTRL+C to quit)
```

서버 엔드포인트:
- API 엔드포인트: `http://localhost:23000/v1/chat/completions`
- API 문서 (Swagger UI): `http://localhost:23000/docs`
- Health check: `http://localhost:23000/health`

## API 사용 방법

### OpenAI Python 클라이언트 사용

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 필수는 아니지만 OpenAI 클라이언트가 요구함
)

# 기본 채팅
response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "안녕하세요!"}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "서울 날씨 알려줘"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL 사용

```bash
# 기본 채팅
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "안녕하세요!"}
    ],
    "temperature": 0.7
  }'

# 스트리밍
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "서울 날씨 알려줘"}
    ],
    "stream": true
  }'
```

### LangChain에서 사용

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="llama3.2"
)

response = llm.invoke("안녕하세요!")
print(response.content)
```

## API 엔드포인트

### `POST /v1/chat/completions`

OpenAI 호환 채팅 완성 엔드포인트

**요청 본문:**
```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "메시지 내용"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

**응답 (non-streaming):**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "응답 내용"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**응답 (streaming):**
Server-Sent Events (SSE) 형식으로 스트리밍:
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",...}

data: [DONE]
```

### `GET /v1/models`

사용 가능한 모델 목록

**응답:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ollama"
    }
  ]
}
```

### `GET /health`

서버 상태 확인

**응답:**
```json
{
  "status": "healthy"
}
```

### `DELETE /v1/sessions/{session_id}`

세션 삭제 (대화 기록 제거)

### `POST /v1/sessions/{session_id}/reset`

세션 초기화 (대화 기록 리셋)

## 세션 관리

API 서버는 자동으로 세션을 생성하고 관리합니다:

- 각 요청은 독립적인 세션으로 처리됨
- 같은 세션에서 대화를 이어가려면 `messages` 배열에 전체 대화 기록을 포함
- 응답 헤더에 `X-Session-ID`가 포함됨 (커스텀 확장)

## 예제 코드

### Python 클라이언트 예제

```bash
uv run python examples/api_client_example.py
```

예제 파일: [examples/api_client_example.py](examples/api_client_example.py)

### cURL 예제

```bash
./examples/api_client_curl.sh
```

예제 파일: [examples/api_client_curl.sh](examples/api_client_curl.sh)

## 기능

### ✅ OpenAI 호환
- OpenAI Python SDK와 완벽 호환
- LangChain, LlamaIndex 등 OpenAI API를 사용하는 모든 프레임워크와 호환

### ✅ 스트리밍 지원
- Server-Sent Events (SSE)를 통한 실시간 응답 스트리밍
- OpenAI API와 동일한 스트리밍 형식

### ✅ MCP Tools 통합
- 날씨 정보 조회 (`get_weather`)
- 웹 검색 (`search_web`)
- 환경 변수로 켜고 끌 수 있음

### ✅ 대화 기록 관리
- 메시지 배열을 통한 대화 컨텍스트 유지
- 세션 기반 메모리 관리

### ✅ FastAPI Swagger UI
- 자동 생성된 API 문서: `http://localhost:8000/docs`
- 인터랙티브 API 테스트 가능

## 프로덕션 배포

### Docker 컨테이너로 실행

```bash
# Dockerfile 예제
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "src/api_server.py"]
```

### Nginx 리버스 프록시

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # For streaming support
        proxy_buffering off;
        proxy_cache off;
    }
}
```

### 환경 변수 설정

프로덕션에서는 다음 환경 변수를 설정:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export OLLAMA_BASE_URL=http://ollama-server:11434/v1
export USE_MCP_TOOLS=true
export MCP_BASE_URL=https://mcp-server:22000
```

## 제한사항

1. **세션 관리**: 현재 메모리 기반 세션 관리
   - 프로덕션에서는 Redis 등 외부 스토리지 사용 권장

2. **토큰 사용량**: 대략적인 추정치만 제공
   - 정확한 토큰 계산을 위해서는 tiktoken 등 사용 필요

3. **동시 요청**: Uvicorn의 기본 워커 설정 사용
   - 고부하 환경에서는 워커 수 조정 필요

## 문제 해결

### API 서버가 시작되지 않음

```bash
# 포트가 이미 사용 중인지 확인
lsof -i :8000

# 다른 포트로 실행
export API_PORT=8001
uv run python src/api_server.py
```

### Ollama 연결 오류

```bash
# Ollama가 실행 중인지 확인
ollama list

# Ollama 서버 시작
ollama serve
```

### MCP Tools 오류

```bash
# MCP Tools 비활성화
export USE_MCP_TOOLS=false

# 또는 .env 파일에서
USE_MCP_TOOLS=false
```

## 참고 자료

- [OpenAI API 문서](https://platform.openai.com/docs/api-reference)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Uvicorn 문서](https://www.uvicorn.org/)
