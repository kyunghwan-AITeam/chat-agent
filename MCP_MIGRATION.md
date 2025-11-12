# MCP 마이그레이션 가이드

## 개요

이 프로젝트의 MCP (Model Context Protocol) 통합이 두 가지 방식으로 개선되었습니다:

1. **커스텀 구현 → 공식 라이브러리**: `langchain-mcp-adapters` 사용
2. **하드코딩된 프롬프트 → 동적 프롬프트**: MCP 서버에서 툴 정보 자동 추출

## 변경 사항

### 이전 구현 (커스텀)

- **파일**: `src/tools/mcp_client.py`, `src/tools/mcp_tools.py`
- **특징**:
  - HTTP 기반 MCP 클라이언트 직접 구현
  - SSE (Server-Sent Events) 파싱 수동 처리
  - async/sync 브리지 수동 관리
  - 단일 서버만 지원
  - 약 200줄의 커스텀 코드

### 새로운 구현 (공식 라이브러리)

- **파일**: `src/tools/mcp_tools_v2.py`
- **라이브러리**: `langchain-mcp-adapters==0.1.12`
- **특징**:
  - Anthropic 공식 MCP 프로토콜 사용
  - 여러 전송 방식 지원 (stdio, HTTP, SSE)
  - **멀티 서버 지원**: 여러 MCP 서버 동시 연결 가능
  - 자동 세션 관리
  - 약 100줄의 간결한 코드

## 주요 개선 사항

### 1. 코드 간소화
```python
# 이전: 커스텀 클라이언트 + 래퍼
mcp_client = MCPClient(mcp_base_url, verify_ssl=verify_ssl)
wrapper = MCPToolWrapper(mcp_client=mcp_client, service="weather", ...)

# 새로운: 공식 라이브러리
client = MultiServerMCPClient(server_config)
tools = await client.get_tools()
```

### 2. 멀티 서버 지원
```python
server_config = {
    "weather": {
        "transport": "streamable_http",
        "url": "https://localhost:22000/weather/mcp",
    },
    "search": {
        "transport": "streamable_http",
        "url": "https://localhost:22000/search/mcp",
    }
}
```

### 3. SSL 검증 비활성화 (개발 환경)
```python
def create_httpx_client(**kwargs):
    return httpx.AsyncClient(verify=False, **kwargs)

server_config[server_name]["httpx_client_factory"] = create_httpx_client
```

## 사용 방법

### 기본 사용 (동기 방식)
```python
from tools.mcp_tools_v2 import create_mcp_tools

tools = create_mcp_tools(
    mcp_base_url="https://localhost:22000",
    verify_ssl=False
)
```

### 비동기 사용
```python
from tools.mcp_tools_v2 import create_mcp_tools_async

tools = await create_mcp_tools_async(
    mcp_base_url="https://localhost:22000",
    verify_ssl=False
)
```

### 세션 기반 사용 (Stateful)
```python
from tools.mcp_tools_v2 import create_mcp_tools_with_session

async with client.session("weather") as session:
    tools = await load_mcp_tools(session)
```

## 마이그레이션 체크리스트

- [x] `langchain-mcp-adapters` 패키지 설치
- [x] `src/tools/mcp_tools_v2.py` 생성
- [x] `src/main.py`에서 새로운 모듈 사용
- [x] SSL 검증 비활성화 구현 (httpx_client_factory)
- [x] 테스트 성공 확인 (test_mcp_v2.py)
- [x] 기존 파일 백업

## 테스트

마이그레이션 후 테스트:
```bash
uv run python test_mcp_v2.py
```

예상 출력:
- 2개의 툴 로드 성공 (`get_current_weather`, `search_web`)
- 날씨 API 호출 성공
- 검색 API 사용 가능

## 호환성

### LangChain 통합
- 기존 `ChatAgent` 클래스는 변경 없이 사용 가능
- 툴은 표준 LangChain `BaseTool` 인터페이스 준수
- 동기/비동기 호출 모두 지원

### 환경 변수
기존과 동일:
```env
USE_MCP_TOOLS=true
MCP_BASE_URL=https://localhost:22000
MCP_VERIFY_SSL=false
```

## 문제 해결

### SSL 인증서 오류
```
httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]
```
**해결**: `MCP_VERIFY_SSL=false` 설정 및 `httpx_client_factory` 사용

### 비동기 호출 오류
```
StructuredTool does not support sync invocation
```
**해결**: `tool.ainvoke()` 대신 `await tool.ainvoke()` 사용

## 이전 파일 위치

기존 커스텀 구현은 다음 위치에 백업되어 있습니다:
- `src/tools/legacy/mcp_client.py`
- `src/tools/legacy/mcp_tools.py`

## 동적 프롬프트 생성

### 개요

시스템 프롬프트가 MCP 툴 정보를 하드코딩하는 방식에서, MCP 서버로부터 동적으로 툴 정보를 추출하여 프롬프트를 생성하는 방식으로 변경되었습니다.

### 이전 방식 (하드코딩)
```python
# src/prompts/system_prompts.py
HOME_ASSISTANT_PROMPT = """...
## AVAILABLE FUNCTIONS

{
  "functions": [
    {
      "name": "get_weather",
      "description": "...",
      ...
    }
  ]
}
..."""
```

**문제점**:
- MCP 서버에 새로운 툴을 추가해도 프롬프트 수동 업데이트 필요
- 툴 정보가 MCP 서버와 불일치할 수 있음
- 유지보수 어려움

### 새로운 방식 (동적 생성)
```python
# src/prompts/prompt_builder.py
from prompts.prompt_builder import build_home_assistant_prompt

# MCP 툴 로드
tools = create_mcp_tools(mcp_base_url, mcp_verify_ssl)

# 동적 프롬프트 생성 (툴 정보 자동 추출)
system_prompt = build_home_assistant_prompt(tools)
```

**장점**:
- ✅ MCP 서버의 툴 정보 자동 반영
- ✅ 툴 추가/변경 시 코드 수정 불필요
- ✅ MCP 서버와 프롬프트 정보 자동 동기화
- ✅ 툴 파라미터 스키마 자동 추출

### 동적 프롬프트 빌더 기능

#### 1. 툴 정보 자동 추출
```python
def build_tool_description(tool: BaseTool) -> Dict[str, Any]:
    """툴의 이름, 설명, 파라미터 스키마를 자동으로 추출"""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": extract_from_args_schema(tool.args_schema)
    }
```

#### 2. JSON 스키마 생성
```python
def generate_tools_json(tools: List[BaseTool]) -> str:
    """LangChain 툴 목록을 JSON 스키마로 변환"""
    # 자동으로 중괄호 이스케이프 처리 (LangChain 호환)
```

#### 3. 예제 자동 생성
```python
def generate_tool_examples(tools: List[BaseTool]) -> str:
    """각 툴에 대한 사용 예제를 자동 생성"""
```

### 사용 방법

#### 기본 사용
```python
from prompts.prompt_builder import build_home_assistant_prompt

# 툴 있을 때
prompt = build_home_assistant_prompt(tools)

# 툴 없을 때
prompt = build_home_assistant_prompt(None)
```

#### 커스텀 프롬프트 생성
```python
# 자신만의 프롬프트 빌더 작성 가능
def my_custom_prompt_builder(tools: List[BaseTool]) -> str:
    tools_json = generate_tools_json(tools)
    return f"My custom prompt with tools:\n{tools_json}"
```

### 테스트

동적 프롬프트 생성 테스트:
```bash
uv run python test_dynamic_prompt.py
```

예상 출력:
- ✓ MCP 툴 정보 자동 추출
- ✓ JSON 스키마 생성 (이스케이프 처리)
- ✓ 툴 이름, 설명, 파라미터 포함
- ✓ 프롬프트 크기 동적 조정

## 참고 자료

- [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [LangChain MCP 문서](https://docs.langchain.com/oss/python/langchain/mcp)
- [MCP 프로토콜 사양](https://modelcontextprotocol.io)
