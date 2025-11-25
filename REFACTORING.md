# Refactoring: Code Consolidation

## Overview

공통 로직을 통합하여 `main.py` (CLI)와 `api_server.py` (HTTP API)가 동일한 agent 설정 로직을 공유하도록 리팩토링했습니다.

## Changes

### 1. 새로운 파일: `src/utils/agent_factory.py`

공통 agent 설정 로직을 모두 추출하여 재사용 가능한 팩토리 함수들로 만들었습니다.

#### 주요 함수들

- **`create_memory()`**: Mem0 Memory 인스턴스 생성 (Qdrant + OpenAI)
- **`load_mcp_tools()`**: MCP 서버에서 도구 로드 (v2 사용)
- **`load_memory_tools()`**: Memory 검색 도구 로드
- **`initialize_langfuse()`**: Langfuse 추적 초기화
- **`build_system_prompt_with_tools()`**: 도구 정보를 포함한 동적 시스템 프롬프트 생성
- **`create_chat_agent()`**: ChatAgent 인스턴스 생성
- **`setup_agent()`**: 완전한 agent 설정 (원스톱 함수)

### 2. `src/main.py` 리팩토링

**Before**: ~230 lines (모든 설정 로직 포함)
**After**: ~90 lines (인터페이스 로직만)

#### 주요 변경사항
- `setup_agent()` 함수 사용으로 모든 설정 자동화
- MCP, Memory, Langfuse 초기화 로직 제거
- CLI 인터페이스 로직만 유지

```python
# 단순화된 설정
agent, memory = setup_agent(
    user_id="alex",
    verbose=True
)
```

### 3. `src/api_server.py` 리팩토링

**Before**: ~510 lines (구버전 MCP, 정적 프롬프트)
**After**: ~465 lines (최신 로직 사용)

#### 주요 변경사항
- `agent_factory` 함수들 사용으로 최신 로직 적용
- `mcp_tools_v2.py` 사용 (서버별 도구 그룹화)
- 동적 프롬프트 빌더 사용 (`build_home_assistant_prompt`)
- Memory tools를 agent instructions에 통합
- 타입 안정성 개선 (model 파라미터)

```python
# 공통 로직 재사용
mcp_tools, agent_instructions, _ = load_mcp_tools(...)
memory_tools = load_memory_tools(MEMORY, session_id)
system_prompt = build_system_prompt_with_tools(agent_instructions, memory_tools)
agent = create_chat_agent(...)
```

## Benefits

### 1. 코드 중복 제거
- MCP 도구 로딩, Memory 설정, Langfuse 초기화 등의 로직이 한 곳에만 존재
- 버그 수정 시 한 번만 수정하면 모든 곳에 적용

### 2. 일관성 보장
- CLI와 API가 완전히 동일한 agent 설정 사용
- 동일한 MCP 서버, Memory 구성, System Prompt 사용

### 3. 유지보수성 향상
- 설정 변경 시 `agent_factory.py`만 수정
- 각 인터페이스는 고유 로직(CLI/API)에만 집중

### 4. 테스트 용이성
- 공통 로직을 독립적으로 테스트 가능
- Mock 및 Unit test 작성 용이

## Migration Guide

### For Developers

기존 코드에서 agent를 설정하던 방식:

```python
# Old way (before refactoring)
from tools.mcp_tools import create_all_mcp_tools
from prompts import HOME_ASSISTANT_PROMPT

tools = create_all_mcp_tools(...)
agent = ChatAgent(
    model=model,
    system_prompt=HOME_ASSISTANT_PROMPT,
    tools=tools,
    ...
)
```

새로운 방식:

```python
# New way (after refactoring)
from utils.agent_factory import setup_agent

agent, memory = setup_agent(
    user_id="user123",
    verbose=True
)
```

### 커스터마이징

더 세밀한 제어가 필요한 경우:

```python
from utils.agent_factory import (
    load_mcp_tools,
    load_memory_tools,
    build_system_prompt_with_tools,
    create_chat_agent
)

# 개별 함수 사용
mcp_tools, instructions, _ = load_mcp_tools(url, verify_ssl)
memory_tools = load_memory_tools(memory, user_id)
prompt = build_system_prompt_with_tools(instructions, memory_tools)
agent = create_chat_agent(system_prompt=prompt, tools=all_tools)
```

## Testing

모든 파일의 구문 검사 완료:
```bash
python3 -m py_compile src/utils/agent_factory.py
python3 -m py_compile src/main.py
python3 -m py_compile src/api_server.py
```

## Next Steps

1. ✅ 공통 로직 추출 완료
2. ✅ main.py 리팩토링 완료
3. ✅ api_server.py 리팩토링 완료
4. ⏳ 통합 테스트 실행 권장
5. ⏳ 실제 환경에서 검증

## Breaking Changes

**None** - 외부 API 및 CLI 사용법은 변경 없음. 내부 구조만 개선됨.
