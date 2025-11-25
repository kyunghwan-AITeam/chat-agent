# Session Management

세션 기반 대화 히스토리 관리 시스템입니다.

## 개요

각 사용자의 대화 컨텍스트를 세션 ID로 구분하여 관리합니다. 메모리 기반(단일 워커)과 Redis 기반(멀티 워커) 두 가지 방식을 지원합니다.

## 아키텍처

```
┌─────────────────────────────────────────┐
│  API Request (with session_id)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  api_server.py                           │
│  - Extract session_id                    │
│  - Create ChatAgent for session          │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  ChatAgent                               │
│  - Load chat_history from SessionStore   │
│  - Process user message                  │
│  - Save chat_history to SessionStore     │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  SessionStore                            │
│  ┌────────────┐  ┌──────────────────┐   │
│  │  Memory    │  │  Redis           │   │
│  │  (worker=1)│  │  (multi-worker)  │   │
│  └────────────┘  └──────────────────┘   │
└──────────────────────────────────────────┘
```

## 주요 구성 요소

### 1. SessionStore (`src/utils/session_store.py`)

세션 히스토리를 저장/로드하는 추상 레이어입니다.

**지원하는 저장소 타입:**
- `memory`: 인메모리 저장 (단일 워커 전용)
- `redis`: Redis 기반 저장 (멀티 워커 지원)

**주요 기능:**
- `get_history(session_id)`: 세션 히스토리 가져오기
- `delete_session(session_id)`: 세션 삭제
- `cleanup_expired_sessions()`: 만료된 세션 정리 (memory 모드만)

### 2. ChatAgent (`src/agents/chat_agent.py`)

대화 에이전트가 자동으로 세션 히스토리를 관리합니다.

**세션 관리 메서드:**
- `_load_chat_history()`: 초기화 시 자동 로드
- `_save_chat_history()`: 대화 후 자동 저장
- `reset_memory()`: 세션 히스토리 초기화

### 3. API Server (`src/api_server.py`)

매 요청마다 새로운 ChatAgent를 생성하지만, 세션 히스토리는 자동으로 복원됩니다.

**변경 사항:**
- ❌ 제거: `agent_sessions` 전역 딕셔너리
- ✅ 추가: `create_agent_for_session()` - 세션별 에이전트 생성
- ✅ 추가: 백그라운드 정리 작업 (5분 간격)

## 설정 (.env)

```bash
# 세션 저장소 타입 (memory | redis)
SESSION_STORE_TYPE=memory

# 세션 TTL - 초 단위 (기본값: 1800초 = 30분)
SESSION_TTL_SECONDS=1800

# Redis URL (redis 모드일 때 사용)
REDIS_URL=redis://localhost:6379/0
```

## 사용 방법

### 메모리 기반 (기본값)

**장점:**
- 추가 인프라 불필요
- 빠른 성능
- 간단한 설정

**제한 사항:**
- 단일 워커만 지원 (`--workers 1`)
- 서버 재시작 시 세션 손실

**실행:**
```bash
# .env 설정
SESSION_STORE_TYPE=memory
SESSION_TTL_SECONDS=1800

# 서버 실행 (단일 워커)
python src/api_server.py
# 또는
uvicorn src.api_server:app --host 0.0.0.0 --port 24000 --workers 1
```

### Redis 기반

**장점:**
- 멀티 워커 지원
- 영구 저장 가능 (persistence 설정에 따라)
- 서버 재시작 후에도 세션 유지

**요구 사항:**
- Redis 서버 필요

**실행:**
```bash
# 1. Redis 서버 시작 (메모리 전용 - persistence 비활성화)
redis-server --save "" --appendonly no

# 2. .env 설정
SESSION_STORE_TYPE=redis
SESSION_TTL_SECONDS=1800
REDIS_URL=redis://localhost:6379/0

# 3. 서버 실행 (멀티 워커 가능)
uvicorn src.api_server:app --host 0.0.0.0 --port 24000 --workers 4
```

## CLI 사용 예시

### 세션 이름 지정

CLI는 **세션 이름**을 지정하여 여러 독립적인 대화 컨텍스트를 관리할 수 있습니다:

```bash
# 기본 세션 (default)
python src/main.py

# 'work' 세션 사용
python src/main.py work

# 'personal' 세션 사용
python src/main.py personal

# 'project1' 세션 사용
python src/main.py project1

# 출력 예시:
# ============================================================
# Chat Agent CLI
# ============================================================
# Session Name: work
# Session ID: cli-work-abc123...
# Session Store: memory
# Loaded History: 4 messages  (이전 대화가 있을 경우)
# ============================================================
```

**세션 이름의 장점:**
- 업무용(`work`), 개인용(`personal`)으로 대화 분리
- 프로젝트별(`project1`, `project2`) 별도 컨텍스트 관리
- 각 세션은 독립적인 대화 히스토리 유지

**사용 가능한 명령어:**
- `session` - 현재 세션 정보 보기
- `sessions` - 모든 사용 가능한 세션 목록 보기
- `new` - 현재 세션 이름으로 새로운 세션 ID 생성
- `reset` - 현재 세션의 대화 기록만 삭제
- `prompt` - 시스템 프롬프트 보기
- `quit` / `exit` / `bye` - 종료

**세션 동작 방식:**
1. 첫 실행: 세션 이름으로 새 세션 ID 생성, `.sessions/<name>.session` 파일에 저장
2. 재실행: 저장된 세션 ID로 이전 대화 복원 (store type에 따라)
3. `new` 명령: 같은 이름으로 새 세션 ID 생성

**예시:**
```bash
# work 세션 시작
$ python src/main.py work

You: 내 이름은 김철수야
Agent: 반갑습니다, 김철수님!

You: sessions
=== Available Sessions ===
1. default
2. personal
3. work (current)

To switch sessions, restart with:
  python src/main.py <session_name>
===

You: quit
Goodbye!

# 나중에 work 세션 재개 - 대화 히스토리 복원됨!
$ python src/main.py work
# Session Name: work
# Loaded History: 2 messages  ← 이전 대화 복원!

You: 내 이름이 뭐였지?
Agent: 김철수님이라고 말씀하셨습니다.

# 다른 세션으로 전환 - 완전히 다른 컨텍스트
$ python src/main.py personal
# Session Name: personal
# Loaded History: 0 messages  ← 새로운 대화

You: 내 이름은 이영희야
Agent: 반갑습니다, 이영희님!
```

## API 사용 예시

### 세션 ID 전달 방법

OpenAI 호환 API의 system 메시지에 세션 ID를 포함합니다:

```json
{
  "model": "google/gemma-3-27b-it",
  "messages": [
    {
      "role": "system",
      "content": "session_id: user-work-session"
    },
    {
      "role": "user",
      "content": "안녕하세요"
    }
  ]
}
```

**세션 ID 규칙:**
- 세션 ID를 제공하지 않으면 자동으로 UUID가 생성됩니다
- **같은 세션 ID를 사용하면 이전 대화가 복원됩니다** ✅
- 세션 ID는 자유롭게 지정 가능 (예: `user-work`, `customer-123`, `project-abc`)

**멀티턴 대화 예시:**

첫 번째 요청:
```bash
curl -X POST http://localhost:24000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "session_id: my-work-session"},
      {"role": "user", "content": "내 이름은 김철수야"}
    ]
  }'
```

두 번째 요청 (같은 세션 ID 사용):
```bash
curl -X POST http://localhost:24000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "session_id: my-work-session"},
      {"role": "user", "content": "내 이름이 뭐였지?"}
    ]
  }'
# 응답: "김철수님이라고 말씀하셨습니다."  ← 이전 대화 기억!
```

다른 세션 (다른 세션 ID):
```bash
curl -X POST http://localhost:24000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "session_id: my-personal-session"},
      {"role": "user", "content": "내 이름이 뭐였지?"}
    ]
  }'
# 응답: "죄송합니다, 아직 말씀하지 않으셨습니다."  ← 다른 세션!
```

### 세션 관리 엔드포인트

**세션 삭제:**
```bash
curl -X DELETE http://localhost:24000/v1/sessions/{session_id}
```

**세션 초기화:**
```bash
curl -X POST http://localhost:24000/v1/sessions/{session_id}/reset
```

## 세션 수명 주기

### CLI (main.py)

**Memory 모드:**
1. 첫 실행: 새 세션 생성, `.cli_session_id`에 저장
2. 재실행: 세션 ID로 히스토리 로드 (프로그램 종료 후 30분 이내)
3. 30분 경과: 백그라운드 정리 작업이 세션 삭제
4. 재실행 (30분 후): 세션 ID는 동일하지만 히스토리 없음

**Redis 모드:**
1. 첫 실행: 새 세션 생성, `.cli_session_id`에 저장
2. 재실행: Redis에서 히스토리 로드 (언제든 가능)
3. 30분 미사용: Redis TTL로 자동 만료
4. 재실행 (만료 후): 세션 ID는 동일하지만 히스토리 없음

### API (api_server.py)

**Memory 모드:**
1. 첫 요청: 새 세션 생성
2. 후속 요청: 기존 히스토리 로드
3. 30분 미사용: 백그라운드 태스크가 자동 삭제
4. 서버 재시작: 모든 세션 손실

**Redis 모드:**
1. 첫 요청: 새 세션 생성 (Redis에 저장)
2. 후속 요청: Redis에서 히스토리 로드
3. 30분 미사용: Redis TTL로 자동 만료
4. 서버 재시작: 세션 유지 (Redis에 보관됨)

## 전환 가이드

### Memory → Redis 전환

1. Redis 서버 설치 및 실행
2. `.env` 파일 수정:
   ```bash
   SESSION_STORE_TYPE=redis
   REDIS_URL=redis://localhost:6379/0
   ```
3. 서버 재시작

코드 수정 불필요! ✅

## 모니터링

세션 관리 로그 예시:

```
[SessionStore] Initialized with type: memory, TTL: 1800s
[ChatAgent] Loaded 4 messages for session: abc-123
[ChatAgent] Saved 6 messages for session: abc-123
[Background] Session cleanup task started (5 minute interval)
[Background] Cleaned up 3 expired sessions
```

## 보안 고려사항

### Memory 모드
- ✅ 디스크에 저장 안 됨
- ✅ 서버 재시작 시 자동 삭제
- ✅ TTL로 자동 만료

### Redis 모드
- ⚠️ Redis persistence 비활성화 권장 (완전 메모리 기반):
  ```bash
  redis-server --save "" --appendonly no
  ```
- ✅ Redis TTL로 자동 만료
- ✅ 네트워크 격리 권장 (localhost만 허용)

## 문제 해결

### Q: 대화 히스토리가 유지되지 않음
- 세션 ID가 올바르게 전달되는지 확인
- 서버 로그에서 `[ChatAgent] Loaded X messages` 확인
- TTL 시간 확인 (기본 30분)

### Q: 멀티 워커 사용 시 세션이 끊김
- `SESSION_STORE_TYPE=redis`로 설정했는지 확인
- Redis 서버가 실행 중인지 확인

### Q: 메모리 사용량이 계속 증가
- 백그라운드 정리 작업이 실행 중인지 확인
- `SESSION_TTL_SECONDS` 값 조정
- Redis 모드로 전환 고려

## 성능

### Memory 모드
- 읽기/쓰기: < 1ms
- 메모리: ~1KB per session (10개 메시지 기준)

### Redis 모드
- 읽기/쓰기: ~2-5ms (로컬 Redis)
- 메모리: Redis에서 관리

## 다음 단계

현재 구현으로 충분한 경우가 많지만, 필요에 따라 다음 기능 추가 가능:

- [ ] PostgreSQL 백엔드 (장기 보관)
- [ ] 세션 통계 API
- [ ] 세션 병합/이동 기능
- [ ] 사용자 인증 통합
