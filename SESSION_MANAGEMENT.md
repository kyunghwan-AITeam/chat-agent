# Session Management

세션 기반 인증 및 대화 히스토리 관리 시스템입니다.

## 개요

Home Assistant 시스템을 위한 세션 관리로, 다음을 통합 관리합니다:
- **세션 인증**: API 서버에서 발급 및 관리
- **WebSocket 연결**: 클라이언트와 Home Assistant 간 통신
- **대화 히스토리**: 메모리 또는 Redis 기반 저장

## 아키텍처

```
Client
  │
  ├─1─→ API Server: POST /v1/sessions
  │     └─→ session_id 발급
  │
  ├─2─→ WebSocket Server: ws://.../ws/{session_id}
  │     └─→ 연결 등록
  │
  └─3─→ API Server: POST /v1/chat/completions (X-Session-ID: ...)
        ├─→ 세션 인증 확인
        ├─→ WebSocket 연결 확인
        ├─→ Chat 처리
        └─→ Home Assistant MCP → WebSocket → Client
```

### 상세 플로우

```
┌─────────────────────────────────────────────────────────┐
│  Client                                                  │
└───────┬─────────────────────────────────────────────────┘
        │
        │ 1. POST /v1/sessions
        ▼
┌─────────────────────────────────────────────────────────┐
│  API Server (chat-agent)                                │
│  - 세션 ID 발급 (UUID)                                   │
│  - AUTHORIZED_SESSIONS에 저장                            │
│  - TTL 설정 (기본 30분)                                  │
└───────┬─────────────────────────────────────────────────┘
        │
        │ {"session_id": "abc-123", ...}
        ▼
┌─────────────────────────────────────────────────────────┐
│  Client                                                  │
│  - session_id 수신                                       │
└───────┬─────────────────────────────────────────────────┘
        │
        │ 2. WebSocket 접속: ws://.../ws/abc-123
        ▼
┌─────────────────────────────────────────────────────────┐
│  WebSocket Server (ai-voice-home-assistant)             │
│  - connections["abc-123"] = websocket                   │
│  - 연결 상태 유지                                         │
└───────┬─────────────────────────────────────────────────┘
        │
        │ 3. POST /v1/chat/completions (X-Session-ID: abc-123)
        ▼
┌─────────────────────────────────────────────────────────┐
│  API Server (chat-agent)                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 검증 단계:                                          │  │
│  │ 1. X-Session-ID 헤더 존재?                         │  │
│  │ 2. AUTHORIZED_SESSIONS에 존재?                     │  │
│  │ 3. 만료되지 않음?                                   │  │
│  │ 4. WebSocket 연결됨? (HTTP 확인)                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  모두 통과 → Chat 처리                                   │
│  ├─→ ChatAgent 생성 (대화 히스토리 자동 로드)            │
│  ├─→ LLM 응답 생성                                       │
│  └─→ Home Assistant MCP 툴 호출 시                       │
│       └─→ WebSocket Server로 요청 전달                   │
│            └─→ Client로 프록시                           │
└──────────────────────────────────────────────────────────┘
```

## 주요 구성 요소

### 1. API Server (`src/api_server.py`)

**세션 인증 관리:**
- `AUTHORIZED_SESSIONS`: 발급된 세션 정보 저장 (메모리)
- 세션 생성, 검증, 만료 처리
- WebSocket 서버와 HTTP 통신

**주요 엔드포인트:**
- `POST /v1/sessions` - 세션 생성
- `GET /v1/sessions/{session_id}` - 세션 정보 조회
- `DELETE /v1/sessions/{session_id}` - 세션 삭제
- `POST /v1/sessions/{session_id}/reset` - 대화 히스토리만 초기화
- `POST /v1/chat/completions` - Chat API (세션 검증 필수)

### 2. WebSocket Server (`~/Projects/ai-voice-home-assistant/websocket_server`)

**연결 관리:**
- `connections`: session_id → WebSocket 매핑
- 실시간 연결 상태 관리 (연결 해제 시 즉시 삭제)

**API 엔드포인트:**
- `GET /api/sessions/{session_id}/status` - 연결 상태 확인
- `DELETE /api/sessions/{session_id}` - 강제 연결 해제
- `POST /api/send-request` - Home Assistant → Client 요청 프록시

### 3. SessionStore (`src/utils/session_store.py`)

**대화 히스토리 저장:**
- `memory`: 인메모리 저장 (단일 워커)
- `redis`: Redis 기반 저장 (멀티 워커)

### 4. ChatAgent (`src/agents/chat_agent.py`)

**자동 히스토리 관리:**
- 초기화 시 SessionStore에서 로드
- 대화 후 자동 저장

## 설정 (.env)

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=23000

# WebSocket Server URL (세션 검증용)
WEBSOCKET_SERVER_URL=http://localhost:21000

# Session Store
SESSION_STORE_TYPE=memory  # memory | redis
SESSION_TTL_SECONDS=1800   # 30분

# Redis (redis 모드일 때)
REDIS_URL=redis://localhost:6379/0
```

## 사용 방법

### 1. 서버 실행

```bash
# 1. WebSocket 서버 (먼저 실행)
cd ~/Projects/ai-voice-home-assistant/websocket_server
python main.py
# → http://localhost:21000

# 2. API 서버
cd ~/Projects/chat-agent
python src/api_server.py
# → http://localhost:23000
```

### 2. 클라이언트 플로우

#### Step 1: 세션 생성

```bash
curl -X POST http://localhost:23000/v1/sessions \
  -H "Content-Type: application/json"

# 응답:
{
  "session_id": "abc-123-def-456",
  "created_at": 1732500000,
  "expires_at": 1732501800,
  "message": "Session created successfully. Connect to WebSocket server with this session_id."
}
```

#### Step 2: WebSocket 연결

```javascript
// JavaScript 예시
const sessionId = "abc-123-def-456";
const ws = new WebSocket(`ws://localhost:21000/ws/${sessionId}`);

ws.onopen = () => {
  console.log("Connected to WebSocket");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "request") {
    // Home Assistant 요청 처리
    handleHomeAssistantRequest(data.request_id, data.data);
  }
};

function handleHomeAssistantRequest(requestId, data) {
  // 기기 제어 등 처리
  const result = controlDevice(data);

  // 응답 전송
  ws.send(JSON.stringify({
    type: "response",
    request_id: requestId,
    result: result
  }));
}
```

#### Step 3: Chat API 호출

```bash
curl -X POST http://localhost:23000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: abc-123-def-456" \
  -d '{
    "messages": [
      {"role": "user", "content": "거실 불 켜줘"}
    ]
  }'
```

**검증 프로세스:**
1. ✅ `X-Session-ID` 헤더 존재
2. ✅ 인증된 세션 (POST /v1/sessions로 생성됨)
3. ✅ 만료되지 않음 (TTL 30분)
4. ✅ WebSocket 연결됨

**실패 시 에러:**
- `400`: X-Session-ID 헤더 없음
- `401`: 인증되지 않은 세션 또는 만료된 세션
- `400`: WebSocket 연결 안 됨

### 3. 세션 관리

#### 세션 정보 조회

```bash
curl -X GET http://localhost:23000/v1/sessions/abc-123-def-456

# 응답:
{
  "session_id": "abc-123-def-456",
  "created_at": 1732500000,
  "expires_at": 1732501800,
  "last_activity": 1732500500,
  "websocket_connected": true,
  "chat_history_count": 10
}
```

#### 대화 히스토리 초기화

```bash
curl -X POST http://localhost:23000/v1/sessions/abc-123-def-456/reset
# 세션은 유지, 대화 기록만 삭제
```

#### 세션 완전 삭제

```bash
curl -X DELETE http://localhost:23000/v1/sessions/abc-123-def-456

# 다음이 모두 삭제됨:
# 1. 인증 세션 (AUTHORIZED_SESSIONS)
# 2. WebSocket 연결
# 3. 대화 히스토리
```

## 세션 생명주기

### 정상 플로우

```
1. 생성: POST /v1/sessions
   └─→ AUTHORIZED_SESSIONS에 등록
   └─→ expires_at = now + 30분

2. WebSocket 연결: ws://.../ws/{session_id}
   └─→ connections에 등록

3. Chat 사용: POST /v1/chat/completions
   └─→ 검증 통과
   └─→ expires_at 갱신 (활동 시 30분 연장)
   └─→ last_activity 업데이트

4. 30분 미사용
   └─→ 백그라운드 정리 작업 (5분마다)
   └─→ 인증 세션 삭제
   └─→ WebSocket 연결 해제
   └─→ 대화 히스토리 삭제 (SessionStore TTL)
```

### 비정상 종료 처리

```
WebSocket 연결 끊김 (클라이언트 종료, 네트워크 오류 등)
└─→ WebSocket Server: connections에서 즉시 삭제
└─→ 다음 Chat API 호출 시:
    └─→ WebSocket 연결 확인 실패
    └─→ 400 에러 반환
    └─→ 클라이언트는 재연결 필요
```

## 보안

### 현재 구현

- ✅ 세션 ID는 UUID로 예측 불가능
- ✅ WebSocket 연결 필수 (임의 접근 방지)
- ✅ TTL 기반 자동 만료 (30분)
- ⚠️ 인증 없음 (누구나 세션 생성 가능)

### 추후 인증 시스템 추가 예정

```bash
# 세션 생성 시 API 키 검증
curl -X POST http://localhost:23000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key-here"}'

# 검증 과정:
# 1. API 키 유효성 확인
# 2. 사용자 정보 연결
# 3. 권한 확인
# 4. 세션 발급
```

**코드 위치:**
- [src/api_server.py:330](src/api_server.py#L330) - `# TODO: 인증 로직 추가`

## 모니터링

### 로그 메시지

```
# 세션 생성
[Session] Created: abc-123-def-456

# WebSocket 연결 확인
[WebSocket] Failed to check connection for abc-123: Connection refused

# 백그라운드 정리
[Background] Expired session: abc-123-def-456
[Background] Disconnected WebSocket for expired session: abc-123-def-456
[Background] Cleaned up 3 expired chat histories
[Background] Total cleaned: 3 sessions
```

### 세션 상태 확인

```bash
# API 서버에서 관리 중인 세션 수
# (현재는 로그로만 확인 가능, 추후 API 추가 예정)

# WebSocket 서버 헬스체크 (연결된 세션 확인)
curl http://localhost:21000/health
```

## 문제 해결

### Q: 세션 생성 후 Chat API 사용 시 400 에러

**에러:** `Session X is not connected to WebSocket server`

**원인:** WebSocket 연결 안 됨

**해결:**
1. WebSocket 서버 실행 확인: `curl http://localhost:21000/health`
2. 클라이언트에서 WebSocket 연결: `ws://localhost:21000/ws/{session_id}`
3. 연결 상태 확인: `GET /v1/sessions/{session_id}` → `websocket_connected: true`

### Q: 401 Unauthorized session

**원인:**
- 세션 ID가 발급되지 않음
- 세션이 만료됨 (30분)

**해결:**
1. 새 세션 생성: `POST /v1/sessions`
2. 받은 session_id 사용

### Q: WebSocket 연결이 자주 끊김

**원인:** 네트워크 불안정, 타임아웃

**해결:**
```javascript
// 주기적으로 ping 전송
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "ping" }));
  }
}, 30000); // 30초마다

// 재연결 로직
ws.onclose = () => {
  console.log("Disconnected, reconnecting...");
  setTimeout(connectWebSocket, 1000);
};
```

### Q: 서버 재시작 후 세션이 사라짐

**현재 동작:**
- 인증 세션 (AUTHORIZED_SESSIONS): 메모리 저장 → 재시작 시 손실
- 대화 히스토리: SESSION_STORE_TYPE에 따라 다름
  - `memory`: 손실
  - `redis`: 유지 (하지만 인증 세션이 없어서 사용 불가)

**해결:**
- 서버 재시작 후 클라이언트는 새 세션 생성 필요
- 추후 개선: 인증 세션도 Redis에 저장 가능

## 성능

### 메모리 사용량

- 인증 세션: ~200 bytes per session
- WebSocket 연결: ~1KB per connection
- 대화 히스토리: ~1KB per 10 messages

### 응답 시간

- 세션 생성: < 5ms
- 세션 검증: < 10ms (WebSocket 연결 확인 포함)
- Chat API: LLM 응답 시간에 따라 다름

### 권장 설정

- 단일 서버: 1000 동시 세션까지 무리 없음
- 멀티 워커 필요 시: Redis 모드 사용
- SESSION_TTL_SECONDS 조정으로 메모리 관리

## 다음 단계

### 필수 (보안)
- [ ] API 키 기반 인증 시스템
- [ ] 사용자별 권한 관리
- [ ] Rate limiting

### 선택 (기능 개선)
- [ ] 세션 목록 조회 API
- [ ] 세션 연장 API (TTL 수동 갱신)
- [ ] WebSocket 재연결 자동화
- [ ] 세션 통계 (생성/만료/활성 수)

### 선택 (인프라)
- [ ] 인증 세션도 Redis에 저장 (서버 재시작 대응)
- [ ] PostgreSQL 기반 장기 보관
- [ ] 세션 백업/복구

## 참고

### 관련 파일

- API 서버: [src/api_server.py](src/api_server.py)
- WebSocket 서버: `~/Projects/ai-voice-home-assistant/websocket_server/main.py`
- SessionStore: [src/utils/session_store.py](src/utils/session_store.py)
- 설정 예시: [.env.example](.env.example)

### 외부 의존성

- WebSocket 서버: `ai-voice-home-assistant` 프로젝트
- Redis (선택): SESSION_STORE_TYPE=redis 일 때
