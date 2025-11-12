# Langfuse Integration Guide

Langfuse는 LLM 호출을 추적하고 모니터링하는 오픈소스 LLM 옵저빌리티 플랫폼입니다. Chat Agent는 Langfuse와 통합되어 모든 LLM 호출을 자동으로 추적할 수 있습니다.

## Langfuse란?

Langfuse는 다음 기능을 제공합니다:
- **LLM 호출 추적**: 모든 프롬프트, 응답, 토큰 사용량 기록
- **성능 모니터링**: 응답 시간, 비용 분석
- **디버깅**: 대화 기록, 도구 호출 추적
- **분석**: 사용 패턴, 오류 분석

## 설정 방법

### 1. Langfuse 서버 설정

Langfuse 서버가 필요합니다. 다음 중 하나를 선택:

#### 옵션 A: 클라우드 버전 사용
```bash
# https://cloud.langfuse.com 에서 계정 생성
# API 키 발급 받기
```

#### 옵션 B: 로컬 Docker로 실행
```bash
# Langfuse 서버 실행
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up -d

# 서버 접속: http://localhost:3000
```

### 2. 환경 변수 설정

`.env` 파일에 Langfuse 설정 추가:

```env
# Langfuse 활성화
LANGFUSE_ENABLED=true

# Langfuse 서버 주소
LANGFUSE_HOST=http://192.168.3.20:3000

# Langfuse API 키 (옵션 - 인증이 필요한 경우)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key

# 환경 구분
LANGFUSE_ENVIRONMENT=development
```

**필수 환경 변수**:
- `LANGFUSE_ENABLED`: Langfuse 사용 여부 (`true`/`false`)
- `LANGFUSE_HOST`: Langfuse 서버 주소

**선택적 환경 변수**:
- `LANGFUSE_PUBLIC_KEY`: 공개 키 (인증이 필요한 경우)
- `LANGFUSE_SECRET_KEY`: 비밀 키 (인증이 필요한 경우)
- `LANGFUSE_ENVIRONMENT`: 환경 구분 (`development`, `staging`, `production`)

### 3. 설정 파일 조정 (선택사항)

[langfuse_config.yaml](langfuse_config.yaml) 파일에서 고급 설정 가능:

```yaml
langfuse:
  enabled: false  # .env의 LANGFUSE_ENABLED가 우선
  debug: false    # 디버그 로그 출력
  flush_at: 15    # 버퍼 크기
  flush_interval: 0.5  # 전송 간격 (초)

  tags:
    - "smart-home"
    - "korean-llm"
    - "${LANGFUSE_ENVIRONMENT:development}"
```

## 사용 방법

### API 서버에서 사용

API 서버를 시작하면 자동으로 Langfuse에 추적됩니다:

```bash
uv run python src/api_server.py
```

서버 시작 시 Langfuse 설정 확인:

```
============================================================
Configuration Loaded
============================================================
LLM_MODEL: google/gemma-3-27b-it
LLM_BASE_URL: http://localhost:8002/v1
TEMPERATURE: 0.3
USE_MCP_TOOLS: true
LANGFUSE_ENABLED: true                    ← Langfuse 활성화 확인
LANGFUSE_HOST: http://192.168.3.20:3000  ← 서버 주소 확인
LANGFUSE_ENVIRONMENT: development
API_HOST: 0.0.0.0
API_PORT: 23000
============================================================
```

### CLI 챗봇에서 사용

main.py에서도 Langfuse를 활성화하려면 ChatAgent 초기화 시 `enable_langfuse=True` 추가:

```python
from agents.chat_agent import ChatAgent

agent = ChatAgent(
    model=model,
    temperature=temperature,
    base_url=base_url,
    api_key="ollama",
    system_prompt=HOME_ASSISTANT_PROMPT,
    enable_langfuse=True  # Langfuse 활성화
)
```

## 추적되는 정보

Langfuse에 다음 정보가 추적됩니다:

### 1. LLM 호출 정보
- **Model**: 사용된 모델 이름
- **Input**: 사용자 입력 메시지 (전체 대화 기록 포함)
- **Output**: LLM 응답
- **Tokens**: 입력/출력 토큰 수 (추정값)

### 2. 메타데이터
- **Temperature**: 사용된 temperature 값
- **Tools Available**: 사용 가능한 도구 수
- **Tools Used**: 실제 사용된 도구 목록
- **Execution Time**: 실제 LLM 호출 시작부터 완료까지의 시간 (Langfuse에 duration으로 표시됨)

### 3. 태그
- `smart-home`: 스마트홈 관련 호출
- `korean-llm`: 한국어 LLM
- `development`/`staging`/`production`: 환경 구분

## Langfuse 대시보드 사용

### 1. 대시보드 접속

Langfuse 서버에 접속:
```
http://192.168.3.20:3000
```

### 2. 추적 데이터 확인

**Traces** 메뉴에서:
- 모든 LLM 호출 기록 확인
- 각 호출의 입력/출력 확인
- 응답 시간, 토큰 사용량 확인

**Sessions** 메뉴에서:
- 사용자 세션별로 대화 그룹화
- 전체 대화 흐름 추적

**Analytics** 메뉴에서:
- 토큰 사용량 통계
- 비용 분석
- 응답 시간 분포

## 디버깅

### Langfuse 연결 확인

디버그 모드 활성화:

`langfuse_config.yaml`:
```yaml
langfuse:
  enabled: true
  debug: true  # 디버그 로그 활성화
```

또는 `.env`:
```env
LANGFUSE_DEBUG=true
```

디버그 모드에서는 다음 로그가 출력됩니다:
```
Langfuse initialized: http://192.168.3.20:3000
Tags: ['smart-home', 'korean-llm', 'development']
Langfuse trace created: chat_completion
```

### 문제 해결

#### Langfuse가 활성화되지 않음
```bash
# 환경 변수 확인
echo $LANGFUSE_ENABLED
echo $LANGFUSE_HOST

# .env 파일 확인
cat .env | grep LANGFUSE
```

#### 연결 오류
```
Warning: Could not initialize Langfuse: ...
```

확인 사항:
1. Langfuse 서버가 실행 중인지 확인
2. `LANGFUSE_HOST` 주소가 올바른지 확인
3. 네트워크 연결 확인

#### 데이터가 전송되지 않음

```python
# ChatAgent에서 flush 호출
agent.langfuse_config.flush()
```

또는 `langfuse_config.yaml`에서 설정 조정:
```yaml
langfuse:
  flush_at: 1  # 매 호출마다 전송
  flush_interval: 0.1  # 더 자주 전송
```

## 예제

### 1. 기본 사용

```python
# .env 설정
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://192.168.3.20:3000
LANGFUSE_ENVIRONMENT=development

# API 서버 실행
uv run python src/api_server.py

# 요청 보내기
curl -X POST http://localhost:23000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [{"role": "user", "content": "안녕하세요!"}]
  }'

# Langfuse 대시보드에서 추적 데이터 확인
```

### 2. 프로덕션 환경

```env
# .env.production
LANGFUSE_ENABLED=true
LANGFUSE_HOST=https://langfuse.your-domain.com
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_ENVIRONMENT=production
```

### 3. 개발 환경 (디버그 모드)

```env
# .env.development
LANGFUSE_ENABLED=true
LANGFUSE_DEBUG=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_ENVIRONMENT=development
```

`langfuse_config.yaml`:
```yaml
langfuse:
  debug: true
  flush_at: 1  # 즉시 전송
```

## 비용 절감 팁

### 1. 선택적 활성화

개발 중에만 활성화:
```env
LANGFUSE_ENABLED=true  # 개발
LANGFUSE_ENABLED=false # 프로덕션 (비용 절감)
```

### 2. 샘플링

모든 요청을 추적하지 않고 일부만 추적:

```python
import random

enable_langfuse = random.random() < 0.1  # 10%만 추적

agent = ChatAgent(
    # ...
    enable_langfuse=enable_langfuse
)
```

### 3. 배치 전송

이벤트를 모아서 전송:
```yaml
langfuse:
  flush_at: 50  # 50개씩 모아서 전송
  flush_interval: 5.0  # 5초마다 전송
```

## 참고 자료

- [Langfuse 공식 문서](https://langfuse.com/docs)
- [Langfuse GitHub](https://github.com/langfuse/langfuse)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
