# LangChain Home Assistant Chat Agent with Ollama

Python과 LangChain을 사용한 Ollama 기반 홈어시스턴트 대화형 AI Agent 프로젝트입니다.

## 기능

- **홈어시스턴트 챗봇** - 스마트홈 제어를 위한 대화형 인터페이스
- **백엔드 에이전트 연동** - 장치 제어, 씬 관리, 센서 모니터링 등
- Ollama 로컬 LLM 모델 사용
- LangChain Agent 프레임워크 활용
- **스트리밍 응답** - 실시간으로 응답 출력
- 대화 히스토리 메모리 관리
- 커스터마이징 가능한 시스템 프롬프트
- 환경 변수를 통한 설정 관리
- OpenAI 호환 API 방식으로 Ollama 연동

## 프로젝트 구조

```
chat-agent/
├── src/
│   ├── agents/
│   │   └── chat_agent.py      # 메인 Chat Agent 구현
│   ├── tools/                  # 커스텀 도구들 (확장 가능)
│   ├── prompts/                # 시스템 프롬프트 템플릿들
│   │   ├── __init__.py
│   │   └── system_prompts.py  # 홈어시스턴트 프롬프트
│   └── main.py                 # 진입점
├── pyproject.toml             # 프로젝트 설정 및 의존성 (uv)
├── requirements.txt           # Python 의존성 (pip 방식)
├── .env.example               # 환경 변수 예제
├── .gitignore
└── README.md
```

## 홈어시스턴트 기능

이 챗봇은 스마트홈 제어를 위한 특별한 시스템 프롬프트로 구성되어 있습니다.

### 지원하는 백엔드 에이전트

1. **device_control** - 스마트 장치 제어
   - 조명, 온도 조절기, 도어락 등 제어
   - 예: "거실 불 켜줘"

2. **scene_manager** - 씬 관리
   - 미리 설정된 씬 활성화
   - 예: "영화 보기 모드 켜줘"

3. **automation** - 자동화 관리
   - 자동화 규칙 생성/수정
   - 예: "밤 11시에 불 끄는 자동화 만들어줘"

4. **sensor_monitor** - 센서 데이터 확인
   - 온도, 습도, 모션 센서 읽기
   - 예: "침실 온도 알려줘"

5. **energy_manager** - 에너지 관리
   - 에너지 사용량 모니터링
   - 예: "오늘 전기 얼마나 썼어?"

### 에이전트 요청 형식

챗봇이 백엔드 에이전트에 요청할 때는 다음 형식을 사용합니다:

```
[AGENT_REQUEST]
agent: device_control
action: turn_on
parameters:
  - device: living_room_lights
[/AGENT_REQUEST]

거실 불을 켜드리겠습니다.
```

## 설치 방법

### 방법 1: uv 사용 (권장)

[uv](https://github.com/astral-sh/uv)는 빠른 Python 패키지 매니저입니다.

#### 1. uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip로 설치
pip install uv
```

#### 2. 의존성 설치

```bash
uv sync
```

#### 3. Ollama 설치 및 모델 다운로드

먼저 Ollama를 설치하고 사용할 모델을 다운로드합니다:

```bash
# Ollama 설치 (https://ollama.ai)
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드 (예: llama3.2)
ollama pull llama3.2

# Ollama 서버 실행 (기본 포트: 11434)
ollama serve
```

#### 4. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 설정을 확인합니다:

```bash
cp .env.example .env
```

`.env` 파일 내용 (기본값으로도 작동):

```env
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
TEMPERATURE=0.7
```

#### 5. 실행

```bash
uv run python src/main.py
```

### 방법 2: pip 사용 (기존 방식)

#### 1. Python 가상 환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

#### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai)
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull llama3.2

# Ollama 서버 실행
ollama serve
```

#### 4. 환경 변수 설정

```bash
cp .env.example .env
```

#### 5. 실행

```bash
cd src
python main.py
```

## 사용 방법

### 기본 실행

**uv 사용시:**
```bash
uv run python src/main.py
```

**pip 사용시:**
```bash
cd src
python main.py
```

### 대화 명령어

- 일반 대화: 메시지를 입력하고 Enter
- `reset`: 대화 히스토리 초기화
- `quit`, `exit`, `bye`: 프로그램 종료

## 사용 예제

### 일반 대화

```
You: 안녕하세요!
Agent: 안녕하세요! 스마트홈 제어에 도움이 필요하신가요?
```

### 장치 제어

```
You: 거실 불 켜줘
Agent:
[AGENT_REQUEST]
agent: device_control
action: turn_on
parameters:
  - device: living_room_lights
[/AGENT_REQUEST]

거실 불을 켜드리겠습니다.
```

### 센서 확인

```
You: 침실 온도 알려줘
Agent:
[AGENT_REQUEST]
agent: sensor_monitor
action: read_sensor
parameters:
  - sensor_type: temperature
  - location: bedroom
[/AGENT_REQUEST]

침실의 온도를 확인해드리겠습니다.
```

### 씬 활성화

```
You: 잘 시간이야
Agent:
[AGENT_REQUEST]
agent: scene_manager
action: activate_scene
parameters:
  - scene: goodnight
[/AGENT_REQUEST]

좋은 밤 되세요! 취침 모드를 활성화합니다 - 주요 조명을 끄고, 문을 잠그고, 온도를 편안한 수면 온도로 설정하겠습니다.
```

### 명령어

```
You: reset
Conversation history cleared.

You: quit
Goodbye!
```

## 커스터마이징

### 모델 변경

`.env` 파일에서 `OLLAMA_MODEL`을 변경:
- `llama3.2` (기본, 빠르고 가벼움)
- `mistral` (균형잡힌 성능)
- `codellama` (코딩 특화)
- `llama3.1:70b` (고성능, 대용량)
- `qwen2.5` (최신 모델)

사용 가능한 모델 확인:
```bash
ollama list
```

새로운 모델 다운로드:
```bash
ollama pull <model-name>
```

### Base URL 변경 (다른 포트 사용 시)

사용자 정의 포트로 Ollama를 실행하는 경우:
```env
OLLAMA_BASE_URL=http://localhost:YOUR_PORT/v1
```

### Temperature 조정

`.env` 파일에서 `TEMPERATURE` 값 조정 (0.0 ~ 2.0):
- 0.0: 매우 결정론적, 일관된 응답
- 0.7: 균형 잡힌 창의성 (기본값)
- 2.0: 매우 창의적, 다양한 응답

### 시스템 프롬프트 변경

다른 역할의 챗봇을 만들고 싶다면 시스템 프롬프트를 변경할 수 있습니다:

**방법 1: 새로운 프롬프트 만들기**

[src/prompts/system_prompts.py](src/prompts/system_prompts.py)에 새 프롬프트 추가:

```python
MY_CUSTOM_PROMPT = """You are a helpful assistant that..."""
```

**방법 2: main.py에서 직접 변경**

```python
from prompts import SIMPLE_ASSISTANT_PROMPT  # 또는 커스텀 프롬프트

agent = ChatAgent(
    model=model,
    temperature=temperature,
    base_url=base_url,
    api_key="ollama",
    system_prompt=SIMPLE_ASSISTANT_PROMPT  # 또는 직접 문자열
)
```

**현재 사용 가능한 프롬프트:**
- `HOME_ASSISTANT_PROMPT`: 스마트홈 제어용 (기본값)
- `SIMPLE_ASSISTANT_PROMPT`: 일반 대화용

### 스트리밍 vs 일반 모드

프로젝트는 기본적으로 스트리밍 모드로 실행됩니다. 필요시 코드에서 변경 가능합니다:

```python
# 스트리밍 모드 (기본) - 응답이 실시간으로 출력됨
for chunk in agent.chat_stream(user_input):
    print(chunk, end="", flush=True)

# 일반 모드 - 전체 응답이 한번에 출력됨
response = agent.chat(user_input)
print(response)
```

## VSCode 디버깅

VSCode에서 디버그 모드로 실행할 수 있습니다.

### 설정 파일

프로젝트에는 다음 VSCode 설정 파일이 포함되어 있습니다:

- [.vscode/launch.json](.vscode/launch.json) - 디버그 설정
- [.vscode/settings.json](.vscode/settings.json) - Python 환경 설정

### 디버그 실행 방법

1. **F5** 키를 누르거나 VSCode의 "Run and Debug" 패널에서 실행
2. 다음 설정 중 하나를 선택:
   - **Python: Chat Agent** - 통합 터미널에서 실행 (권장)
   - **Python: Chat Agent (External Terminal)** - 외부 터미널에서 실행
   - **Python: Current File** - 현재 열린 파일 실행

### 브레이크포인트 설정

코드의 원하는 줄 번호 왼쪽을 클릭하여 브레이크포인트를 설정할 수 있습니다:

- [src/main.py](src/main.py) - 메인 진입점
- [src/agents/chat_agent.py](src/agents/chat_agent.py) - Agent 로직
- [src/prompts/system_prompts.py](src/prompts/system_prompts.py) - 프롬프트 확인

### 디버그 단축키

- **F5**: 디버그 시작/계속
- **F9**: 브레이크포인트 토글
- **F10**: Step Over (다음 줄)
- **F11**: Step Into (함수 내부로)
- **Shift+F11**: Step Out (함수 밖으로)
- **Shift+F5**: 디버그 중지

## 요구사항

- Python 3.10 이상
- Ollama 설치 ([다운로드](https://ollama.ai))

## 라이선스

MIT License

## 문제 해결

### Ollama 연결 오류
```
Error: Connection refused
```
→ Ollama 서버가 실행 중인지 확인하세요: `ollama serve`
→ 포트 번호가 올바른지 확인하세요 (기본: 11434)

### 모델을 찾을 수 없음
```
Error: model not found
```
→ 모델이 다운로드되어 있는지 확인: `ollama list`
→ 필요한 모델 다운로드: `ollama pull llama3.2`

### 모듈을 찾을 수 없음
```
ModuleNotFoundError: No module named 'langchain'
```
→ uv 사용시: `uv sync`를 실행하세요.
→ pip 사용시: `pip install -r requirements.txt`를 실행하세요.

### Agent 기능이 작동하지 않음

일부 Ollama 모델은 function calling을 지원하지 않을 수 있습니다. 이 경우:
- `llama3.2` 또는 `mistral` 같은 최신 모델 사용
- 또는 단순 대화 모드로 변경

### 한글 입력 시 백스페이스 문제

터미널에서 한글 입력 시 백스페이스가 제대로 작동하지 않는 경우:
- 이 프로젝트는 `prompt-toolkit`을 사용하여 이 문제를 해결했습니다
- `uv sync` 또는 `pip install -r requirements.txt`로 의존성을 다시 설치하세요
- `prompt-toolkit`은 한글, 일본어, 중국어 등 멀티바이트 문자를 올바르게 처리합니다

## 추가 리소스

- [Ollama 공식 사이트](https://ollama.ai)
- [Ollama 모델 라이브러리](https://ollama.ai/library)
- [uv 문서](https://docs.astral.sh/uv/)
- [LangChain 문서](https://python.langchain.com/)
- [LangChain-Ollama 가이드](https://python.langchain.com/docs/integrations/chat/ollama)
- [Open-Meteo Weather MCP Server](open-meteo-weather-mcp/README.md)
