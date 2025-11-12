"""
Example client for Chat Agent API Server
Demonstrates how to use the OpenAI-compatible API
"""
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client to use our API server
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:23000/v1")

client = openai.OpenAI(
    base_url=API_BASE_URL,
    api_key="dummy"  # API key not required but OpenAI client expects it
)


def chat_completion_example():
    """Example: Basic chat completion (non-streaming)"""
    print("\n=== Non-Streaming Chat Example ===\n")

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "user", "content": "안녕하세요! 서울 날씨 알려주세요."}
        ],
        temperature=0.7
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"\nSession ID: {response}")  # Session ID is in response headers (X-Session-ID)


def chat_completion_streaming_example():
    """Example: Streaming chat completion"""
    print("\n=== Streaming Chat Example ===\n")

    stream = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "user", "content": "파이썬으로 API 서버 만드는 방법 검색해줘"}
        ],
        temperature=0.7,
        stream=True
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def chat_with_history_example():
    """Example: Chat with conversation history"""
    print("\n=== Chat with History Example ===\n")

    messages = [
        {"role": "user", "content": "내 이름은 철수야"},
        {"role": "assistant", "content": "안녕하세요, 철수님! 만나서 반갑습니다."},
        {"role": "user", "content": "내 이름이 뭐라고 했지?"}
    ]

    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages,
        temperature=0.7
    )

    print(f"Response: {response.choices[0].message.content}")


def multi_turn_conversation_example():
    """Example: Multi-turn conversation"""
    print("\n=== Multi-turn Conversation Example ===\n")

    messages = []

    # First turn
    messages.append({"role": "user", "content": "서울의 현재 날씨를 알려줘"})
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages,
        temperature=0.7
    )
    assistant_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_response})
    print(f"User: {messages[-2]['content']}")
    print(f"Assistant: {assistant_response}\n")

    # Second turn
    messages.append({"role": "user", "content": "내일은 어떨까?"})
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages,
        temperature=0.7
    )
    assistant_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_response})
    print(f"User: {messages[-2]['content']}")
    print(f"Assistant: {assistant_response}\n")


def main():
    """Run all examples"""
    try:
        print("="*60)
        print("Chat Agent API Client Examples")
        print("="*60)

        # Run examples
        chat_completion_example()
        chat_completion_streaming_example()
        chat_with_history_example()
        multi_turn_conversation_example()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the API server is running:")
        print("  uv run python src/api_server.py")


if __name__ == "__main__":
    main()
