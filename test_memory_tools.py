"""
Test script for memory search tools.
"""
import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from mem0 import Memory
from tools.memory_tools import create_memory_tools

# Load environment variables
load_dotenv()

# Configure mem0
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "chat_memories",
            "embedding_model_dims": 1024,
            "on_disk": True,
            "path": "/tmp/qdrant_mem0"
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen3:32b",
            "ollama_base_url": "http://172.168.0.201:11434",
            "temperature": 0.1
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:0.6b",
            "ollama_base_url": "http://172.168.0.201:11434",
            "embedding_dims": 1024,
        },
    },
}


def test_memory_tools():
    """Test memory search tools"""
    print("Initializing mem0...")
    memory = Memory.from_config(config)
    user_id = "test_user"

    print("\n1. Adding test memories...")
    # Add some test memories
    test_conversations = [
        {"role": "user", "content": "내 이름은 김철수야"},
        {"role": "assistant", "content": "안녕하세요 김철수님! 반갑습니다."},
        {"role": "user", "content": "나는 파이썬 개발자야"},
        {"role": "assistant", "content": "파이썬 개발자시군요! 무엇을 도와드릴까요?"},
        {"role": "user", "content": "오늘 날씨가 좋네"},
        {"role": "assistant", "content": "네, 오늘 날씨가 정말 좋습니다!"},
    ]

    for conv in test_conversations:
        memory.add([conv], user_id=user_id)
        print(f"  Added: {conv['role']} - {conv['content']}")

    print("\n2. Creating memory tools...")
    tools = create_memory_tools(memory, user_id=user_id)
    print(f"  Created {len(tools)} tools:")
    for tool in tools:
        print(f"    - {tool.name}: {tool.description[:60]}...")

    print("\n3. Testing search_memory tool...")
    search_tool = tools[0]  # MemorySearchTool

    # Test search 1: Search for name
    print("\n  Test 1 - Searching for '이름':")
    result = search_tool.run({"query": "이름", "limit": 3})
    print(f"  Result:\n{result}")

    # Test search 2: Search for occupation
    print("\n  Test 2 - Searching for '개발자':")
    result = search_tool.run({"query": "개발자", "limit": 3})
    print(f"  Result:\n{result}")

    # Test search 3: Search for weather
    print("\n  Test 3 - Searching for '날씨':")
    result = search_tool.run({"query": "날씨", "limit": 3})
    print(f"  Result:\n{result}")

    print("\n4. Testing get_all_memories tool...")
    get_all_tool = tools[1]  # MemoryGetAllTool

    print("\n  Getting all memories (limit 5):")
    result = get_all_tool.run({"limit": 5})
    print(f"  Result:\n{result}")

    print("\n5. Testing with LangChain-style invocation...")
    # Test using the tool's invoke method (LangChain standard)
    result = search_tool.invoke({"query": "김철수"})
    print(f"  Search result:\n{result}")

    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_memory_tools()
