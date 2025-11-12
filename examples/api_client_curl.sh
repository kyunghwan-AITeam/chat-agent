#!/bin/bash
# Example curl commands for Chat Agent API Server

API_BASE_URL="${API_BASE_URL:-http://localhost:23000}"

echo "========================================"
echo "Chat Agent API Server - cURL Examples"
echo "========================================"

# 1. Health check
echo -e "\n1. Health Check"
echo "Command: curl $API_BASE_URL/health"
curl -s "$API_BASE_URL/health" | jq
sleep 1

# 2. List models
echo -e "\n2. List Models"
echo "Command: curl $API_BASE_URL/v1/models"
curl -s "$API_BASE_URL/v1/models" | jq
sleep 1

# 3. Non-streaming chat completion
echo -e "\n3. Non-Streaming Chat Completion"
echo "Command: curl -X POST $API_BASE_URL/v1/chat/completions"
curl -s -X POST "$API_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "안녕하세요!"}
    ],
    "temperature": 0.7,
    "stream": false
  }' | jq
sleep 1

# 4. Streaming chat completion
echo -e "\n4. Streaming Chat Completion"
echo "Command: curl -X POST $API_BASE_URL/v1/chat/completions (stream=true)"
curl -s -X POST "$API_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "1부터 5까지 세어줘"}
    ],
    "temperature": 0.7,
    "stream": true
  }'
echo -e "\n"
sleep 1

# 5. Chat with history
echo -e "\n5. Chat with Conversation History"
curl -s -X POST "$API_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "내 이름은 철수야"},
      {"role": "assistant", "content": "안녕하세요, 철수님!"},
      {"role": "user", "content": "내 이름이 뭐라고?"}
    ],
    "temperature": 0.7
  }' | jq
sleep 1

# 6. Weather query (requires MCP tools)
echo -e "\n6. Weather Query (requires MCP tools enabled)"
curl -s -X POST "$API_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "서울 날씨 알려줘"}
    ],
    "temperature": 0.7
  }' | jq
sleep 1

# 7. Web search query (requires MCP tools)
echo -e "\n7. Web Search (requires MCP tools enabled)"
curl -s -X POST "$API_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "파이썬 최신 뉴스 검색해줘"}
    ],
    "temperature": 0.7
  }' | jq

echo -e "\n========================================"
echo "Examples completed!"
echo "========================================"
