#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from tools.mcp_tools import create_all_mcp_tools

load_dotenv()
tools = create_all_mcp_tools(
    os.getenv("MCP_BASE_URL", "https://localhost:22000"),
    os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"
)

from prompts import HOME_ASSISTANT_PROMPT
agent = ChatAgent(
    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    system_prompt=HOME_ASSISTANT_PROMPT,
    tools=tools,
    use_agent=True
)

print("Query: What's the weather in Seoul?")
print("\nResponse:")
for chunk in agent.chat_stream("What's the weather in Seoul?"):
    print(chunk, end="", flush=True)
print("\n")
