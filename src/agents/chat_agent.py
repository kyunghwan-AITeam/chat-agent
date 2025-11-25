"""
LangChain Chat Agent with Ollama (via OpenAI-compatible API)
"""
import re
import ast
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from utils import JsonFixer

class ChatAgent:
    """LangChain 기반 대화형 AI Agent (Ollama 연동)"""

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        system_prompt: Optional[str] = None,
        agents: Optional[List[Any]] = None,
        use_agent: bool = False,
        enable_langfuse: bool = False,
        session_id: Optional[str] = None
    ):
        """
        Initialize Chat Agent

        Args:
            model: Ollama model name (default: llama3.2)
            temperature: Temperature for response generation (default: 0.7)
            base_url: Ollama API base URL (default: http://localhost:11434/v1)
            api_key: API key (Ollama doesn't require a real key, default: "ollama")
            system_prompt: Custom system prompt (default: simple assistant prompt)
            tools: List of LangChain tools to use (default: None)
            use_agent: Whether to use agent with tools (default: False)
            enable_langfuse: Whether to enable Langfuse tracing (default: False)
            session_id: Session ID for session history management (default: None)
        """
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )

        self.model = model
        self.temperature = temperature
        self.tools = agents or []
        self.use_agent = use_agent and len(self.tools) > 0
        self.session_id = session_id
        self.enable_langfuse = enable_langfuse
        self.handler = None

        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Always be conversational and friendly."

        self.system_prompt = system_prompt

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create chain
        self.chain = self.prompt | self.llm

        # Load chat history from session store if session_id is provided
        if self.session_id:
            self.chat_history = self._load_chat_history()
        else:
            self.chat_history: List[Any] = []

    @staticmethod
    def extract_tool_calls_and_text(content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract THOUGHT, TOOL_CALL, and remaining text from model response.

        Args:
            content: Full model response

        Returns:
            Tuple of (thought, tool_call, remaining_text)
            - thought: Content inside <THOUGHT> tags (or None)
            - tool_call: Content inside <TOOL_CALL> tags (or None)
            - remaining_text: Text outside of any tags (or None)
        """
        thought = None
        tool_call = None
        remaining_text = content

        # Extract THOUGHT
        thought_match = re.search(r'<THOUGHT>\s*(.*?)\s*</THOUGHT>', content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            remaining_text = remaining_text.replace(thought_match.group(0), '', 1).strip()

        # Extract TOOL_CALL
        tool_call_match = re.search(r'<AGENT_CALL>\s*(.*?)\s*</AGENT_CALL>', content, re.DOTALL)
        if tool_call_match:
            tool_call = tool_call_match.group(1).strip()
            remaining_text = remaining_text.replace(tool_call_match.group(0), '', 1).strip()

        # If remaining text is empty, set to None
        remaining_text = remaining_text if remaining_text else None

        return (thought, tool_call, remaining_text)

    def _parse_pythonic_tool_calls(self, content: str, max_retries: int = 2) -> List[Dict[str, Any]]:
        """Parse pythonic format tool calls from model response (for Gemma 3)

        Handles:
        1. Tagged format: <TOOL_CALL>[func1(arg1='val1')]</TOOL_CALL>
        2. Markdown code blocks: ```[func1(arg1='val1')]```
        3. Plain format: [func1(arg1='val1')]

        Args:
            content: Model response content
            max_retries: Maximum number of LLM correction attempts (default: 2)

        Returns list of dicts: [{'name': 'func1', 'args': {'arg1': 'val1'}}, ...]
        """
        if not content:
            print("No content to parse for tool calls.")
            return []

        # Extract content from <TOOL_CALL> tags if present
        agent_call_match = re.search(r'<AGENT_CALL>\s*(.*?)\s*</AGENT_CALL>', content, re.DOTALL)
        agent_call_str = ''
        if agent_call_match:
            agent_call_str = agent_call_match.group(1).strip()

            if not (agent_call_str.startswith('{') and agent_call_str.endswith('}')):
                print("No json agent calls found in content.")
                return []

        # Try parsing JSON with LLM correction on failure
        for attempt in range(max_retries + 1):
            try:
                agent_call = json.loads(agent_call_str)
                return [agent_call]
            except json.JSONDecodeError as e:
                agent_call_str = JsonFixer.fix_json_with_llm(
                    invalid_json=agent_call_str,
                    error_message=str(e),
                    max_retries=2
                )
                attempt += 1
            except Exception as e:
                print(f"Unexpected error parsing tool calls: {str(e)}")
                return []

        return []

    def _handle_tool_calls(self, response: Any, original_message: str = "") -> str:
        """
        Handle tool calls from LLM response and get LLM to interpret the results.

        Args:
            response: Response from LLM
            original_message: Original user message for context

        Returns:
            Final response text after executing tools and LLM interpretation
        """
        thought_text: str = ""

        # Try to parse pythonic format tool calls (for Gemma 3)
        if hasattr(response, 'content') and isinstance(response.content, str):
            # Extract THOUGHT, TOOL_CALL, and remaining text
            thought, _tool_call, remaining_text = self.extract_tool_calls_and_text(response.content)

            if thought:
                thought_text = thought

            # Parse tool calls
            parsed_calls = self._parse_pythonic_tool_calls(response.content)
            if parsed_calls:
                response.tool_calls = parsed_calls
                response.content = remaining_text or ""

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                # Find the tool by name
                tool_found = False
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_found = True
                        try:
                            # Invoke the tool with proper error handling
                            # Use synchronous invoke() to avoid event loop conflicts
                            result = tool.invoke(tool_args)
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}")
                        except Exception as e:
                            # Detailed error logging
                            error_msg = f"Tool: {tool_name}\nArguments: {tool_args}\nError: {str(e)}"
                            tool_results.append(error_msg)
                            import traceback
                            print(f"\n=== Tool Invocation Error ===")
                            print(f"Tool: {tool_name}")
                            print(f"Arguments: {tool_args}")
                            print(f"Error: {str(e)}")
                            traceback.print_exc()
                            print("=" * 30 + "\n")
                        break

                # Handle tool not found case
                if not tool_found:
                    error_msg = f"Tool: {tool_name}\nError: Tool not found in available tools"
                    tool_results.append(error_msg)
                    print(f"\n=== Tool Not Found ===")
                    print(f"Requested tool: {tool_name}")
                    print(f"Available tools: {[t.name for t in self.tools]}")
                    print("=" * 30 + "\n")

            if tool_results:
                try:
                    # Ask LLM to interpret the tool results
                    tool_context = "\n\n".join(tool_results)
                    interpretation_prompt = f"""The user asked: "{original_message}"

You called some tools and got these results:

{tool_context}

IMPORTANT: The tool results may contain mixed Korean/English text (e.g., "Name is 김철수").
When creating your response:
1. Translate any English phrases to Korean naturally
2. Rephrase the information in a conversational Korean style
3. Example: "Name is 김철수" → "이름은 김철수입니다"

Provide a natural, conversational response to the user in Korean. Don't mention that you used tools - just provide the information naturally."""

                    interpretation_response = self.llm.invoke(
                        interpretation_prompt,
                        config={"callbacks": [self.handler]} if self.handler else {}
                    )
                    final_response = str(interpretation_response.content) if hasattr(interpretation_response, 'content') else str(interpretation_response)

                    # Prepend thought if it existed
                    if thought_text:
                        return f"{thought_text}\n\n{final_response}"
                    return final_response
                except Exception as e:
                    # Handle LLM interpretation errors
                    error_msg = f"LLM 해석 중 오류 발생: {str(e)}"
                    import traceback
                    print(f"\n=== LLM Interpretation Error ===")
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                    print("=" * 30 + "\n")
                    if thought_text:
                        return f"{thought_text}\n\n{error_msg}"
                    return error_msg

        # Return content if no tool calls (remove tags if present)
        if hasattr(response, 'content'):
            content = str(response.content)
            _thought, _tool_call, remaining_text = self.extract_tool_calls_and_text(content)
            if remaining_text:
                return remaining_text
            return content
        return str(response)

    async def _handle_tool_calls_async(self, response: Any, original_message: str = "") -> str:
        """
        Handle tool calls from LLM response and get LLM to interpret the results (async).

        Args:
            response: Response from LLM
            original_message: Original user message for context

        Returns:
            Final response text after executing tools and LLM interpretation
        """
        thought_text: str = ""

        # Try to parse pythonic format tool calls (for Gemma 3)
        if hasattr(response, 'content') and isinstance(response.content, str):
            # Extract THOUGHT, TOOL_CALL, and remaining text
            thought, _tool_call, remaining_text = self.extract_tool_calls_and_text(response.content)

            if thought:
                thought_text = thought

            # Parse tool calls
            parsed_calls = self._parse_pythonic_tool_calls(response.content)
            if parsed_calls:
                response.tool_calls = parsed_calls
                response.content = remaining_text or ""

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                # Find the tool by name
                tool_found = False
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_found = True
                        try:
                            # Invoke the tool asynchronously
                            result = await tool.ainvoke(tool_args)
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}")
                        except Exception as e:
                            # Detailed error logging
                            error_msg = f"Tool: {tool_name}\nArguments: {tool_args}\nError: {str(e)}"
                            tool_results.append(error_msg)
                            import traceback
                            print(f"\n=== Tool Invocation Error ===")
                            print(f"Tool: {tool_name}")
                            print(f"Arguments: {tool_args}")
                            print(f"Error: {str(e)}")
                            traceback.print_exc()
                            print("=" * 30 + "\n")
                        break

                # Handle tool not found case
                if not tool_found:
                    error_msg = f"Tool: {tool_name}\nError: Tool not found in available tools"
                    tool_results.append(error_msg)
                    print(f"\n=== Tool Not Found ===")
                    print(f"Requested tool: {tool_name}")
                    print(f"Available tools: {[t.name for t in self.tools]}")
                    print("=" * 30 + "\n")

            if tool_results:
                try:
                    # Ask LLM to interpret the tool results (async)
                    tool_context = "\n\n".join(tool_results)
                    interpretation_prompt = f"""The user asked: "{original_message}"

You called some tools and got these results:

{tool_context}

IMPORTANT: The tool results may contain mixed Korean/English text (e.g., "Name is 김철수").
When creating your response:
1. Translate any English phrases to Korean naturally
2. Rephrase the information in a conversational Korean style
3. Example: "Name is 김철수" → "이름은 김철수입니다"

Provide a natural, conversational response to the user in Korean. Don't mention that you used tools - just provide the information naturally."""

                    interpretation_response = await self.llm.ainvoke(
                        interpretation_prompt,
                        config={"callbacks": [self.handler]} if self.handler else {}
                    )
                    final_response = str(interpretation_response.content) if hasattr(interpretation_response, 'content') else str(interpretation_response)

                    # Prepend thought if it existed
                    if thought_text:
                        return f"{thought_text}\n\n{final_response}"
                    return final_response
                except Exception as e:
                    # Handle LLM interpretation errors
                    error_msg = f"LLM 해석 중 오류 발생: {str(e)}"
                    import traceback
                    print(f"\n=== LLM Interpretation Error ===")
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                    print("=" * 30 + "\n")
                    if thought_text:
                        return f"{thought_text}\n\n{error_msg}"
                    return error_msg

        # Return content if no tool calls (remove tags if present)
        if hasattr(response, 'content'):
            content = str(response.content)
            _thought, _tool_call, remaining_text = self.extract_tool_calls_and_text(content)
            if remaining_text:
                return remaining_text
            return content
        return str(response)

    async def _handle_tool_calls_streaming(self, response: Any, original_message: str = ""):
        """
        Handle tool calls and stream the LLM interpretation.

        Args:
            response: Response from LLM
            original_message: Original user message for context

        Yields:
            Chunks of the interpreted response
        """
        thought_text: str = ""

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('tool')
                tool_args = tool_call.get('params', {})

                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            result = await tool.ainvoke(tool_args)
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}")
                        except Exception as e:
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nError: {str(e)}")
                        break

            if tool_results:
                # Yield thought first if it exists
                if thought_text:
                    yield thought_text
                    yield "\n\n"

                # Ask LLM to interpret the tool results
                tool_context = "\n\n".join(tool_results)
                interpretation_prompt = f"""The user asked: "{original_message}"

You called some tools and got these results:

{tool_context}

IMPORTANT: The tool results may contain mixed Korean/English text (e.g., "Name is 김철수").
When creating your response:
1. Translate any English phrases to Korean naturally
2. Rephrase the information in a conversational Korean style
3. Example: "Name is 김철수" → "이름은 김철수입니다"

Provide a natural, conversational response to the user in Korean. Don't mention that you used tools - just provide the information naturally."""

                # Stream LLM interpretation
                for chunk in self.llm.stream(
                    interpretation_prompt,
                    config={"callbacks": [self.handler]} if self.handler else {}
                ):
                    if hasattr(chunk, 'content'):
                        content = str(chunk.content)
                        if content:
                            yield content
                return

        # Return content if no tool calls (remove tags if present)
        if hasattr(response, 'content'):
            content = str(response.content)
            _thought, _tool_call, remaining_text = self.extract_tool_calls_and_text(content)
            if remaining_text:
                yield remaining_text
            else:
                yield content

    async def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response (non-streaming, async)

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        # Create callback handler only if Langfuse is enabled
        if self.enable_langfuse:
            self.handler = CallbackHandler()
        else:
            self.handler = None

        try:
            # Pass handler to the chain invocation only if enabled
            config = {"callbacks": [self.handler]} if self.handler else {}

            # Invoke chain with message and history (async)
            response = await self.chain.ainvoke({
                "input": message,
                "chat_history": self.chat_history},
                config=config)

            # Handle tool calls if present
            if self.use_agent:
                response_text = await self._handle_tool_calls_async(response, message)
            else:
                response_text = str(response.content)

            # Update chat history
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=response_text))

            # Save chat history to session store
            self._save_chat_history()

            return response_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"오류가 발생했습니다: {str(e)}"

    async def chat_stream(self, message: str):
        """
        Send a message to the agent and stream the response (async)

        Args:
            message: User's message

        Yields:
            Chunks of the response as they arrive
        """
        # Create callback handler only if Langfuse is enabled
        if self.enable_langfuse:
            self.handler = CallbackHandler()
        else:
            self.handler = None

        try:
            full_response = ""
            response_obj = None

            # Pass handler to the chain streaming only if enabled
            config = {"callbacks": [self.handler]} if self.handler else {}

            # Stream the response (async)
            async for chunk in self.chain.astream({
                "input": message,
                "chat_history": self.chat_history},
                config=config
            ):
                response_obj = chunk
                if hasattr(chunk, 'content'):
                    content = str(chunk.content)
                    if content:
                        full_response += content
                        yield content

            # Handle tool calls if present (after streaming completes)
            if self.use_agent and response_obj:
                # Try pythonic parsing
                if full_response:
                    parsed_calls = self._parse_pythonic_tool_calls(full_response)
                    if parsed_calls:
                        response_obj.tool_calls = parsed_calls
                        response_obj.content = full_response

                # Execute tool calls and stream interpretation ONLY if tool calls exist
                if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                    yield "\n\n"

                    # Stream tool execution results
                    tool_results = []
                    for tool_call in response_obj.tool_calls:
                        tool_name = tool_call.get('tool')
                        tool_args = tool_call.get('params', {})
                        if "user_id" in tool_args:
                            tool_args["user_id"] = self.session_id
                        # Find the tool by name
                        tool_found = False
                        for tool in self.tools:
                            if tool.name == tool_name:
                                tool_found = True
                                try:
                                    # Invoke the tool asynchronously
                                    result = await tool.ainvoke(tool_args)
                                    tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}")
                                    print(f"[Tool executed: {tool_name}] {result}\n")
                                except Exception as e:
                                    # Detailed error logging
                                    error_msg = f"Tool: {tool_name}\nArguments: {tool_args}\nError: {str(e)}"
                                    tool_results.append(error_msg)
                                    yield f"[Tool execution failed: {tool_name} - {str(e)}]\n"
                                    import traceback
                                    print(f"\n=== Tool Invocation Error ===")
                                    print(f"Tool: {tool_name}")
                                    print(f"Arguments: {tool_args}")
                                    print(f"Error: {str(e)}")
                                    traceback.print_exc()
                                    print("=" * 30 + "\n")
                                break

                        # Handle tool not found case
                        if not tool_found:
                            error_msg = f"Tool: {tool_name}\nError: Tool not found in available tools"
                            tool_results.append(error_msg)
                            yield f"[Tool not found: {tool_name}]\n"
                            print(f"\n=== Tool Not Found ===")
                            print(f"Requested tool: {tool_name}")
                            print(f"Available tools: {[t.name for t in self.tools]}")
                            print("=" * 30 + "\n")

                    if tool_results:
                        try:
                            # Ask LLM to interpret the tool results and stream response
                            tool_context = "\n\n".join(tool_results)
                            interpretation_prompt = f"""The user asked: "{message}"

You called some tools and got these results:

{tool_context}

IMPORTANT: The tool results may contain mixed Korean/English text (e.g., "Name is 김철수").
When creating your response:
1. Translate any English phrases to Korean naturally
2. Rephrase the information in a conversational Korean style
3. Example: "Name is 김철수" → "이름은 김철수입니다"

Provide a natural, conversational response to the user in Korean. Don't mention that you used tools - just provide the information naturally."""

                            # Stream LLM interpretation (async)
                            config = {"callbacks": [self.handler]} if self.handler else {}
                            async for chunk in self.chain.astream({
                                "input": interpretation_prompt,
                                "chat_history": []  # Don't include history for interpretation
                            }, config=config):
                                if hasattr(chunk, 'content'):
                                    content = str(chunk.content)
                                    if content:
                                        full_response += content
                                        yield content
                        except Exception as e:
                            # Handle LLM interpretation errors
                            error_msg = f"\n[LLM 해석 중 오류 발생: {str(e)}]"
                            full_response += error_msg
                            yield error_msg
                            import traceback
                            print(f"\n=== LLM Interpretation Error ===")
                            print(f"Error: {str(e)}")
                            traceback.print_exc()
                            print("=" * 30 + "\n")

            # Update chat history after streaming is complete
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=full_response))

            # Save chat history to session store
            self._save_chat_history()

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n오류가 발생했습니다: {str(e)}"

    def _load_chat_history(self) -> List[Any]:
        """Load chat history from session store"""
        if not self.session_id:
            return []

        try:
            from utils.session_store import session_store
            history = session_store.get_history(self.session_id)
            messages = history.messages
            print(f"[ChatAgent] Loaded {len(messages)} messages for session: {self.session_id}")
            return list(messages)
        except Exception as e:
            print(f"[ChatAgent] Failed to load chat history: {e}")
            return []

    def _save_chat_history(self):
        """Save chat history to session store"""
        if not self.session_id:
            return

        try:
            from utils.session_store import session_store
            history = session_store.get_history(self.session_id)

            # Clear existing and add all messages
            history.clear()
            for msg in self.chat_history:
                history.add_message(msg)

            print(f"[ChatAgent] Saved {len(self.chat_history)} messages for session: {self.session_id}")
        except Exception as e:
            print(f"[ChatAgent] Failed to save chat history: {e}")

    def reset_memory(self):
        """Reset conversation history"""
        self.chat_history.clear()

        # Also clear from session store
        if self.session_id:
            self._save_chat_history()

    def get_system_prompt(self) -> str:
        """Get the current system prompt for debugging"""
        return self.system_prompt
