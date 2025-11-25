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
from langchain_core.messages import HumanMessage, AIMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

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
            session_id: Session ID for Langfuse tracing (default: None)
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
        self.chat_history: List[Any] = []
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

        # # Remove markdown code blocks
        # agent_call_str = re.sub(r'```(?:tool_code|python)?\s*\n', '', agent_call_str)
        # agent_call_str = re.sub(r'\n?```\s*$', '', agent_call_str)
        # agent_call_str = agent_call_str.strip()

        # # Fix common model errors
        # agent_call_str = re.sub(r"(\]|'|\")r(?=\s*[,\)])", r'\1', agent_call_str)
        # agent_call_str = re.sub(r'(\w)r(?=\s*[,\)\]])', r'\1', agent_call_str)

            if not (agent_call_str.startswith('{') and agent_call_str.endswith('}')):
                print("No json agent calls found in content.")
                return []

        # Try parsing JSON with LLM correction on failure
        for attempt in range(max_retries + 1):
            try:
                # Parse as JSON
                agent_call = json.loads(agent_call_str)
                # tree = ast.parse(content, mode='eval')
                # if not isinstance(tree.body, ast.List):
                #     return []

                # tool_calls = []
                # for idx, call_node in enumerate(tree.body.elts):
                #     if not isinstance(call_node, ast.Call):
                #         continue

                #     # Extract function name
                #     if isinstance(call_node.func, ast.Name):
                #         func_name = call_node.func.id
                #     else:
                #         continue

                #     # Extract arguments
                #     args_dict = {}
                #     for keyword in call_node.keywords:
                #         arg_name = keyword.arg
                #         try:
                #             arg_value = ast.literal_eval(keyword.value)
                #         except (ValueError, TypeError):
                #             if isinstance(keyword.value, ast.Name):
                #                 arg_value = keyword.value.id
                #             else:
                #                 continue
                #         args_dict[arg_name] = arg_value

                #     tool_calls.append({
                #         'name': func_name,
                #         'args': args_dict,
                #         'id': f'call_{idx}',
                #         'type': 'tool_call'
                #     })

                return [agent_call]
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # Ask LLM to fix the JSON
                    print(f"\nJSON parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Asking LLM to correct the JSON format...")

                    correction_prompt = f"""
/no_think                    
The following JSON in <AGENT_CALL> tag has syntax errors:

<AGENT_CALL>
{agent_call_str}
</AGENT_CALL>

Error: {str(e)}

Please fix the JSON syntax errors and return ONLY the corrected JSON inside <AGENT_CALL> tags. Do not include any explanations.
The JSON should be valid and properly formatted."""

                    try:
                        correction_response = self.llm.invoke(
                            correction_prompt,
                            config={"callbacks": [self.handler]} if self.handler else {}
                        )
                        corrected_content = str(correction_response.content) if hasattr(correction_response, 'content') else str(correction_response)

                        # Extract corrected JSON from response
                        corrected_match = re.search(r'<AGENT_CALL>\s*(.*?)\s*</AGENT_CALL>', corrected_content, re.DOTALL)
                        if corrected_match:
                            agent_call_str = corrected_match.group(1).strip()
                            print(f"LLM corrected JSON:\n{agent_call_str}")
                        else:
                            print("LLM failed to return corrected JSON in proper format")
                            break
                    except Exception as llm_error:
                        print(f"LLM correction failed: {str(llm_error)}")
                        break
                else:
                    print(f"\nFailed to parse JSON after {max_retries} correction attempts")
                    return []
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

                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nResult: {result}")
                        except Exception as e:
                            tool_results.append(f"Tool: {tool_name}\nArguments: {tool_args}\nError: {str(e)}")
                        break

            if tool_results:
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

    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response (non-streaming)

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

            # Invoke chain with message and history
            response = self.chain.invoke({
                "input": message,
                "chat_history": self.chat_history},
                config=config)

            # Handle tool calls if present
            if self.use_agent:
                response_text = self._handle_tool_calls(response, message)
            else:
                response_text = str(response.content)

            # Update chat history
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=response_text))

            return response_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"오류가 발생했습니다: {str(e)}"

    def chat_stream(self, message: str):
        """
        Send a message to the agent and stream the response

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

            # Stream the response
            for chunk in self.chain.stream({
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

                # Execute tool calls and stream interpretation
                if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                    yield "\n\n"
                    # Run async generator in sync context
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    async_gen = self._handle_tool_calls_streaming(response_obj, message)
                    while True:
                        try:
                            chunk = loop.run_until_complete(async_gen.__anext__())
                            full_response += chunk
                            yield chunk
                        except StopAsyncIteration:
                            break

            # Update chat history after streaming is complete
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=full_response))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n오류가 발생했습니다: {str(e)}"

    def reset_memory(self):
        """Reset conversation history"""
        self.chat_history.clear()

    def get_system_prompt(self) -> str:
        """Get the current system prompt for debugging"""
        return self.system_prompt
