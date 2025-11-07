"""
LangChain Chat Agent with Ollama (via OpenAI-compatible API)
"""
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from tools.weather_service import WeatherService, WeatherServiceError


class ChatAgent:
    """LangChain 기반 대화형 AI Agent (Ollama 연동)"""

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        use_agent: bool = False
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
        """
        # Initialize LLM
        base_llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )

        self.chat_history: List[Any] = []
        self.weather_service = WeatherService()
        self.tools = tools or []
        self.use_agent = use_agent and len(self.tools) > 0

        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Always be conversational and friendly."

        # Store system prompt for debugging
        self.system_prompt = system_prompt

        # Bind tools to LLM if using tools
        if self.use_agent:
            try:
                # Bind tools to LLM for function calling
                self.llm = base_llm.bind_tools(self.tools)
                print(f"Tools bound to LLM: {[t.name for t in self.tools]}")
            except Exception as e:
                print(f"Warning: Could not bind tools to LLM: {e}")
                print("Falling back to simple chain without tools.")
                self.use_agent = False
                self.llm = base_llm
        else:
            self.llm = base_llm

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create chain
        self.chain = self.prompt | self.llm

    @staticmethod
    def _extract_agent_requests(text: str) -> List[Dict[str, Any]]:
        """Extract [AGENT_REQUEST] blocks from the model output."""
        pattern = re.compile(r"\[AGENT_REQUEST\](.*?)\[/AGENT_REQUEST\]", re.DOTALL)
        requests: List[Dict[str, Any]] = []

        for match in pattern.finditer(text):
            block = match.group(1)
            request_data: Dict[str, Any] = {}
            parameters: Dict[str, Any] = {}
            in_parameters = False

            for raw_line in block.strip().splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.lower().startswith("parameters"):
                    in_parameters = True
                    continue

                if in_parameters and line.startswith("-"):
                    line = line.lstrip("-").strip()
                    if ":" in line:
                        key, value = [part.strip() for part in line.split(":", 1)]
                        parameters[key] = value
                    continue

                if ":" in line:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    request_data[key.lower()] = value

            if parameters:
                request_data["parameters"] = parameters

            if request_data:
                requests.append(request_data)

        return requests

    def _enrich_with_weather(self, text: str) -> Tuple[str, List[str]]:
        """Run weather requests found in the response and append their results."""
        additions: List[str] = []
        requests = self._extract_agent_requests(text)

        for request in requests:
            agent_name = str(request.get("agent", "")).strip().lower()
            if agent_name != "weather":
                continue

            params = request.get("parameters", {})
            location = params.get("location")

            latitude = params.get("latitude")
            longitude = params.get("longitude")

            lat_value: Optional[float]
            lon_value: Optional[float]

            try:
                lat_value = float(latitude) if latitude is not None else None
            except (ValueError, TypeError):
                lat_value = None

            try:
                lon_value = float(longitude) if longitude is not None else None
            except (ValueError, TypeError):
                lon_value = None

            try:
                weather = self.weather_service.get_current_weather(
                    location=location,
                    latitude=lat_value,
                    longitude=lon_value,
                )
                additions.append(weather["summary"])
            except WeatherServiceError as exc:
                additions.append(f"[날씨 오류] {exc}")

        if additions:
            base = text.rstrip()
            combined = base + "\n\n" + "\n".join(additions)
            return combined, additions

        return text, []

    def _parse_pythonic_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse pythonic format tool calls from model response (for Gemma 3)

        Handles markdown code blocks like:
        ```tool_code
        [func1(arg1='val1'), func2(arg2='val2')]
        ```

        Returns list of dicts: [{'name': 'func1', 'args': {'arg1': 'val1'}}, ...]
        """
        if not content:
            return []

        # Remove markdown code blocks
        content = re.sub(r'```(?:tool_code|python)?\s*\n', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

        # Fix common model errors
        # Remove trailing 'r' after closing brackets/quotes: ['서울']r -> ['서울']
        content = re.sub(r"(\]|'|\")r(?=\s*[,\)])", r'\1', content)
        # Remove 'r' at end of identifiers: weatherr -> weather
        content = re.sub(r'(\w)r(?=\s*[,\)\]])', r'\1', content)

        # Check if it looks like a pythonic tool call
        if not (content.startswith('[') and content.endswith(']')):
            return []

        try:
            # Parse as Python AST
            tree = ast.parse(content, mode='eval')
            if not isinstance(tree.body, ast.List):
                return []

            tool_calls = []
            for idx, call_node in enumerate(tree.body.elts):
                if not isinstance(call_node, ast.Call):
                    continue

                # Extract function name
                if isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                else:
                    continue

                # Extract arguments
                args_dict = {}
                for keyword in call_node.keywords:
                    arg_name = keyword.arg
                    try:
                        # Try to evaluate the value
                        arg_value = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        # If it fails, check if it's a Name node (unquoted identifier)
                        if isinstance(keyword.value, ast.Name):
                            # Treat as string
                            arg_value = keyword.value.id
                        else:
                            # Skip this argument if we can't parse it
                            continue
                    args_dict[arg_name] = arg_value

                tool_calls.append({
                    'name': func_name,
                    'args': args_dict,
                    'id': f'call_{idx}',
                    'type': 'tool_call'
                })

            return tool_calls
        except Exception as e:
            print(f"[DEBUG] Failed to parse pythonic tool calls: {e}")
            return []

    def _handle_tool_calls(self, response: Any) -> str:
        """
        Handle tool calls from LLM response.

        Args:
            response: Response from LLM

        Returns:
            Final response text after executing tools
        """
        # Debug logging - show ALL attributes
        print(f"\n[DEBUG] ===== LLM Response Debug =====")
        print(f"[DEBUG] Response type: {type(response)}")
        print(f"[DEBUG] Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        print(f"[DEBUG] Has tool_calls: {hasattr(response, 'tool_calls')}")

        if hasattr(response, 'tool_calls'):
            print(f"[DEBUG] Tool calls type: {type(response.tool_calls)}")
            print(f"[DEBUG] Tool calls length: {len(response.tool_calls) if response.tool_calls else 0}")
            print(f"[DEBUG] Tool calls content: {response.tool_calls}")

        if hasattr(response, 'content'):
            print(f"[DEBUG] Content type: {type(response.content)}")
            print(f"[DEBUG] Content: {response.content}")

        if hasattr(response, 'additional_kwargs'):
            print(f"[DEBUG] Additional kwargs: {response.additional_kwargs}")

        if hasattr(response, 'response_metadata'):
            print(f"[DEBUG] Response metadata: {response.response_metadata}")

        # Try to get the raw response
        if hasattr(response, 'raw'):
            print(f"[DEBUG] Raw response: {response.raw}")

        print(f"[DEBUG] Full response object: {response}")
        print(f"[DEBUG] ===========================\n")

        # If no tool calls but content exists, try to parse pythonic format (for Gemma 3)
        if (not hasattr(response, 'tool_calls') or not response.tool_calls) and \
           hasattr(response, 'content') and isinstance(response.content, str):
            print(f"[DEBUG] Attempting to parse pythonic tool calls from content...")
            parsed_calls = self._parse_pythonic_tool_calls(response.content)
            if parsed_calls:
                print(f"[DEBUG] Successfully parsed {len(parsed_calls)} tool calls")
                # Manually set tool_calls on response
                response.tool_calls = parsed_calls
                # Clear content when tool calls are present
                response.content = ""

        # Check if response has tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            results = []
            for tool_call in response.tool_calls:
                print(f"\n[DEBUG] Processing tool call: {tool_call}")
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})
                print(f"[DEBUG] Tool name: {tool_name}, args: {tool_args}")

                # Find and execute the tool
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            print(f"[DEBUG] Executing tool: {tool_name}")
                            # Use invoke instead of run for StructuredTool
                            result = tool.invoke(tool_args)
                            print(f"[DEBUG] Tool result: {result[:200] if result else 'None'}...")
                            results.append(f"[{tool_name}] {result}")
                        except Exception as e:
                            print(f"[DEBUG] Tool error: {e}")
                            import traceback
                            traceback.print_exc()
                            results.append(f"[{tool_name}] Error: {str(e)}")
                        break

            if results:
                return "\n\n".join(results)

        # Return content if no tool calls
        return str(response.content) if hasattr(response, 'content') else str(response)

    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response (non-streaming)

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        try:
            # Invoke chain with message and history
            response = self.chain.invoke({
                "input": message,
                "chat_history": self.chat_history
            })

            # Handle tool calls if present
            if self.use_agent:
                response_text = self._handle_tool_calls(response)
            else:
                response_text = str(response.content)

            # Update chat history
            self.chat_history.append(HumanMessage(content=message))
            enriched_response, _ = self._enrich_with_weather(response_text)
            self.chat_history.append(AIMessage(content=enriched_response))

            # Return response content as string
            return enriched_response
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
        try:
            full_response = ""
            response_obj = None
            # Accumulate tool call chunks for streaming
            tool_call_accumulator = {}

            # Stream the response
            for chunk in self.chain.stream({
                "input": message,
                "chat_history": self.chat_history
            }):
                # Accumulate tool call chunks
                if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                    print(f"[DEBUG STREAM] Got tool_call_chunks: {chunk.tool_call_chunks}")
                    for tc_chunk in chunk.tool_call_chunks:
                        # Use index as the primary key, fall back to id
                        tc_index = tc_chunk.get('index', 0)
                        tc_id = tc_chunk.get('id')

                        if tc_index not in tool_call_accumulator:
                            tool_call_accumulator[tc_index] = {
                                'name': tc_chunk.get('name', ''),
                                'args': tc_chunk.get('args', ''),
                                'id': tc_id,
                                'type': 'tool_call'
                            }
                            print(f"[DEBUG STREAM] Created new accumulator for index {tc_index}: {tool_call_accumulator[tc_index]}")
                        else:
                            # Merge args (they come as string chunks)
                            if 'args' in tc_chunk and tc_chunk['args']:
                                tool_call_accumulator[tc_index]['args'] += tc_chunk['args']
                                print(f"[DEBUG STREAM] Appended args to index {tc_index}: {tc_chunk['args']}")
                            if 'name' in tc_chunk and tc_chunk['name']:
                                tool_call_accumulator[tc_index]['name'] = tc_chunk['name']
                                print(f"[DEBUG STREAM] Set name for index {tc_index}: {tc_chunk['name']}")
                            # Update id if it's set (first chunk has it)
                            if tc_id and not tool_call_accumulator[tc_index]['id']:
                                tool_call_accumulator[tc_index]['id'] = tc_id
                                print(f"[DEBUG STREAM] Set id for index {tc_index}: {tc_id}")

                # Save the last chunk for metadata
                response_obj = chunk

                if hasattr(chunk, 'content'):
                    content = str(chunk.content)
                    # Only yield non-empty content
                    if content:
                        full_response += content
                        yield content

            # Handle tool calls if present (after streaming)
            if self.use_agent and tool_call_accumulator:
                # Reconstruct tool calls from accumulated chunks
                try:
                    import json
                    print(f"[DEBUG STREAM] Final accumulator: {tool_call_accumulator}")
                    reconstructed_tool_calls = []
                    for tc_id, tc_data in tool_call_accumulator.items():
                        # Parse the accumulated args string as JSON
                        args_str = tc_data['args']
                        print(f"[DEBUG STREAM] Processing {tc_id}, args_str: {args_str}")
                        if args_str:
                            try:
                                args_dict = json.loads(args_str)
                                print(f"[DEBUG STREAM] Parsed args: {args_dict}")
                            except json.JSONDecodeError as e:
                                print(f"[DEBUG STREAM] Failed to parse args JSON: {args_str}, error: {e}")
                                args_dict = {}
                        else:
                            args_dict = {}

                        reconstructed_tool_calls.append({
                            'name': tc_data['name'],
                            'args': args_dict,
                            'id': tc_id,
                            'type': 'tool_call'
                        })

                    print(f"[DEBUG STREAM] Reconstructed tool calls: {reconstructed_tool_calls}")
                    # Create a mock response object with tool_calls
                    if response_obj:
                        response_obj.tool_calls = reconstructed_tool_calls
                        tool_results = self._handle_tool_calls(response_obj)
                        if tool_results and tool_results != full_response:
                            yield "\n\n" + tool_results
                            full_response = full_response + "\n\n" + tool_results
                except Exception as e:
                    print(f"[DEBUG STREAM] Error reconstructing tool calls: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.use_agent and response_obj:
                # Fallback to old method
                tool_results = self._handle_tool_calls(response_obj)
                if tool_results and tool_results != full_response:
                    yield "\n\n" + tool_results
                    full_response = full_response + "\n\n" + tool_results

            enriched_response, additions = self._enrich_with_weather(full_response)
            for index, addition in enumerate(additions):
                prefix = "\n\n" if index == 0 else "\n"
                yield prefix + addition

            # Update chat history after streaming is complete
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=enriched_response))

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
