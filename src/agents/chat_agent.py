"""
LangChain Chat Agent with Ollama (via OpenAI-compatible API)
"""
import re
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
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Chat Agent

        Args:
            model: Ollama model name (default: llama3.2)
            temperature: Temperature for response generation (default: 0.7)
            base_url: Ollama API base URL (default: http://localhost:11434/v1)
            api_key: API key (Ollama doesn't require a real key, default: "ollama")
            system_prompt: Custom system prompt (default: simple assistant prompt)
        """
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.chat_history: List[Any] = []
        self.weather_service = WeatherService()

        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Always be conversational and friendly."

        # Store system prompt for debugging
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

            # Update chat history
            self.chat_history.append(HumanMessage(content=message))
            response_text = str(response.content)
            enriched_response, _ = self._enrich_with_weather(response_text)
            self.chat_history.append(AIMessage(content=enriched_response))

            # Return response content as string
            return enriched_response
        except Exception as e:
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

            # Stream the response
            for chunk in self.chain.stream({
                "input": message,
                "chat_history": self.chat_history
            }):
                if hasattr(chunk, 'content'):
                    content = str(chunk.content)
                    # Only yield non-empty content
                    if content:
                        full_response += content
                        yield content

            enriched_response, additions = self._enrich_with_weather(full_response)
            for index, addition in enumerate(additions):
                prefix = "\n\n" if index == 0 else "\n"
                yield prefix + addition

            # Update chat history after streaming is complete
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=enriched_response))

        except Exception as e:
            yield f"\n오류가 발생했습니다: {str(e)}"

    def reset_memory(self):
        """Reset conversation history"""
        self.chat_history.clear()

    def get_system_prompt(self) -> str:
        """Get the current system prompt for debugging"""
        return self.system_prompt
