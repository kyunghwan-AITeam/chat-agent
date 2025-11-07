"""
Utility wrapper for interacting with the Open-Meteo weather MCP tool.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Optional


class WeatherServiceError(Exception):
    """Raised when the weather service cannot fulfill a request."""


class WeatherService:
    """Simple client wrapper around the Open-Meteo MCP tool."""

    def __init__(self, language: str = "ko", timezone: str = "auto") -> None:
        self.language = language
        self.timezone = timezone
        self._tool: Optional[Callable[..., Dict[str, Any]]] = None

    def get_current_weather(
        self,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fetch current weather information."""
        tool = self._load_tool()
        result = tool(
            location=location,
            latitude=latitude,
            longitude=longitude,
            language=self.language,
            timezone=self.timezone,
        )

        if "error" in result:
            raise WeatherServiceError(result["error"])

        return result

    def _load_tool(self) -> Callable[..., Dict[str, Any]]:
        """Lazy import of the weather MCP tool."""
        if self._tool is None:
            try:
                module = import_module("open_meteo_weather_mcp")
                self._tool = getattr(module, "get_current_weather")
            except ModuleNotFoundError as exc:
                raise WeatherServiceError(
                    "open-meteo-weather-mcp 패키지가 설치되어 있지 않습니다. "
                    "별도의 프로젝트를 설치하고 MCP 서버를 실행하세요."
                ) from exc

        return self._tool
