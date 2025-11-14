"""
Langfuse Configuration and Integration
"""
import os
import yaml
from typing import Optional, Dict, Any, List
from pathlib import Path
from langfuse import Langfuse


class LangfuseConfig:
    """Langfuse configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Langfuse configuration

        Args:
            config_path: Path to langfuse_config.yaml (default: project root)
        """
        self.enabled = False
        self.debug = False
        self.flush_at = 15
        self.flush_interval = 0.5
        self.tags = []
        self.client: Optional[Langfuse] = None

        # Load configuration
        self._load_config(config_path)

        # Initialize Langfuse if enabled
        if self.enabled:
            self._initialize_client()

    def _load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file and environment variables"""
        # Load from YAML
        if config_path is None:
            # Look for config in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "langfuse_config.yaml"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                langfuse_config = config.get('langfuse', {})

                self.enabled = langfuse_config.get('enabled', False)
                self.debug = langfuse_config.get('debug', False)
                self.flush_at = langfuse_config.get('flush_at', 15)
                self.flush_interval = langfuse_config.get('flush_interval', 0.5)
                self.tags = langfuse_config.get('tags', [])

                # Replace environment variable placeholders in tags
                self.tags = [self._replace_env_vars(tag) for tag in self.tags]

        # Environment variables override YAML config
        if os.getenv('LANGFUSE_ENABLED', '').lower() == 'true':
            self.enabled = True
        elif os.getenv('LANGFUSE_ENABLED', '').lower() == 'false':
            self.enabled = False

        if os.getenv('LANGFUSE_DEBUG', '').lower() == 'true':
            self.debug = True

    def _replace_env_vars(self, text: str) -> str:
        """Replace ${VAR_NAME:default} with environment variable value"""
        import re

        def replacer(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_expr, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', replacer, text)

    def _initialize_client(self):
        """Initialize Langfuse client"""
        try:
            # Get credentials from environment
            # Support both LANGFUSE_HOST and LANGFUSE_BASE_URL
            host = os.getenv('LANGFUSE_HOST') or os.getenv('LANGFUSE_BASE_URL')
            public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
            secret_key = os.getenv('LANGFUSE_SECRET_KEY')

            if not host:
                print("Warning: LANGFUSE_HOST or LANGFUSE_BASE_URL not set, disabling Langfuse")
                self.enabled = False
                return

            # Initialize client
            self.client = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key,
                debug=self.debug,
                flush_at=self.flush_at,
                flush_interval=self.flush_interval
            )

            if self.debug:
                print(f"Langfuse initialized: {host}")
                print(f"Tags: {self.tags}")

        except Exception as e:
            print(f"Error initializing Langfuse: {e}")
            self.enabled = False
            self.client = None

    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled"""
        return self.enabled and self.client is not None

    def get_tags(self) -> List[str]:
        """Get configured tags"""
        return self.tags.copy()

    def start_trace(
        self,
        name: str,
        model: str,
        input_messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """
        Start tracing an LLM call to Langfuse

        Args:
            name: Name of the trace (e.g., "chat_completion")
            model: Model name
            input_messages: List of input messages
            metadata: Additional metadata
            session_id: Session ID for grouping traces

        Returns:
            Generation object to be used with end_trace
        """
        if not self.is_enabled():
            return None

        try:
            # Prepare tags and metadata
            trace_tags = self.get_tags()
            if metadata and 'tags' in metadata:
                trace_tags.extend(metadata['tags'])

            # Add tags and session to metadata
            full_metadata = metadata.copy() if metadata else {}
            full_metadata['tags'] = trace_tags
            if session_id:
                full_metadata['session_id'] = session_id

            # Start generation (without output)
            # The new Langfuse SDK creates a trace automatically
            generation = self.client.start_generation(
                name=name,
                model=model,
                input=input_messages,
                metadata=full_metadata
            )

            if self.debug:
                print(f"Langfuse trace started: {name}")

            return generation

        except Exception as e:
            if self.debug:
                print(f"Error starting Langfuse trace: {e}")
                import traceback
                traceback.print_exc()
            return None

    def end_trace(
        self,
        generation,
        output: str,
        usage: Optional[Dict[str, int]] = None
    ):
        """
        End tracing an LLM call to Langfuse

        Args:
            generation: Generation object from start_trace
            output: LLM output
            usage: Token usage information
        """
        if not self.is_enabled() or generation is None:
            return

        try:
            # Update with output and usage
            update_params = {"output": output}

            if usage:
                update_params["usage"] = {
                    "input": usage.get("input", 0),
                    "output": usage.get("output", 0),
                    "total": usage.get("total", 0)
                }

                if self.debug:
                    print(f"Updating generation with usage: {update_params['usage']}")

            generation.update(**update_params)
            generation.end()

            if self.debug:
                print(f"Langfuse trace ended")

        except Exception as e:
            if self.debug:
                print(f"Error ending Langfuse trace: {e}")
                import traceback
                traceback.print_exc()

    def trace_llm_call(
        self,
        name: str,
        model: str,
        input_messages: List[Dict[str, Any]],
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Trace an LLM call to Langfuse (legacy method - uses start/end internally)

        Args:
            name: Name of the trace (e.g., "chat_completion")
            model: Model name
            input_messages: List of input messages
            output: LLM output
            metadata: Additional metadata
            usage: Token usage information
            session_id: Session ID for grouping traces
            **kwargs: Additional trace parameters
        """
        generation = self.start_trace(name, model, input_messages, metadata, session_id)
        if generation:
            self.end_trace(generation, output, usage)

    def flush(self):
        """Flush pending traces to Langfuse"""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                if self.debug:
                    print(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Shutdown Langfuse client"""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                if self.debug:
                    print(f"Error shutting down Langfuse: {e}")


# Global Langfuse configuration instance
_langfuse_config: Optional[LangfuseConfig] = None


def get_langfuse_config() -> LangfuseConfig:
    """Get or create global Langfuse configuration"""
    global _langfuse_config
    if _langfuse_config is None:
        _langfuse_config = LangfuseConfig()
    return _langfuse_config


def init_langfuse(config_path: Optional[str] = None) -> LangfuseConfig:
    """
    Initialize Langfuse with optional config path

    Args:
        config_path: Path to langfuse_config.yaml

    Returns:
        LangfuseConfig instance
    """
    global _langfuse_config
    _langfuse_config = LangfuseConfig(config_path)
    return _langfuse_config
