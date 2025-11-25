"""
JSON Fixer - LLM-based JSON syntax error correction
"""
import os
import re
import json
from typing import Optional, Any
from openai import OpenAI


class JsonFixer:
    """LLM-based JSON fixer implemented as a singleton."""
    __instance = None
    
    def __new__(cls, *args, **kwargs):
        # Ensure only one instance is created
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        # Prevent reinitialization for subsequent instantiations
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        try:
            # Prefer JsonFixer-specific env vars, then fall back to shared LLM settings
            self.model = os.getenv("JsonFixer_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
            base_url = os.getenv("JsonFixer_URL") or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
            api_key = os.getenv("JsonFixer_KEY") or os.getenv("LLM_API_KEY", "local")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self.temperature = float(os.getenv("JsonFixer_TEMPERATURE", 0.2) )

        except Exception as e:
            print(f"Error initializing JsonFixer: {str(e)}")
            raise e

    @classmethod
    def fix_json_with_llm(
        cls,
        invalid_json: str,
        error_message: str,
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Use LLM to fix JSON syntax errors.

        Args:
            invalid_json: The invalid JSON string
            error_message: The error message from JSONDecodeError
            handler: Optional callback handler (e.g., Langfuse)
            max_retries: Maximum number of correction attempts

        Returns:
            Corrected JSON string, or None if correction failed
        """
        print(f"\n[JSON Fixer] Attempting to fix JSON syntax errors...")

        def build_prompt(current_json: str, current_error: str) -> str:
            return f"""
    /no_think
    The following JSON has syntax errors:

    {current_json}

    Error: {current_error}

    Please fix the JSON syntax errors and return ONLY the corrected JSON (no explanations).
    ## Guide:
    - Trailing commas are not allowed in JSON.
    """

        for attempt in range(max_retries):
            try:
                print(f"[JSON Fixer] Invalid JSON:\n{invalid_json}")
                print(f"[JSON Fixer] Error: {error_message}")
                print(f"[JSON Fixer] Correction attempt {attempt + 1}/{max_retries}")

                # Invoke OpenAI-compatible chat completion using stored LLM config
                instance = cls()
                client = instance._client

                correction_response = client.chat.completions.create(
                    model=instance.model,
                    messages=[
                        {"role": "user", "content": build_prompt(invalid_json, error_message)}
                    ],
                    temperature=instance.temperature,
                )

                # Extract content from response
                corrected_content = ""
                if correction_response and correction_response.choices:
                    corrected_content = correction_response.choices[0].message.content or ""

                if not corrected_content:
                    print("[JSON Fixer] LLM returned empty content")
                    return None

                # Directly use the returned content as JSON
                corrected_json = corrected_content.strip().rpartition("</think>")[2]

                # Validate the corrected JSON
                try:
                    json.loads(corrected_json)
                    print(f"[JSON Fixer] Successfully corrected JSON")
                    return corrected_json
                except json.JSONDecodeError as validation_error:
                    print(f"[JSON Fixer] Corrected JSON still invalid: {validation_error}")
                    # Update the invalid_json for next iteration
                    invalid_json = corrected_json
                    error_message = str(validation_error)
                    continue

            except Exception as llm_error:
                print(f"[JSON Fixer] LLM correction failed: {str(llm_error)}")
                return None

        print(f"[JSON Fixer] Failed to fix JSON after {max_retries} attempts")
        return None


def extract_json_from_tags(
    content: str,
    tag_name: str = "AGENT_CALL"
) -> Optional[str]:
    """
    Extract JSON content from XML-like tags.

    Args:
        content: Content containing tagged JSON
        tag_name: Name of the tag (default: "AGENT_CALL")

    Returns:
        Extracted JSON string, or None if not found
    """
    pattern = rf'<{tag_name}>\s*(.*?)\s*</{tag_name}>'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def validate_and_parse_json(json_str: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Validate and parse JSON string.

    Args:
        json_str: JSON string to validate

    Returns:
        Tuple of (parsed_json, error_message)
        - If successful: (parsed_json, None)
        - If failed: (None, error_message)
    """
    try:
        parsed = json.loads(json_str)
        return parsed, None
    except json.JSONDecodeError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
