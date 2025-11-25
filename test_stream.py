#!/usr/bin/env python3
"""
Test streaming chat completions API with multi-turn conversation support
"""
import sys
import json
import uuid
import requests
from typing import List, Dict, Optional


class ChatSession:
    """Manages a chat session with the API"""

    def __init__(self, port: int = 24000, session_name: Optional[str] = None):
        self.port = port
        self.url = f"http://localhost:{port}/v1/chat/completions"

        # Use provided session name or generate one
        if session_name:
            self.session_id = f"test-{session_name}"
            self.session_name = session_name
        else:
            self.session_id = f"test-{uuid.uuid4()}"
            self.session_name = "auto-generated"

        self.messages: List[Dict[str, str]] = []

        print(f"=" * 60)
        print(f"Chat Session Started")
        print(f"=" * 60)
        print(f"Session Name: {self.session_name}")
        print(f"Session ID: {self.session_id[:30]}...")
        print(f"Port: {port}")
        print(f"=" * 60)
        print("\nCommands:")
        print("  - 'quit', 'exit', 'bye': Exit")
        print("  - 'clear', 'reset': Clear chat history")
        print("  - 'session': Show session info")
        print("  - 'history': Show message history")
        print()

    def send_message(self, user_message: str) -> str:
        """Send a message and return the response"""
        # Add session ID in system message
        messages = [
            {"role": "system", "content": f"session_id: {self.session_id}"}
        ]
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_message})

        data = {
            "messages": messages,
            "stream": True
        }

        print(f"User: {user_message}")
        print("Agent: ", end="", flush=True)

        full_response = ""

        try:
            response = requests.post(
                self.url,
                json=data,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')

                # Skip empty lines and [DONE] marker
                if not line.startswith('data: '):
                    continue

                if line.strip() == 'data: [DONE]':
                    break

                # Parse SSE data
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix

                    # Extract content from delta
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                except json.JSONDecodeError:
                    continue

            print("\n")

            # Update message history
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": full_response})

            return full_response

        except requests.exceptions.RequestException as e:
            print(f"\nError: {e}", file=sys.stderr)
            return ""

    def clear_history(self):
        """Clear message history"""
        self.messages.clear()

        # Reset session on server
        try:
            reset_url = f"http://localhost:{self.port}/v1/sessions/{self.session_id}/reset"
            requests.post(reset_url, timeout=5)
            print("✓ Chat history cleared\n")
        except Exception as e:
            print(f"Warning: Could not reset session on server: {e}\n")

    def show_session_info(self):
        """Show session information"""
        print("\n" + "=" * 60)
        print("Session Information")
        print("=" * 60)
        print(f"Session Name: {self.session_name}")
        print(f"Session ID: {self.session_id}")
        print(f"Message Count: {len(self.messages)}")
        print(f"API Endpoint: {self.url}")
        print("=" * 60 + "\n")

    def show_history(self):
        """Show message history"""
        if not self.messages:
            print("\nNo message history\n")
            return

        print("\n" + "=" * 60)
        print("Message History")
        print("=" * 60)
        for i, msg in enumerate(self.messages, 1):
            role = msg['role'].upper()
            content = msg['content'][:100] + ("..." if len(msg['content']) > 100 else "")
            print(f"{i}. [{role}] {content}")
        print("=" * 60 + "\n")

    def chat_loop(self):
        """Interactive chat loop"""
        try:
            while True:
                try:
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("Goodbye!")
                        break

                    if user_input.lower() in ['clear', 'reset']:
                        self.clear_history()
                        continue

                    if user_input.lower() == 'session':
                        self.show_session_info()
                        continue

                    if user_input.lower() == 'history':
                        self.show_history()
                        continue

                    # Send message
                    self.send_message(user_input)

                except EOFError:
                    print("\nGoodbye!")
                    break

        except KeyboardInterrupt:
            print("\n\nGoodbye!")


def main():
    """Main function"""
    port = 24000
    session_name = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage: python test_stream.py [--session NAME] [--port PORT] [message]")
            print("\nOptions:")
            print("  --session NAME    Session name for conversation history")
            print("  --port PORT       API server port (default: 24000)")
            print("  message           Single message to send (interactive mode if not provided)")
            print("\nExamples:")
            print("  python test_stream.py                               # Interactive mode")
            print("  python test_stream.py --session work                # Use 'work' session")
            print("  python test_stream.py --port 24000 --session work  # Port and session")
            print("  python test_stream.py --session work \"안녕하세요\"   # Single message with session")
            sys.exit(0)

        # Parse named arguments
        i = 1
        message_parts = []
        while i < len(sys.argv):
            if sys.argv[i] == '--session' and i + 1 < len(sys.argv):
                session_name = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--port' and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid port '{sys.argv[i + 1]}'")
                    sys.exit(1)
            else:
                message_parts.append(sys.argv[i])
                i += 1

        # Check if there's a message
        if message_parts:
            prompt = " ".join(message_parts)
            session = ChatSession(port=port, session_name=session_name)
            session.send_message(prompt)
            return

    # Interactive mode
    session = ChatSession(port=port, session_name=session_name)
    session.chat_loop()


if __name__ == "__main__":
    main()
