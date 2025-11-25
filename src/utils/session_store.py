"""
Session Store for managing chat history across different backends.
Supports in-memory storage (single worker) and Redis (multi-worker).
"""
import os
import time
from typing import Dict
from datetime import datetime, timedelta
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Environment configuration
SESSION_STORE_TYPE = os.getenv("SESSION_STORE_TYPE", "memory")  # memory | redis
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))  # 30 minutes default


class SessionStore:
    """
    Session history storage factory.
    Supports switching between memory-based and Redis-based storage via environment variables.
    """

    def __init__(self):
        self.store_type = SESSION_STORE_TYPE
        self._memory_store: Dict[str, ChatMessageHistory] = {}
        self._access_times: Dict[str, float] = {}  # Track last access time for TTL
        self.ttl_seconds = SESSION_TTL_SECONDS

        print(f"[SessionStore] Initialized with type: {self.store_type}, TTL: {self.ttl_seconds}s")

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get chat history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            BaseChatMessageHistory implementation based on store_type
        """
        if self.store_type == "redis":
            return self._get_redis_history(session_id)
        else:
            return self._get_memory_history(session_id)

    def _get_memory_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create in-memory chat history.
        Note: Only works with single worker process.

        Args:
            session_id: Unique session identifier

        Returns:
            ChatMessageHistory instance
        """
        # Update access time
        self._access_times[session_id] = time.time()

        # Check if session exists
        if session_id not in self._memory_store:
            self._memory_store[session_id] = ChatMessageHistory()
            print(f"[SessionStore] Created new memory session: {session_id}")

        return self._memory_store[session_id]

    def _get_redis_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get Redis-based chat history.
        Supports multi-worker deployment.

        Args:
            session_id: Unique session identifier

        Returns:
            RedisChatMessageHistory instance
        """
        from langchain_community.chat_message_histories import RedisChatMessageHistory

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        return RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            ttl=self.ttl_seconds  # Redis native TTL support
        )

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from memory store.
        Only applicable for memory-based storage.

        Returns:
            Number of sessions cleaned up
        """
        if self.store_type != "memory":
            return 0

        current_time = time.time()
        expired_sessions = [
            sid for sid, last_access in self._access_times.items()
            if current_time - last_access > self.ttl_seconds
        ]

        for sid in expired_sessions:
            if sid in self._memory_store:
                del self._memory_store[sid]
            del self._access_times[sid]

        if expired_sessions:
            print(f"[SessionStore] Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_active_session_count(self) -> int:
        """Get number of active sessions"""
        if self.store_type == "memory":
            return len(self._memory_store)
        else:
            # For Redis, we can't easily count without scanning all keys
            return -1  # Unknown

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.

        Args:
            session_id: Session to delete

        Returns:
            True if session was deleted, False if not found
        """
        if self.store_type == "memory":
            deleted = False
            if session_id in self._memory_store:
                del self._memory_store[session_id]
                deleted = True
            if session_id in self._access_times:
                del self._access_times[session_id]
                deleted = True
            return deleted
        else:
            # For Redis, the history object handles cleanup
            # We just need to clear the messages
            try:
                history = self._get_redis_history(session_id)
                history.clear()
                return True
            except Exception as e:
                print(f"[SessionStore] Error deleting Redis session {session_id}: {e}")
                return False


# Global session store instance
session_store = SessionStore()
