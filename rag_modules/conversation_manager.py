"""
对话历史管理模块
提供多轮对话历史管理功能
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """对话消息"""
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        return cls(**data)

    def to_langchain_format(self) -> Dict[str, str]:
        """转换为 LangChain 格式"""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """添加消息"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = time.time()

    def get_messages(self, limit: int = None) -> List[Message]:
        """获取消息列表"""
        if limit is None or limit >= len(self.messages):
            return self.messages
        return self.messages[-limit:]

    def get_langchain_messages(self, limit: int = None) -> List[Dict[str, str]]:
        """获取 LangChain 格式的消息列表"""
        messages = self.get_messages(limit)
        return [msg.to_langchain_format() for msg in messages]

    def clear(self):
        """清空对话历史"""
        self.messages.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """从字典创建"""
        messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            messages=messages,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {})
        )


class ConversationManager:
    """对话管理器"""

    def __init__(
        self,
        history_dir: str = "./conversation_history",
        max_history_length: int = 20,
        auto_save: bool = True
    ):
        """
        初始化对话管理器

        Args:
            history_dir: 历史记录保存目录
            max_history_length: 最大历史消息数量
            auto_save: 是否自动保存
        """
        self.history_dir = Path(history_dir)
        self.max_history_length = max_history_length
        self.auto_save = auto_save
        self.current_session: Optional[ConversationSession] = None
        self._ensure_history_dir()

    def _ensure_history_dir(self):
        """确保历史目录存在"""
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: str = None) -> ConversationSession:
        """
        创建新会话

        Args:
            session_id: 会话ID，默认自动生成

        Returns:
            新会话
        """
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"

        self.current_session = ConversationSession(session_id=session_id)
        logger.info(f"创建新会话: {session_id}")
        return self.current_session

    def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        """
        获取或创建会话

        Args:
            session_id: 会话ID

        Returns:
            会话
        """
        if self.current_session is None:
            return self.create_session(session_id)
        return self.current_session

    def add_user_message(self, content: str, metadata: Dict[str, Any] = None):
        """添加用户消息"""
        session = self.get_or_create_session()
        session.add_message("user", content, metadata)
        self._trim_history()
        if self.auto_save:
            self.save_session()

    def add_assistant_message(self, content: str, metadata: Dict[str, Any] = None):
        """添加助手消息"""
        session = self.get_or_create_session()
        session.add_message("assistant", content, metadata)
        self._trim_history()
        if self.auto_save:
            self.save_session()

    def _trim_history(self):
        """修剪历史记录"""
        if self.current_session and len(self.current_session.messages) > self.max_history_length:
            # 保留最近的对话
            self.current_session.messages = self.current_session.messages[-self.max_history_length:]
            logger.debug(f"历史记录已修剪至 {self.max_history_length} 条")

    def get_context_messages(self, limit: int = None) -> List[Dict[str, str]]:
        """
        获取上下文消息（LangChain 格式）

        Args:
            limit: 限制消息数量

        Returns:
            消息列表
        """
        if self.current_session is None:
            return []
        return self.current_session.get_langchain_messages(limit)

    def get_last_n_turns(self, n: int = 2) -> List[Dict[str, str]]:
        """
        获取最近 n 轮对话（每轮包含用户和助手消息）

        Args:
            n: 轮数

        Returns:
            消息列表
        """
        if self.current_session is None:
            return []

        # 每轮对话包含用户和助手两条消息
        messages = self.current_session.get_messages(limit=n * 2)
        return [msg.to_langchain_format() for msg in messages]

    def clear_current_session(self):
        """清空当前会话"""
        if self.current_session:
            self.current_session.clear()
            logger.info("当前会话已清空")

    def save_session(self):
        """保存当前会话"""
        if self.current_session is None:
            return

        session_file = self.history_dir / f"{self.current_session.session_id}.json"
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(self.current_session.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"会话已保存: {session_file}")
        except Exception as e:
            logger.warning(f"保存会话失败: {e}")

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        加载会话

        Args:
            session_id: 会话ID

        Returns:
            会话
        """
        session_file = self.history_dir / f"{session_id}.json"
        if not session_file.exists():
            logger.warning(f"会话文件不存在: {session_file}")
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_session = ConversationSession.from_dict(data)
            logger.info(f"会话已加载: {session_id}")
            return self.current_session
        except Exception as e:
            logger.warning(f"加载会话失败: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Returns:
            会话列表
        """
        sessions = []
        for session_file in self.history_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "message_count": len(data.get("messages", [])),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
            except Exception:
                continue

        # 按更新时间排序
        sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功
        """
        session_file = self.history_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None
            logger.info(f"会话已删除: {session_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        sessions = self.list_sessions()
        total_messages = sum(s["message_count"] for s in sessions)

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "current_session_id": self.current_session.session_id if self.current_session else None,
            "current_session_messages": len(self.current_session.messages) if self.current_session else 0,
            "history_dir": str(self.history_dir)
        }
