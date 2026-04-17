"""
对话管理模块单元测试
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_modules.conversation_manager import (
    Message,
    ConversationSession,
    ConversationManager
)


class TestMessage(unittest.TestCase):
    """测试 Message 类"""

    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(role="user", content="测试消息")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "测试消息")
        self.assertIsNotNone(msg.timestamp)
        self.assertEqual(msg.metadata, {})

    def test_message_with_metadata(self):
        """测试带元数据的消息"""
        metadata = {"query_type": "detail", "route": "list"}
        msg = Message(role="assistant", content="回答", metadata=metadata)
        self.assertEqual(msg.metadata, metadata)

    def test_message_to_dict(self):
        """测试消息转字典"""
        msg = Message(role="user", content="测试")
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["role"], "user")
        self.assertEqual(msg_dict["content"], "测试")
        self.assertIn("timestamp", msg_dict)

    def test_message_from_dict(self):
        """测试从字典创建消息"""
        data = {
            "role": "assistant",
            "content": "回答内容",
            "timestamp": 1234567890.0,
            "metadata": {"key": "value"}
        }
        msg = Message.from_dict(data)
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "回答内容")
        self.assertEqual(msg.timestamp, 1234567890.0)

    def test_message_to_langchain_format(self):
        """测试转换为 LangChain 格式"""
        msg = Message(role="user", content="问题")
        lc_format = msg.to_langchain_format()
        self.assertEqual(lc_format, {"role": "user", "content": "问题"})


class TestConversationSession(unittest.TestCase):
    """测试 ConversationSession 类"""

    def test_session_creation(self):
        """测试会话创建"""
        session = ConversationSession(session_id="test_session")
        self.assertEqual(session.session_id, "test_session")
        self.assertEqual(len(session.messages), 0)
        self.assertIsNotNone(session.created_at)

    def test_add_message(self):
        """测试添加消息"""
        session = ConversationSession(session_id="test")
        session.add_message("user", "问题1")
        session.add_message("assistant", "回答1")
        
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].role, "user")
        self.assertEqual(session.messages[1].role, "assistant")

    def test_get_messages_with_limit(self):
        """测试获取限制数量的消息"""
        session = ConversationSession(session_id="test")
        for i in range(5):
            session.add_message("user", f"问题{i}")
        
        messages = session.get_messages(limit=3)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].content, "问题2")
        self.assertEqual(messages[2].content, "问题4")

    def test_get_langchain_messages(self):
        """测试获取 LangChain 格式消息"""
        session = ConversationSession(session_id="test")
        session.add_message("user", "问题")
        session.add_message("assistant", "回答")
        
        lc_messages = session.get_langchain_messages()
        self.assertEqual(len(lc_messages), 2)
        self.assertEqual(lc_messages[0], {"role": "user", "content": "问题"})
        self.assertEqual(lc_messages[1], {"role": "assistant", "content": "回答"})

    def test_clear_session(self):
        """测试清空会话"""
        session = ConversationSession(session_id="test")
        session.add_message("user", "问题")
        session.clear()
        
        self.assertEqual(len(session.messages), 0)

    def test_session_to_dict_and_from_dict(self):
        """测试会话序列化和反序列化"""
        session = ConversationSession(session_id="test", metadata={"key": "value"})
        session.add_message("user", "问题")
        session.add_message("assistant", "回答")
        
        session_dict = session.to_dict()
        restored = ConversationSession.from_dict(session_dict)
        
        self.assertEqual(restored.session_id, "test")
        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.metadata, {"key": "value"})


class TestConversationManager(unittest.TestCase):
    """测试 ConversationManager 类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ConversationManager(
            history_dir=self.temp_dir,
            max_history_length=10,
            auto_save=False
        )

    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_session(self):
        """测试创建会话"""
        session = self.manager.create_session()
        self.assertIsNotNone(session)
        self.assertIsNotNone(session.session_id)
        self.assertEqual(self.manager.current_session, session)

    def test_create_session_with_id(self):
        """测试创建指定ID的会话"""
        session = self.manager.create_session("custom_id")
        self.assertEqual(session.session_id, "custom_id")

    def test_get_or_create_session(self):
        """测试获取或创建会话"""
        session1 = self.manager.get_or_create_session()
        session2 = self.manager.get_or_create_session()
        self.assertEqual(session1, session2)

    def test_add_messages(self):
        """测试添加消息"""
        self.manager.add_user_message("问题1")
        self.manager.add_assistant_message("回答1")
        
        messages = self.manager.get_context_messages()
        self.assertEqual(len(messages), 2)

    def test_trim_history(self):
        """测试历史修剪"""
        manager = ConversationManager(
            history_dir=self.temp_dir,
            max_history_length=4,
            auto_save=False
        )
        
        for i in range(6):
            manager.add_user_message(f"问题{i}")
            manager.add_assistant_message(f"回答{i}")
        
        messages = manager.get_context_messages()
        self.assertLessEqual(len(messages), 4)

    def test_get_last_n_turns(self):
        """测试获取最近N轮对话"""
        self.manager.add_user_message("问题1")
        self.manager.add_assistant_message("回答1")
        self.manager.add_user_message("问题2")
        self.manager.add_assistant_message("回答2")
        self.manager.add_user_message("问题3")
        self.manager.add_assistant_message("回答3")
        
        turns = self.manager.get_last_n_turns(2)
        self.assertEqual(len(turns), 4)  # 2轮 = 4条消息
        self.assertEqual(turns[0]["content"], "问题2")

    def test_clear_current_session(self):
        """测试清空当前会话"""
        self.manager.add_user_message("问题")
        self.manager.clear_current_session()
        
        messages = self.manager.get_context_messages()
        self.assertEqual(len(messages), 0)

    def test_save_and_load_session(self):
        """测试保存和加载会话"""
        manager = ConversationManager(
            history_dir=self.temp_dir,
            auto_save=True
        )
        
        manager.add_user_message("问题1")
        manager.add_assistant_message("回答1")
        session_id = manager.current_session.session_id
        
        # 创建新管理器并加载会话
        new_manager = ConversationManager(
            history_dir=self.temp_dir,
            auto_save=False
        )
        loaded = new_manager.load_session(session_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.messages), 2)

    def test_list_sessions(self):
        """测试列出会话"""
        manager = ConversationManager(
            history_dir=self.temp_dir,
            auto_save=True
        )
        
        manager.create_session("session1")
        manager.add_user_message("问题1")
        
        manager.create_session("session2")
        manager.add_user_message("问题2")
        
        sessions = manager.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_delete_session(self):
        """测试删除会话"""
        manager = ConversationManager(
            history_dir=self.temp_dir,
            auto_save=True
        )
        
        manager.create_session("to_delete")
        manager.add_user_message("问题")
        
        result = manager.delete_session("to_delete")
        self.assertTrue(result)
        
        sessions = manager.list_sessions()
        self.assertEqual(len(sessions), 0)

    def test_get_stats(self):
        """测试获取统计信息"""
        self.manager.add_user_message("问题1")
        self.manager.add_assistant_message("回答1")
        
        stats = self.manager.get_stats()
        self.assertEqual(stats["current_session_messages"], 2)
        self.assertIsNotNone(stats["current_session_id"])


if __name__ == '__main__':
    unittest.main()
