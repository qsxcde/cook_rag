"""
索引构建模块单元测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_modules.index_construction import IndexConstructionModule, _resolve_embedding_model_id
from langchain_core.documents import Document


class TestHelperFunctions(unittest.TestCase):
    """辅助函数测试"""

    def test_resolve_embedding_model_id(self):
        """测试嵌入模型ID解析"""
        # 测试包含斜杠的模型名
        self.assertEqual(_resolve_embedding_model_id("BAAI/bge-large-zh"), "BAAI/bge-large-zh")

        # 测试短模型名（这里我们只测试函数逻辑，不测试本地文件）
        # 实际行为取决于文件系统，这里就不详细测试了


class TestIndexConstructionModule(unittest.TestCase):
    """索引构建模块测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temp_dir.name) / "vector_index"

        # 创建测试文档
        self.test_chunks = [
            Document(
                page_content="红烧肉是一道经典的中式菜肴。",
                metadata={"dish_name": "红烧肉"}
            ),
            Document(
                page_content="清炒时蔬是一道简单的素菜。",
                metadata={"dish_name": "清炒时蔬"}
            )
        ]

    def tearDown(self):
        """测试后清理"""
        self.temp_dir.cleanup()

    @patch('rag_modules.index_construction.HuggingFaceEmbeddings')
    def test_initialization(self, mock_embeddings):
        """测试初始化"""
        # 设置mock
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        # 创建模块
        module = IndexConstructionModule(
            model_name="test-model",
            index_save_path=str(self.index_path)
        )

        # 验证
        self.assertEqual(module.model_name, "test-model")
        self.assertEqual(module.index_save_path, str(self.index_path))
        self.assertIsNotNone(module.embeddings)
        mock_embeddings.assert_called_once()

    @patch('rag_modules.index_construction.HuggingFaceEmbeddings')
    @patch('rag_modules.index_construction.FAISS')
    def test_build_vector_index(self, mock_faiss, mock_embeddings):
        """测试构建向量索引"""
        # 设置mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # 创建模块并构建索引
        module = IndexConstructionModule(index_save_path=str(self.index_path))
        result = module.build_vector_index(self.test_chunks)

        # 验证
        self.assertEqual(result, mock_faiss_instance)
        mock_faiss.from_documents.assert_called_once()
        self.assertEqual(module.vectorstore, mock_faiss_instance)

    @patch('rag_modules.index_construction.HuggingFaceEmbeddings')
    @patch('rag_modules.index_construction.FAISS')
    def test_save_index(self, mock_faiss, mock_embeddings):
        """测试保存索引"""
        # 设置mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # 创建模块、构建索引、保存
        module = IndexConstructionModule(index_save_path=str(self.index_path))
        module.build_vector_index(self.test_chunks)
        module.save_index()

        # 验证
        mock_faiss_instance.save_local.assert_called_once_with(str(self.index_path))

    @patch('rag_modules.index_construction.HuggingFaceEmbeddings')
    @patch('rag_modules.index_construction.FAISS')
    def test_load_index(self, mock_faiss, mock_embeddings):
        """测试加载索引"""
        # 设置mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_faiss_instance = MagicMock()
        mock_faiss.load_local.return_value = mock_faiss_instance

        # 创建模块
        module = IndexConstructionModule(index_save_path=str(self.index_path))

        # 创建索引目录（模拟已存在的索引）
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 尝试加载
        result = module.load_index()

        # 验证
        self.assertEqual(result, mock_faiss_instance)
        mock_faiss.load_local.assert_called_once()

    @patch('rag_modules.index_construction.HuggingFaceEmbeddings')
    def test_get_document_count(self, mock_embeddings):
        """测试获取文档数量"""
        # 设置mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        # 创建模块
        module = IndexConstructionModule()

        # 测试无索引时
        self.assertEqual(module.get_document_count(), 0)

        # 测试有索引时
        mock_index = MagicMock()
        mock_index.index.ntotal = 100
        module.vectorstore = mock_index
        self.assertEqual(module.get_document_count(), 100)


class TestQueryCache(unittest.TestCase):
    """查询缓存模块测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "query_cache"

    def tearDown(self):
        """测试后清理"""
        self.temp_dir.cleanup()

    def test_cache_basic_operations(self):
        """测试缓存基本操作"""
        # 延迟导入避免在前面的测试中导入
        from rag_modules.query_cache import QueryCache

        # 创建缓存
        cache = QueryCache(cache_dir=str(self.cache_dir), ttl=3600)

        # 测试设置和获取
        test_query = "红烧肉怎么做"
        test_data = {"answer": "红烧肉很好吃"}
        cache.set(test_query, test_data, "detail")

        # 测试获取
        result = cache.get(test_query, "detail")
        self.assertEqual(result, test_data)

        # 测试获取不存在的
        result = cache.get("不存在的查询", "general")
        self.assertIsNone(result)

    def test_cache_clear(self):
        """测试清空缓存"""
        from rag_modules.query_cache import QueryCache

        cache = QueryCache(cache_dir=str(self.cache_dir))

        # 添加一些缓存
        cache.set("查询1", {"answer": "回答1"})
        cache.set("查询2", {"answer": "回答2"})

        # 清空
        cache.clear()

        # 验证已清空
        self.assertEqual(cache.get_stats()["total_cache_files"], 0)

    def test_cache_stats(self):
        """测试缓存统计"""
        from rag_modules.query_cache import QueryCache

        cache = QueryCache(cache_dir=str(self.cache_dir))

        # 添加一些缓存
        cache.set("查询1", {"answer": "回答1"})

        # 获取统计
        stats = cache.get_stats()
        self.assertIn("total_cache_files", stats)
        self.assertIn("total_size_bytes", stats)
        self.assertIn("cache_dir", stats)


if __name__ == '__main__':
    unittest.main()
