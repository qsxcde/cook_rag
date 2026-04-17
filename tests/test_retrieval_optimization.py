"""
检索优化模块单元测试
"""

import unittest
import tempfile
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_modules.retrieval_optimization import (
    RetrievalOptimizationModule,
    _token_has_content,
    _cjk_bm25_preprocess
)
from langchain_core.documents import Document


class TestRetrievalHelperFunctions(unittest.TestCase):
    """检索辅助函数测试"""

    def test_token_has_content(self):
        """测试token内容判断"""
        # 测试有效token
        self.assertTrue(_token_has_content("红烧肉"))
        self.assertTrue(_token_has_content("pork"))
        self.assertTrue(_token_has_content("123"))
        self.assertTrue(_token_has_content("红烧肉123"))

        # 测试无效token
        self.assertFalse(_token_has_content(""))
        self.assertFalse(_token_has_content("  "))
        self.assertFalse(_token_has_content("，"))
        self.assertFalse(_token_has_content("!!!"))

    def test_cjk_bm25_preprocess(self):
        """测试CJK BM25预处理"""
        # 测试中文文本
        text = "红烧肉怎么做？需要什么材料？"
        tokens = _cjk_bm25_preprocess(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # 测试空文本
        self.assertEqual(_cjk_bm25_preprocess(""), [])
        self.assertEqual(_cjk_bm25_preprocess("   "), [])

        # 测试英文文本
        text = "How to cook pork?"
        tokens = _cjk_bm25_preprocess(text)
        self.assertIn("how", tokens)
        self.assertIn("cook", tokens)
        self.assertIn("pork", tokens)


class TestRetrievalOptimizationModule(unittest.TestCase):
    """检索优化模块测试"""

    def setUp(self):
        """测试前准备"""
        # 创建测试文档
        self.test_chunks = [
            Document(
                page_content="红烧肉是一道经典的中式菜肴，主要食材是五花肉。",
                metadata={
                    "dish_name": "红烧肉",
                    "parent_id": "id1",
                    "category": "荤菜",
                    "difficulty": "中等"
                }
            ),
            Document(
                page_content="制作红烧肉需要先将五花肉切块，焯水去腥。",
                metadata={
                    "dish_name": "红烧肉",
                    "parent_id": "id1",
                    "category": "荤菜",
                    "difficulty": "中等"
                }
            ),
            Document(
                page_content="清炒时蔬是一道简单的素菜，适合夏天食用。",
                metadata={
                    "dish_name": "清炒时蔬",
                    "parent_id": "id2",
                    "category": "素菜",
                    "difficulty": "简单"
                }
            )
        ]

        # 创建mock的vectorstore
        self.mock_vectorstore = unittest.mock.MagicMock()
        self.mock_vectorstore.similarity_search.return_value = self.test_chunks[:2]

    def test_initialization(self):
        """测试初始化"""
        # 使用mock创建模块
        module = RetrievalOptimizationModule(self.mock_vectorstore, self.test_chunks)
        self.assertIsNotNone(module.vectorstore)
        self.assertIsNotNone(module.chunks)
        self.assertIsNotNone(module.bm25_retriever)

    def test_doc_matches_filters(self):
        """测试文档过滤"""
        module = RetrievalOptimizationModule(self.mock_vectorstore, self.test_chunks)

        # 测试单条件过滤
        doc = self.test_chunks[0]
        self.assertTrue(module._doc_matches_filters(doc, {"category": "荤菜"}))
        self.assertFalse(module._doc_matches_filters(doc, {"category": "素菜"}))

        # 测试多条件过滤
        self.assertTrue(module._doc_matches_filters(doc, {"category": "荤菜", "difficulty": "中等"}))
        self.assertFalse(module._doc_matches_filters(doc, {"category": "荤菜", "difficulty": "简单"}))

    def test_boost_chunks_by_dish_name_in_query(self):
        """测试按菜名提升"""
        module = RetrievalOptimizationModule(self.mock_vectorstore, self.test_chunks)

        # 测试包含菜名的查询
        boosted = module._boost_chunks_by_dish_name_in_query("红烧肉怎么做")
        self.assertGreater(len(boosted), 0)
        self.assertEqual(boosted[0].metadata["dish_name"], "红烧肉")

        # 测试不包含菜名的查询
        boosted = module._boost_chunks_by_dish_name_in_query("怎么做菜")
        self.assertEqual(len(boosted), 0)

    def test_rrf_rerank(self):
        """测试RRF重排"""
        module = RetrievalOptimizationModule(self.mock_vectorstore, self.test_chunks)

        # 创建测试用的文档列表
        docs1 = self.test_chunks[:2]
        docs2 = self.test_chunks[1:]

        # 执行重排
        reranked = module._rrf_rerank(docs1, docs2)

        # 验证结果
        self.assertIsInstance(reranked, list)
        self.assertGreater(len(reranked), 0)

    def test_merge_dish_boost_first(self):
        """测试合并提升结果"""
        module = RetrievalOptimizationModule(self.mock_vectorstore, self.test_chunks)

        # 创建测试数据
        reranked = self.test_chunks[1:]  # 从第二个开始
        merged = module._merge_dish_boost_first("红烧肉", reranked)

        # 验证红烧肉在第一位
        self.assertEqual(merged[0].metadata["dish_name"], "红烧肉")


if __name__ == '__main__':
    unittest.main()
