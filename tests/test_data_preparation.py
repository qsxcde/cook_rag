"""
数据准备模块单元测试
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_modules.data_preparation import DataPreparationModule
from langchain_core.documents import Document


class TestDataPreparationModule(unittest.TestCase):
    """数据准备模块测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = self.temp_dir.name
        self.module = DataPreparationModule(self.data_path)

    def tearDown(self):
        """测试后清理"""
        self.temp_dir.cleanup()

    def _create_test_markdown(self, filename: str, content: str, subdir: str = None):
        """创建测试用的Markdown文件"""
        dir_path = Path(self.data_path)
        if subdir:
            dir_path = dir_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(file_path)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.module.data_path, self.data_path)
        self.assertEqual(len(self.module.documents), 0)
        self.assertEqual(len(self.module.chunks), 0)

    def test_get_supported_categories(self):
        """测试获取支持的分类"""
        categories = DataPreparationModule.get_supported_categories()
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        # 检查是否包含常见分类
        self.assertTrue(any(cat in categories for cat in ['荤菜', '素菜', '汤品', '甜品']))

    def test_get_supported_difficulties(self):
        """测试获取支持的难度"""
        difficulties = DataPreparationModule.get_supported_difficulties()
        self.assertIsInstance(difficulties, list)
        self.assertEqual(len(difficulties), 5)
        self.assertIn('非常简单', difficulties)
        self.assertIn('简单', difficulties)
        self.assertIn('中等', difficulties)
        self.assertIn('困难', difficulties)
        self.assertIn('非常困难', difficulties)

    def test_load_single_document(self):
        """测试加载单个文档"""
        # 创建测试文件
        test_content = """# 测试菜品

这是一个测试菜谱。

## 难度
★★

## 食材
- 食材1
- 食材2
"""
        file_path = self._create_test_markdown("测试菜品.md", test_content)

        # 加载文档
        doc = self.module.load_single_document(file_path)

        # 验证结果
        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, Document)
        self.assertIn('测试菜品', doc.page_content)
        self.assertEqual(doc.metadata['dish_name'], '测试菜品')
        self.assertEqual(doc.metadata['difficulty'], '简单')

    def test_load_documents(self):
        """测试加载多个文档"""
        # 创建多个测试文件
        self._create_test_markdown("菜1.md", "# 菜1\n★", "meat_dish")
        self._create_test_markdown("菜2.md", "# 菜2\n★★", "vegetable_dish")

        # 加载文档
        docs = self.module.load_documents()

        # 验证结果
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(self.module.documents), 2)

    def test_chunk_single_document(self):
        """测试单个文档分块"""
        # 创建测试内容
        test_content = """# 主标题

内容1

## 二级标题1

内容2

### 三级标题

内容3

## 二级标题2

内容4
"""
        file_path = self._create_test_markdown("测试.md", test_content)
        doc = self.module.load_single_document(file_path)

        # 分块
        chunks = self.module.chunk_single_document(doc)

        # 验证结果
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document)
            self.assertIn('parent_id', chunk.metadata)

    def test_chunk_documents(self):
        """测试批量文档分块"""
        # 创建测试文件
        self._create_test_markdown("菜1.md", "# 菜1\n内容\n## 步骤1\n步骤内容")
        self._create_test_markdown("菜2.md", "# 菜2\n内容\n## 步骤1\n步骤内容")

        # 加载并分块
        self.module.load_documents()
        chunks = self.module.chunk_documents()

        # 验证结果
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len(self.module.chunks), len(chunks))

    def test_enhance_metadata_difficulty(self):
        """测试元数据增强 - 难度识别"""
        # 测试不同难度星级
        test_cases = [
            ('# 简单菜\n★', '非常简单'),
            ('# 中等菜\n★★★', '中等'),
            ('# 困难菜\n★★★★★', '非常困难'),
            ('# 无星级\n内容', '未知'),
        ]

        for content, expected_difficulty in test_cases:
            file_path = self._create_test_markdown(f"test_{expected_difficulty}.md", content)
            doc = self.module.load_single_document(file_path)
            self.assertEqual(doc.metadata['difficulty'], expected_difficulty)

    def test_get_statistics(self):
        """测试获取统计信息"""
        # 创建测试文件
        self._create_test_markdown("荤菜1.md", "# 荤菜1\n★", "meat_dish")
        self._create_test_markdown("素菜1.md", "# 素菜1\n★★", "vegetable_dish")

        # 加载并分块
        self.module.load_documents()
        self.module.chunk_documents()

        # 获取统计
        stats = self.module.get_statistics()

        # 验证结果
        self.assertIn('total_documents', stats)
        self.assertIn('total_chunks', stats)
        self.assertIn('categories', stats)
        self.assertIn('difficulties', stats)
        self.assertEqual(stats['total_documents'], 2)

    def test_get_parent_documents(self):
        """测试获取父文档"""
        # 创建测试文档
        self._create_test_markdown("菜1.md", "# 菜1\n## 步骤1\n内容1\n## 步骤2\n内容2")

        # 加载并分块
        self.module.load_documents()
        chunks = self.module.chunk_documents()

        # 获取父文档
        parent_docs = self.module.get_parent_documents(chunks)

        # 验证结果
        self.assertEqual(len(parent_docs), 1)
        self.assertEqual(parent_docs[0].metadata['dish_name'], '菜1')


if __name__ == '__main__':
    unittest.main()
