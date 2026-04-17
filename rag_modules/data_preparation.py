import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块：从本地目录递归加载 Markdown 菜谱并维护元数据。"""

    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）

    def load_documents(self) -> List[Document]:
        logger.info(f"正在从 {self.data_path} 加载 Markdown 文档...")
        documents: List[Document] = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent",
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个食谱文档")
        return documents

    def load_single_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档

        Args:
            file_path: 文档文件路径

        Returns:
            加载的文档，失败返回None
        """
        try:
            md_file = Path(file_path)
            if not md_file.exists():
                logger.warning(f"文件不存在: {file_path}")
                return None

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                data_root = Path(self.data_path).resolve()
                relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
            except Exception:
                relative_path = Path(md_file).as_posix()

            parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )
            self._enhance_metadata(doc)
            logger.info(f"成功加载文档: {md_file.stem}")
            return doc
        except Exception as e:
            logger.warning(f"读取文件 {file_path} 失败: {e}")
            return None

    def chunk_single_document(self, doc: Document) -> List[Document]:
        """
        对单个文档进行分块

        Args:
            doc: 单个文档

        Returns:
            分块后的文档列表
        """
        return self._split_markdown_document(doc)

    def _enhance_metadata(self, doc: Document):
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts

        # 提取分类信息
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break

        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        """将父文档按 Markdown 标题结构切分为子块。"""
        logger.info("正在进行Markdown分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        all_chunks: List[Document] = []

        for doc in self.documents:
            all_chunks.extend(self._split_markdown_document(doc))

        for i, chunk in enumerate(all_chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = all_chunks
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个chunk")
        return all_chunks

    def _split_markdown_document(self, doc: Document) -> List[Document]:
        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        parent_id = doc.metadata["parent_id"]
        try:
            content_preview = doc.page_content[:200]
            has_headers = any(line.strip().startswith("#") for line in content_preview.split("\n"))
            if not has_headers:
                logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现 Markdown 标题")

            md_chunks = markdown_splitter.split_text(doc.page_content)
            if len(md_chunks) <= 1:
                logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题充分分割")

            for i, chunk in enumerate(md_chunks):
                cid = str(uuid.uuid4())
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update(
                    {
                        "chunk_id": cid,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i,
                    }
                )
            return md_chunks
        except Exception as e:
            logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown 分割失败: {e}")
            cid = str(uuid.uuid4())
            fallback = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": cid,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": 0,
                },
            )
            return [fallback]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
