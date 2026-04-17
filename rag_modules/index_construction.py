"""
索引构建模块
负责构建索引，包括向量化、索引构建、索引存储等
"""
import logging
from typing import List, Optional
from pathlib import Path
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

# 项目根目录（rag_modules 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_embedding_model_id(model_name: str) -> str:
    """
    解析 HuggingFaceEmbeddings 的 model_name：
    - 已含 '/' 时视为完整 Hub repo id（如 BAAI/bge-large-zh）
    - 否则若存在 ./hf_models/<model_name> 目录则使用本地路径
    - 否则使用 BAAI/<model_name> 从 Hub 加载（首次会自动下载）
    """
    if "/" in model_name:
        return model_name
    local_dir = _PROJECT_ROOT / "hf_models" / model_name
    if local_dir.is_dir():
        return str(local_dir)
    return f"BAAI/{model_name}"


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "bge-large-zh", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型"""
        resolved = _resolve_embedding_model_id(self.model_name)
        logger.info(f"正在初始化嵌入模型: {self.model_name} -> {resolved}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=resolved,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("嵌入模型初始化完成")
    
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            FAISS向量存储对象
        """
        logger.info("正在构建FAISS向量索引...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore

    def save_index(self):
        """
        保存向量索引到配置的路径
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
    
    def load_index(self):
        """
        从配置的路径加载向量索引

        Returns:
            加载的向量存储对象，如果加载失败返回None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def add_documents(self, chunks: List[Document]):
        """
        向现有索引添加文档

        Args:
            chunks: 要添加的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先加载或构建向量索引")
        
        if not chunks:
            logger.info("没有文档需要添加")
            return
        
        logger.info(f"正在向索引添加 {len(chunks)} 个文档块...")
        self.vectorstore.add_documents(chunks)
        logger.info(f"文档添加完成")

    def remove_documents_by_parent_id(self, parent_ids: List[str]):
        """
        根据父文档ID删除文档

        Args:
            parent_ids: 父文档ID列表
        """
        if not self.vectorstore:
            raise ValueError("请先加载或构建向量索引")
        
        if not parent_ids:
            logger.info("没有文档需要删除")
            return
        
        logger.info(f"正在从索引删除与 {len(parent_ids)} 个父文档相关的块...")
        
        # FAISS 不直接支持按元数据删除，我们需要重建索引
        # 获取所有文档
        all_docs = []
        if hasattr(self.vectorstore, "docstore"):
            for doc_id, doc in self.vectorstore.docstore._dict.items():
                all_docs.append(doc)
        
        # 过滤掉要删除的文档
        filtered_docs = [
            doc for doc in all_docs
            if doc.metadata.get("parent_id") not in parent_ids
        ]
        
        removed_count = len(all_docs) - len(filtered_docs)
        logger.info(f"过滤掉 {removed_count} 个文档块")
        
        # 重建索引
        if filtered_docs:
            self.vectorstore = FAISS.from_documents(
                documents=filtered_docs,
                embedding=self.embeddings
            )
            logger.info("索引重建完成")
        else:
            logger.warning("过滤后没有文档剩余")

    def get_document_count(self) -> int:
        """
        获取索引中的文档数量

        Returns:
            文档数量
        """
        if not self.vectorstore:
            return 0
        
        if hasattr(self.vectorstore, "index"):
            return self.vectorstore.index.ntotal
        return 0
