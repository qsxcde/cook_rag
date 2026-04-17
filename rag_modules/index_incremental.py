"""
索引增量更新模块
提供增量索引更新功能，无需每次重建整个索引
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


class DocumentMetadataManager:
    """文档元数据管理器，用于跟踪文档变更"""

    def __init__(self, metadata_path: str):
        """
        初始化文档元数据管理器

        Args:
            metadata_path: 元数据保存路径
        """
        self.metadata_path = Path(metadata_path)
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.doc_metadata = json.load(f)
                logger.info(f"已加载 {len(self.doc_metadata)} 个文档元数据")
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}")
                self.doc_metadata = {}

    def _save_metadata(self):
        """保存元数据"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_metadata, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """
        计算文件哈希值

        Args:
            file_path: 文件路径

        Returns:
            文件哈希值
        """
        if not file_path.exists():
            return ""
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            hash_obj.update(f.read())
        return hash_obj.hexdigest()

    def scan_documents(self, data_path: str) -> Dict[str, Path]:
        """
        扫描数据目录，获取所有文档

        Args:
            data_path: 数据目录路径

        Returns:
            文档ID到文件路径的映射
        """
        data_path_obj = Path(data_path)
        docs: Dict[str, Path] = {}

        for md_file in data_path_obj.rglob("*.md"):
            try:
                data_root = Path(data_path).resolve()
                relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
            except Exception:
                relative_path = Path(md_file).as_posix()
            doc_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
            docs[doc_id] = md_file

        return docs

    def get_changes(self, data_path: str) -> Dict[str, List[str]]:
        """
        获取文档变更

        Args:
            data_path: 数据目录路径

        Returns:
            变更字典，包含 added, modified, deleted
        """
        current_docs = self.scan_documents(data_path)
        current_ids = set(current_docs.keys())
        tracked_ids = set(self.doc_metadata.keys())

        added = list(current_ids - tracked_ids)
        deleted = list(tracked_ids - current_ids)
        modified: List[str] = []

        for doc_id in current_ids & tracked_ids:
            file_path = current_docs[doc_id]
            new_hash = self._get_file_hash(file_path)
            old_hash = self.doc_metadata[doc_id].get("hash", "")
            if new_hash != old_hash:
                modified.append(doc_id)

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted
        }

    def update_metadata(self, doc_id: str, file_path: Path):
        """
        更新文档元数据

        Args:
            doc_id: 文档ID
            file_path: 文件路径
        """
        self.doc_metadata[doc_id] = {
            "hash": self._get_file_hash(file_path),
            "path": str(file_path),
            "updated_at": datetime.now().isoformat()
        }
        self._save_metadata()

    def remove_metadata(self, doc_id: str):
        """
        移除文档元数据

        Args:
            doc_id: 文档ID
        """
        if doc_id in self.doc_metadata:
            del self.doc_metadata[doc_id]
            self._save_metadata()

    def get_doc_path(self, doc_id: str) -> Optional[Path]:
        """
        获取文档路径

        Args:
            doc_id: 文档ID

        Returns:
            文档路径
        """
        if doc_id in self.doc_metadata:
            return Path(self.doc_metadata[doc_id]["path"])
        return None

    def get_all_doc_ids(self) -> Set[str]:
        """
        获取所有文档ID

        Returns:
            文档ID集合
        """
        return set(self.doc_metadata.keys())


class IncrementalIndexManager:
    """增量索引管理器"""

    def __init__(
        self,
        data_path: str,
        index_save_path: str,
        metadata_path: str
    ):
        """
        初始化增量索引管理器

        Args:
            data_path: 数据目录路径
            index_save_path: 索引保存路径
            metadata_path: 元数据保存路径
        """
        self.data_path = data_path
        self.index_save_path = Path(index_save_path)
        self.metadata_manager = DocumentMetadataManager(metadata_path)

    def check_updates(self) -> Dict[str, Any]:
        """
        检查是否有更新

        Returns:
            更新信息
        """
        changes = self.metadata_manager.get_changes(self.data_path)
        has_updates = (
            len(changes["added"]) > 0
            or len(changes["modified"]) > 0
            or len(changes["deleted"]) > 0
        )

        return {
            "has_updates": has_updates,
            "changes": changes
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            "tracked_documents": len(self.metadata_manager.doc_metadata),
            "metadata_path": str(self.metadata_manager.metadata_path),
            "index_path": str(self.index_save_path)
        }
