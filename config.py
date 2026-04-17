"""
RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    data_path: str = "./data"
    index_save_path: str = "./vector_index"

    # 嵌入模型：短名如 bge-large-zh 会解析为本地 hf_models/bge-large-zh 或 Hub 上的 BAAI/bge-large-zh
    # 更换模型后须删除原 vector_index 并重新构建索引（向量空间不兼容）
    embedding_model: str = "bge-large-zh"
    llm_model: str = "qwen3.5-35b-a3b"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "./query_cache"
    cache_ttl: int = 3600

    # 增量更新配置
    enable_incremental_update: bool = True
    index_metadata_path: str = "./vector_index/index_metadata.json"

    # 多轮对话配置
    enable_conversation: bool = True
    conversation_history_dir: str = "./conversation_history"
    max_history_length: int = 20
    context_window_turns: int = 3

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'enable_cache': self.enable_cache,
            'cache_dir': self.cache_dir,
            'cache_ttl': self.cache_ttl,
            'enable_incremental_update': self.enable_incremental_update,
            'index_metadata_path': self.index_metadata_path,
            'enable_conversation': self.enable_conversation,
            'conversation_history_dir': self.conversation_history_dir,
            'max_history_length': self.max_history_length,
            'context_window_turns': self.context_window_turns
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
