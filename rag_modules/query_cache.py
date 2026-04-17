"""
查询缓存模块
提供查询缓存功能，避免重复检索和生成
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


class QueryCache:
    """查询缓存类"""

    def __init__(self, cache_dir: str = "./query_cache", ttl: int = 3600):
        """
        初始化查询缓存

        Args:
            cache_dir: 缓存保存目录
            ttl: 缓存过期时间（秒），默认1小时
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str, route_type: Optional[str] = None) -> str:
        """
        生成缓存键

        Args:
            query: 查询文本
            route_type: 路由类型

        Returns:
            缓存键
        """
        key_data = {"query": query, "route_type": route_type}
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            cache_key: 缓存键

        Returns:
            缓存文件路径
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(self, query: str, route_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取缓存

        Args:
            query: 查询文本
            route_type: 路由类型

        Returns:
            缓存数据，如果不存在或已过期则返回None
        """
        cache_key = self._get_cache_key(query, route_type)
        cache_file = self._get_cache_file_path(cache_key)

        if not cache_file.exists():
            logger.debug(f"缓存未命中: {query}")
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # 检查是否过期
            if "timestamp" in cache_data:
                age = time.time() - cache_data["timestamp"]
                if age > self.ttl:
                    logger.debug(f"缓存已过期: {query}")
                    cache_file.unlink(missing_ok=True)
                    return None

            logger.info(f"缓存命中: {query}")
            return cache_data.get("data")
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            cache_file.unlink(missing_ok=True)
            return None

    def set(self, query: str, data: Dict[str, Any], route_type: Optional[str] = None):
        """
        设置缓存

        Args:
            query: 查询文本
            data: 要缓存的数据
            route_type: 路由类型
        """
        cache_key = self._get_cache_key(query, route_type)
        cache_file = self._get_cache_file_path(cache_key)

        cache_data = {
            "timestamp": time.time(),
            "data": data
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"缓存已保存: {query}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def clear(self):
        """清空所有缓存"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)
            count += 1
        logger.info(f"已清空 {count} 个缓存文件")

    def clear_expired(self):
        """清空过期缓存"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if "timestamp" in cache_data:
                    age = time.time() - cache_data["timestamp"]
                    if age > self.ttl:
                        cache_file.unlink(missing_ok=True)
                        count += 1
            except Exception:
                cache_file.unlink(missing_ok=True)
                count += 1
        logger.info(f"已清空 {count} 个过期缓存文件")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "total_cache_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_kb": total_size / 1024,
            "cache_dir": str(self.cache_dir)
        }
