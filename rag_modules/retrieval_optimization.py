import logging
import string
from typing import List, Dict, Any

import jieba
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


def _token_has_content(tok: str) -> bool:
    """至少含一个汉字或字母数字，过滤纯标点/空白。"""
    for c in tok:
        if c.isalnum() or "\u4e00" <= c <= "\u9fff":
            return True
    return False


def _cjk_bm25_preprocess(text: str) -> List[str]:
    """
    BM25 分词：LangChain 默认按空格切分，不适合中文。
    使用 jieba「搜索引擎模式」切词（lcut_for_search），兼顾词组与召回；
    纯 ASCII 词转小写，与菜谱中的英文单位等对齐。
    """
    if not text or not text.strip():
        return []
    tokens: List[str] = []
    for w in jieba.lcut_for_search(text.strip()):
        w = w.strip()
        if not w or not _token_has_content(w):
            continue
        if w.isascii() and any(ch.isalnum() for ch in w):
            core = w.strip(string.punctuation)
            if core:
                tokens.append(core.lower())
        else:
            tokens.append(w)
    return tokens if tokens else text.lower().split()


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()
    
    def setup_retrievers(self):
        """设置 BM25 检索器（向量检索直接使用 vectorstore.similarity_search）。"""
        logger.info("正在设置检索器...")

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
            preprocess_func=_cjk_bm25_preprocess,
        )

        logger.info("检索器设置完成")

    def _boost_chunks_by_dish_name_in_query(self, query: str) -> List[Document]:
        """用户问题里直接出现菜名时，强制把对应菜谱块排在前面（弥补英文向量对中文弱的问题）。"""
        if not query.strip():
            return []
        scored: List[tuple[int, Document]] = []
        for chunk in self.chunks:
            name = (chunk.metadata.get("dish_name") or "").strip()
            if len(name) < 2 or name not in query:
                continue
            scored.append((len(name), chunk))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        seen_parent: set[str] = set()
        out: List[Document] = []
        for _, chunk in scored:
            pid = chunk.metadata.get("parent_id") or ""
            if pid and pid in seen_parent:
                continue
            if pid:
                seen_parent.add(pid)
            out.append(chunk)
        return out

    def _merge_dish_boost_first(self, query: str, reranked: List[Document]) -> List[Document]:
        boosted = self._boost_chunks_by_dish_name_in_query(query)
        if not boosted:
            return reranked
        seen = {hash(d.page_content) for d in boosted}
        merged = list(boosted)
        for d in reranked:
            h = hash(d.page_content)
            if h in seen:
                continue
            seen.add(h)
            merged.append(d)
        return merged
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        retrieval_k: int | None = None,
        one_per_parent: bool = False,
        min_distinct_parents: int | None = None,
    ) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回的文档块数量上限（未开启 one_per_parent 时）；开启时作为候选池上限
            retrieval_k: 向量/BM25 各自召回条数，默认 5；需要更多候选时调大
            one_per_parent: 为 True 时每个父文档最多保留一条（按 RRF 顺序），用于推荐列表
            min_distinct_parents: one_per_parent 为 True 时至少覆盖的父文档数

        Returns:
            检索到的文档列表
        """
        rk = retrieval_k if retrieval_k is not None else 12
        vector_docs = self.vectorstore.similarity_search(query, k=rk)
        old_bm25_k = getattr(self.bm25_retriever, "k", 5)
        self.bm25_retriever.k = rk
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        finally:
            self.bm25_retriever.k = old_bm25_k

        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        reranked_docs = self._merge_dish_boost_first(query, reranked_docs)

        if not one_per_parent:
            return reranked_docs[:top_k]

        target_parents = min_distinct_parents or max(top_k, 8)
        pool = reranked_docs[: max(top_k, 80)]
        seen_parent: set[str] = set()
        out: List[Document] = []
        for doc in pool:
            pid = doc.metadata.get("parent_id")
            if not pid or pid in seen_parent:
                continue
            seen_parent.add(pid)
            out.append(doc)
            if len(out) >= target_parents:
                break
        return out
    
    def _doc_matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            else:
                if doc.metadata[key] != value:
                    return False
        return True

    def metadata_filtered_search(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        diversify_parents: bool = False,
        min_distinct_parents: int | None = None,
    ) -> List[Document]:
        """
        带元数据过滤的检索

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: diversify_parents 为 False 时，返回的块数上限
            diversify_parents: 为 True 时每个父文档最多保留一条匹配块（按 RRF 顺序），便于列表推荐多道菜
            min_distinct_parents: diversify_parents 为 True 时目标父文档数量

        Returns:
            过滤后的文档列表
        """
        retrieval_k = max(30, top_k * 6)
        rrf_pool = max(100, top_k * 25)
        docs = self.hybrid_search(query, top_k=rrf_pool, retrieval_k=retrieval_k)

        matching = [doc for doc in docs if self._doc_matches_filters(doc, filters)]

        if not diversify_parents:
            return matching[:top_k]

        target = min_distinct_parents or max(top_k, 8)
        seen_parent: set[str] = set()
        out: List[Document] = []
        for doc in matching:
            pid = doc.metadata.get("parent_id")
            if not pid or pid in seen_parent:
                continue
            seen_parent.add(pid)
            out.append(doc)
            if len(out) >= target:
                break
        return out

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, _final_score in sorted_docs:
            if doc_id in doc_objects:
                reranked_docs.append(doc_objects[doc_id])

        logger.info(
            "RRF 重排完成: 向量 %s 条, BM25 %s 条, 合并 %s 条",
            len(vector_docs),
            len(bm25_docs),
            len(reranked_docs),
        )

        return reranked_docs