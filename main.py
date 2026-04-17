"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
    QueryCache,
    IncrementalIndexManager,
    ConversationManager
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱 RAG 系统主类（仅本地 Markdown 知识库）。"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.query_cache = None
        self.incremental_manager = None
        self.conversation_manager = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 4. 初始化查询缓存
        if self.config.enable_cache:
            print("🗄️ 初始化查询缓存...")
            self.query_cache = QueryCache(
                cache_dir=self.config.cache_dir,
                ttl=self.config.cache_ttl
            )
            cache_stats = self.query_cache.get_stats()
            print(f"   缓存状态: {cache_stats['total_cache_files']} 个文件, {cache_stats['total_size_kb']:.2f} KB")

        # 5. 初始化增量更新管理器
        if self.config.enable_incremental_update:
            print("📦 初始化增量更新管理器...")
            self.incremental_manager = IncrementalIndexManager(
                data_path=self.config.data_path,
                index_save_path=self.config.index_save_path,
                metadata_path=self.config.index_metadata_path
            )
            inc_stats = self.incremental_manager.get_stats()
            print(f"   已跟踪 {inc_stats['tracked_documents']} 个文档")

        # 6. 初始化对话管理器
        if self.config.enable_conversation:
            print("💬 初始化对话管理器...")
            self.conversation_manager = ConversationManager(
                history_dir=self.config.conversation_history_dir,
                max_history_length=self.config.max_history_length
            )
            conv_stats = self.conversation_manager.get_stats()
            print(f"   已保存 {conv_stats['total_sessions']} 个会话")

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """加载本地 Markdown、分块；若已有 FAISS 索引则加载，否则构建并保存。支持增量更新。"""
        print("\n正在构建知识库...")

        index_dir = Path(self.config.index_save_path)
        vectorstore = None

        # 检查是否需要增量更新
        if self.config.enable_incremental_update and self.incremental_manager:
            update_info = self.incremental_manager.check_updates()
            if update_info["has_updates"] and index_dir.exists():
                print("🔄 检测到文档变更，执行增量更新...")
                self._apply_incremental_update(update_info["changes"])
                vectorstore = self.index_module.vectorstore

        # 正常加载流程
        if vectorstore is None:
            print("加载菜谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            if index_dir.exists():
                vectorstore = self.index_module.load_index()
                if vectorstore is not None:
                    print("✅ 已加载保存的向量索引。")
                    # 更新元数据（首次加载）
                    if self.config.enable_incremental_update and self.incremental_manager:
                        current_docs = self.incremental_manager.metadata_manager.scan_documents(self.config.data_path)
                        for doc_id, file_path in current_docs.items():
                            self.incremental_manager.metadata_manager.update_metadata(doc_id, file_path)
            else:
                print("正在构建向量索引...")
                vectorstore = self.index_module.build_vector_index(chunks)
                self.index_module.save_index()
                # 初始化元数据
                if self.config.enable_incremental_update and self.incremental_manager:
                    current_docs = self.incremental_manager.metadata_manager.scan_documents(self.config.data_path)
                    for doc_id, file_path in current_docs.items():
                        self.incremental_manager.metadata_manager.update_metadata(doc_id, file_path)

            # 确保所有文档都在 data_module 中
            if not self.data_module.documents:
                self.data_module.load_documents()
                self.data_module.chunk_documents()

            print("初始化检索优化...")
            self.retrieval_module = RetrievalOptimizationModule(vectorstore, self.data_module.chunks)
        else:
            # 增量更新后需要重新加载所有文档到 data_module
            print("重新加载所有文档...")
            self.data_module.load_documents()
            self.data_module.chunk_documents()
            print("重新初始化检索优化...")
            self.retrieval_module = RetrievalOptimizationModule(vectorstore, self.data_module.chunks)

        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def _apply_incremental_update(self, changes: dict):
        """
        应用增量更新

        Args:
            changes: 变更字典
        """
        # 加载现有索引
        vectorstore = self.index_module.load_index()
        if vectorstore is None:
            print("⚠️ 无法加载现有索引，将重建完整索引")
            return

        # 处理删除的文档
        deleted_ids = changes.get("deleted", [])
        if deleted_ids:
            print(f"删除 {len(deleted_ids)} 个文档...")
            self.index_module.remove_documents_by_parent_id(deleted_ids)
            for doc_id in deleted_ids:
                self.incremental_manager.metadata_manager.remove_metadata(doc_id)
            # 清空缓存，因为数据已变化
            if self.query_cache:
                self.query_cache.clear()

        # 处理新增和修改的文档
        added_ids = changes.get("added", [])
        modified_ids = changes.get("modified", [])
        all_ids_to_process = added_ids + modified_ids

        if all_ids_to_process:
            print(f"处理 {len(all_ids_to_process)} 个新增/修改的文档...")

            # 对于修改的文档，先删除旧版本
            if modified_ids:
                self.index_module.remove_documents_by_parent_id(modified_ids)

            # 添加新文档
            new_chunks = []
            for doc_id in all_ids_to_process:
                file_path = self.incremental_manager.metadata_manager.get_doc_path(doc_id)
                if file_path is None:
                    # 尝试重新扫描获取路径
                    current_docs = self.incremental_manager.metadata_manager.scan_documents(self.config.data_path)
                    if doc_id in current_docs:
                        file_path = current_docs[doc_id]

                if file_path and file_path.exists():
                    doc = self.data_module.load_single_document(str(file_path))
                    if doc:
                        chunks = self.data_module.chunk_single_document(doc)
                        new_chunks.extend(chunks)
                        self.incremental_manager.metadata_manager.update_metadata(doc_id, file_path)

            if new_chunks:
                self.index_module.add_documents(new_chunks)
                print(f"添加了 {len(new_chunks)} 个新文档块")
                # 清空缓存，因为数据已变化
                if self.query_cache:
                    self.query_cache.clear()

            self.index_module.save_index()

    def check_for_updates(self) -> dict:
        """
        检查是否有文档更新

        Returns:
            更新信息
        """
        if not self.incremental_manager:
            return {"has_updates": False, "changes": {}}
        return self.incremental_manager.check_updates()

    def apply_updates(self):
        """
        手动应用更新
        """
        update_info = self.check_for_updates()
        if update_info["has_updates"]:
            print("🔄 发现更新，正在应用...")
            self._apply_incremental_update(update_info["changes"])
            # 重新初始化检索模块
            if self.index_module.vectorstore:
                self.data_module.load_documents()
                self.data_module.chunk_documents()
                self.retrieval_module = RetrievalOptimizationModule(
                    self.index_module.vectorstore,
                    self.data_module.chunks
                )
            print("✅ 更新应用完成！")
        else:
            print("✅ 没有发现更新")
    
    def ask_question(self, question: str, stream: bool = False, use_conversation: bool = True):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出
            use_conversation: 是否使用多轮对话

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 获取对话历史
        conversation_history = None
        if use_conversation and self.config.enable_conversation and self.conversation_manager:
            conversation_history = self.conversation_manager.get_last_n_turns(
                self.config.context_window_turns
            )
            if conversation_history:
                print(f"📝 已加载 {len(conversation_history)} 条对话历史")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 检查缓存（仅非流式输出且无对话历史时）
        if not stream and self.config.enable_cache and self.query_cache and not conversation_history:
            cached_data = self.query_cache.get(question, route_type)
            if cached_data:
                print("📦 使用缓存回答")
                return cached_data.get("answer", "")

        # 3. 智能查询重写
        if conversation_history:
            print("🔄 根据对话历史重写查询...")
            rewritten_query = self.generation_module.rewrite_query_with_history(question, conversation_history)
        elif route_type == "list":
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 4. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            list_diversify = route_type == "list"
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=self.config.top_k,
                diversify_parents=list_diversify,
                min_distinct_parents=max(self.config.top_k, 8) if list_diversify else None,
            )
        elif route_type == "list":
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=120,
                retrieval_k=35,
                one_per_parent=True,
                min_distinct_parents=max(self.config.top_k, 8),
            )
        elif route_type == "compare_difficulty":
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=18,
                retrieval_k=45,
            )
        elif route_type == "ingredient":
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=10,
                retrieval_k=28,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 5. 检查是否找到相关内容
        if not relevant_chunks:
            return "食谱 RAG：抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 6. 根据路由类型选择回答方式
        if route_type == "list":
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            doc_names = [d.metadata.get("dish_name", "未知菜品") for d in relevant_docs]
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            answer = self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            doc_names = [d.metadata.get("dish_name", "未知菜品") for d in relevant_docs]
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成回答...")
            
            # 使用带对话历史的生成方法
            if conversation_history:
                if stream:
                    return self._wrap_stream_with_save(
                        self.generation_module.generate_answer_with_history_stream(
                            question, relevant_docs, conversation_history
                        ),
                        question
                    )
                answer = self.generation_module.generate_answer_with_history(
                    question, relevant_docs, conversation_history, route_type
                )
            else:
                if route_type == "ingredient":
                    if stream:
                        return self._wrap_stream_with_save(
                            self.generation_module.generate_ingredient_answer_stream(question, relevant_docs),
                            question
                        )
                    answer = self.generation_module.generate_ingredient_answer(question, relevant_docs)
                elif route_type == "compare_difficulty":
                    if stream:
                        return self._wrap_stream_with_save(
                            self.generation_module.generate_difficulty_compare_answer_stream(question, relevant_docs),
                            question
                        )
                    answer = self.generation_module.generate_difficulty_compare_answer(question, relevant_docs)
                elif route_type == "detail":
                    if stream:
                        return self._wrap_stream_with_save(
                            self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs),
                            question
                        )
                    answer = self.generation_module.generate_step_by_step_answer(question, relevant_docs)
                else:
                    if stream:
                        return self._wrap_stream_with_save(
                            self.generation_module.generate_basic_answer_stream(question, relevant_docs),
                            question
                        )
                    answer = self.generation_module.generate_basic_answer(question, relevant_docs)

        # 7. 保存对话历史
        if use_conversation and self.config.enable_conversation and self.conversation_manager:
            self.conversation_manager.add_user_message(question)
            self.conversation_manager.add_assistant_message(answer)
            print("💾 对话已保存")

        # 8. 保存缓存（仅非流式输出且无对话历史时）
        if not stream and self.config.enable_cache and self.query_cache and not conversation_history:
            self.query_cache.set(question, {"answer": answer}, route_type)
            print("💾 回答已缓存")

        return answer

    def _wrap_stream_with_save(self, stream_gen, question):
        """包装流式输出，在完成后保存对话历史"""
        collected_answer = []
        
        def wrapped_stream():
            for chunk in stream_gen:
                if chunk:
                    collected_answer.append(chunk)
                    yield chunk
            # 流式输出完成后保存对话历史
            if self.config.enable_conversation and self.conversation_manager:
                full_answer = "".join(collected_answer)
                self.conversation_manager.add_user_message(question)
                self.conversation_manager.add_assistant_message(full_answer)
                print("💾 对话已保存（流式）")
        
        return wrapped_stream()
    
    def clear_cache(self):
        """
        清空查询缓存"""
        if self.query_cache:
            self.query_cache.clear()
            print("🗑️ 已清空所有缓存")
        else:
            print("❌ 缓存未启用")

    def clear_conversation(self):
        """清空当前对话历史"""
        if self.conversation_manager:
            self.conversation_manager.clear_current_session()
            print("🗑️ 已清空当前对话历史")
        else:
            print("❌ 对话管理未启用")

    def new_conversation(self):
        """开始新对话"""
        if self.conversation_manager:
            self.conversation_manager.create_session()
            print("💬 已开始新对话")
        else:
            print("❌ 对话管理未启用")

    def show_conversation_stats(self):
        """显示对话统计信息"""
        if self.conversation_manager:
            stats = self.conversation_manager.get_stats()
            print("\n📊 对话统计:")
            print(f"   总会话数: {stats['total_sessions']}")
            print(f"   总消息数: {stats['total_messages']}")
            print(f"   当前会话ID: {stats['current_session_id'] or '无'}")
            print(f"   当前会话消息数: {stats['current_session_messages']}")
            print(f"   历史目录: {stats['history_dir']}")
        else:
            print("❌ 对话管理未启用")

    def show_cache_stats(self):
        """显示缓存统计信息"""
        if self.query_cache:
            stats = self.query_cache.get_stats()
            print("\n📊 缓存统计:")
            print(f"   缓存文件数: {stats['total_cache_files']}")
            print(f"   缓存大小: {stats['total_size_kb']:.2f} KB")
            print(f"   缓存目录: {stats['cache_dir']}")
        else:
            print("❌ 缓存未启用")

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  食谱 RAG · 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别「今天吃什么」的难题！")
        print("📋 特殊命令:")
        print("   /stats - 缓存统计")
        print("   /clear - 清空缓存")
        print("   /update - 检查更新")
        print("   /apply - 应用更新")
        print("   /conv - 对话统计")
        print("   /new - 开始新对话")
        print("   /reset - 清空当前对话")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                if user_input.lower() in ['/stats', 'stats']:
                    self.show_cache_stats()
                    continue
                if user_input.lower() in ['/clear', 'clear']:
                    self.clear_cache()
                    continue
                if user_input.lower() in ['/conv', 'conv']:
                    self.show_conversation_stats()
                    continue
                if user_input.lower() in ['/new', 'new']:
                    self.new_conversation()
                    continue
                if user_input.lower() in ['/reset', 'reset']:
                    self.clear_conversation()
                    continue
                if user_input.lower() in ['/update', 'update']:
                    update_info = self.check_for_updates()
                    if update_info["has_updates"]:
                        changes = update_info["changes"]
                        print("\n🔍 发现更新:")
                        if changes["added"]:
                            print(f"   新增: {len(changes['added'])} 个文档")
                        if changes["modified"]:
                            print(f"   修改: {len(changes['modified'])} 个文档")
                        if changes["deleted"]:
                            print(f"   删除: {len(changes['deleted'])} 个文档")
                        print("\n使用 /apply 命令应用更新")
                    else:
                        print("✅ 没有发现更新")
                    continue
                if user_input.lower() in ['/apply', 'apply']:
                    self.apply_updates()
                    continue
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用食谱 RAG！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()