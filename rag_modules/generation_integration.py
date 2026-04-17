import os
import logging
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "qwen3.5-35b-a3b", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
        
        context = self._build_context(context_docs, max_length=4000)

        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱信息:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要声称「根据提供的食谱」却没有实际引用
4. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请提供简洁、准确的回答：""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
        
        context = self._build_context(context_docs, max_length=5600)

        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪导师，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱信息:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请根据食谱信息，灵活组织回答：

## 🥘 菜品介绍
[简要介绍菜品特点和难度 - 仅当食谱中有相关描述时包含]

## 🛒 所需食材
[列出主要食材和用量 - 严格引用食谱中的内容，不要编造]

## 👨‍🍳 制作步骤
[详细的分步骤说明 - 严格基于食谱内容]

## 💡 制作技巧
[仅当食谱中有明确的技巧或附加内容时包含]

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response
    
    ROUTE_TYPES = frozenset({
        "list",
        "detail",
        "general",
        "ingredient",
        "compare_difficulty",
    })

    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型: list | detail | general | ingredient | compare_difficulty
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下五种类型之一（只输出类型英文单词，不要解释）：

1. list - 用户想要获取菜品列表或推荐，只要菜名或枚举
   例：推荐几个素菜、有什么川菜、给我3个简单的菜

2. ingredient - 用户主要关心「需要哪些食材/原料/材料」，不要求完整制作步骤
   例：宫保鸡丁需要什么材料、红烧肉要准备哪些食材、白灼虾用料有哪些
   （若同时强烈要求完整做法与步骤，则选 detail）

3. compare_difficulty - 用户明确在比较**两道或以上菜**的难易、上手难度、复杂度
   例：西红柿炒鸡蛋和麻婆豆腐哪个难做、清蒸鲈鱼比红烧肉简单吗

4. detail - 用户想要具体制作方法、步骤、做法、火候、时间等操作向信息
   例：宫保鸡丁怎么做、糖醋排骨的步骤、白灼虾煮多久

5. general - 不属于以上类型的泛问：菜系介绍、营养、技巧泛谈等
   例：什么是川菜、怎么切菜更省力

请只返回以下之一：list、ingredient、compare_difficulty、detail、general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        raw = chain.invoke(query).strip().lower()
        # 模型可能输出多余标点或前后缀，从文本中抽取第一个合法类型
        cleaned = raw.replace("、", " ").replace("，", " ")
        for token in cleaned.split():
            t = token.strip(" '\"“”.,，。:：;；")
            if t in self.ROUTE_TYPES:
                return t
        if raw in self.ROUTE_TYPES:
            return raw
        return "general"
    
    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"

        # 提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"食谱 RAG 为您推荐：{dish_names[0]}"
        show = dish_names[:12]
        lines = "\n".join([f"{i+1}. {name}" for i, name in enumerate(show)])
        if len(dish_names) <= 12:
            return f"食谱 RAG 为您推荐以下菜品：\n{lines}"
        return f"食谱 RAG 为您推荐以下菜品：\n{lines}\n\n（共 {len(dish_names)} 道，此处列出前 12 道）"
    
    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        if not context_docs or len(context_docs) == 0:
            yield "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
            return
        
        context = self._build_context(context_docs, max_length=4000)

        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱信息:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要声称「根据提供的食谱」却没有实际引用
4. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请提供简洁、准确的回答：""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk
    
    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        if not context_docs or len(context_docs) == 0:
            yield "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
            return
        
        context = self._build_context(context_docs, max_length=5600)

        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪导师，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱信息:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请根据食谱信息，灵活组织回答：

## 🥘 菜品介绍
[简要介绍菜品特点和难度 - 仅当食谱中有相关描述时包含]

## 🛒 所需食材
[列出主要食材和用量 - 严格引用食谱中的内容，不要编造]

## 👨‍🍳 制作步骤
[详细的分步骤说明 - 严格基于食谱内容]

## 💡 制作技巧
[仅当食谱中有明确的技巧或附加内容时包含]

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_ingredient_answer(self, query: str, context_docs: List[Document]) -> str:
        """仅回答食材/备料类问题。"""
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
        
        context = self._build_context(context_docs, max_length=4800)
        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用

请按「菜品」分条回答，每条尽量包含：
- 主要食材与建议用量（严格来自原文，不要编造）
- 常用调料与工具若原文有则简要列出

不要展开完整烹饪步骤。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_ingredient_answer_stream(self, query: str, context_docs: List[Document]):
        if not context_docs or len(context_docs) == 0:
            yield "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
            return
        
        context = self._build_context(context_docs, max_length=4800)
        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据，除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用

请按「菜品」分条回答；不要展开完整烹饪步骤。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    def generate_difficulty_compare_answer(self, query: str, context_docs: List[Document]) -> str:
        """多菜难度对比。"""
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
        
        context = self._build_context(context_docs, max_length=7200)
        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据，除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 若检索结果中只有一道菜的资料，请说明需要用户补充其它菜名，不要臆造未提供的菜谱

请结合元数据「难度」与正文中的星级/操作复杂度说明：
1. 逐菜简要概括难点或简单点
2. 给出清晰的难度排序或对比结论（谁更难/更简单）

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_difficulty_compare_answer_stream(self, query: str, context_docs: List[Document]):
        if not context_docs or len(context_docs) == 0:
            yield "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
            return
        
        context = self._build_context(context_docs, max_length=7200)
        prompt = ChatPromptTemplate.from_template("""
你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

用户问题: {question}

相关食谱:
{context}

严格遵守以下规则：
1. 只回答在「相关食谱」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据，除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 若检索结果中只有一道菜的资料，请说明需要用户补充其它菜名，不要臆造未提供的菜谱

结合元数据「难度」与正文星级/操作复杂度进行对比与排序。

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 3200) -> str:
        """
        构建送入 LLM 的上下文字符串。

        - 多道父文档时按「均分预算」截断正文，避免第一道菜谱占满窗口导致对比/多菜场景丢菜。
        - 单文档时在 max_length 内尽量多保留正文，超出则截断并标注。
        """
        if not docs:
            return "暂无相关食谱信息。"

        def _meta_line(idx: int, doc: Document) -> str:
            parts = [f"【食谱 {idx}】"]
            if doc.metadata.get("dish_name"):
                parts.append(str(doc.metadata["dish_name"]))
            if doc.metadata.get("category"):
                parts.append(f"| 分类: {doc.metadata['category']}")
            if doc.metadata.get("difficulty"):
                parts.append(f"| 难度: {doc.metadata['difficulty']}")
            return " ".join(parts)

        sep = "\n" + "=" * 50 + "\n"
        n = len(docs)

        if n == 1:
            doc = docs[0]
            meta = _meta_line(1, doc)
            body = doc.page_content or ""
            header_len = len(meta) + len(sep) + 2
            cap = max(max_length - header_len, 800)
            if len(body) > cap:
                body = body[:cap].rstrip() + "\n…（食谱正文过长已截断）"
            return sep + meta + "\n" + body

        # 多文档：为每道菜分配相近长度，保证都能进上下文
        overhead = sum(len(_meta_line(i + 1, d)) + 8 for i, d in enumerate(docs))
        overhead += len(sep) * (n + 1)
        usable = max(max_length - overhead, 400 * n)
        per_doc = max(500, usable // n)

        blocks: List[str] = []
        for i, doc in enumerate(docs, 1):
            meta = _meta_line(i, doc)
            body = doc.page_content or ""
            if len(body) > per_doc:
                body = body[:per_doc].rstrip() + "\n…（该道食谱正文已截断，优先保留前段）"
            blocks.append(f"{meta}\n{body}")

        return sep + sep.join(blocks)

    def generate_answer_with_history(
        self,
        query: str,
        context_docs: List[Document],
        conversation_history: List[Dict[str, str]] = None,
        route_type: str = "detail"
    ) -> str:
        """
        带对话历史的回答生成

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            conversation_history: 对话历史
            route_type: 路由类型

        Returns:
            生成的回答
        """
        if not context_docs or len(context_docs) == 0:
            return "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
        
        context = self._build_context(context_docs, max_length=4000)
        history_str = self._format_conversation_history(conversation_history)

        system_prompt = """你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

注意：这是一个多轮对话，请参考之前的对话上下文来理解用户的问题。

相关食谱信息:
{context}

{history_section}

当前用户问题: {question}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请提供详细、实用的回答。如果用户的问题涉及之前讨论的内容，请结合上下文回答。"""

        history_section = ""
        if history_str:
            history_section = f"之前的对话历史:\n{history_str}\n"

        prompt = ChatPromptTemplate.from_template(system_prompt)

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "history_section": lambda _: history_section
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_answer_with_history_stream(
        self,
        query: str,
        context_docs: List[Document],
        conversation_history: List[Dict[str, str]] = None
    ):
        """
        带对话历史的流式回答生成

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            conversation_history: 对话历史

        Yields:
            生成的回答片段
        """
        if not context_docs or len(context_docs) == 0:
            yield "抱歉，我在当前知识库中没有找到与您问题相关的食谱信息。请尝试其他问题。"
            return
        
        context = self._build_context(context_docs, max_length=4000)
        history_str = self._format_conversation_history(conversation_history)

        system_prompt = """你是一位严谨的烹饪助手，必须严格基于提供的食谱信息回答用户问题。

注意：这是一个多轮对话，请参考之前的对话上下文来理解用户的问题。

相关食谱信息:
{context}

{history_section}

当前用户问题: {question}

严格遵守以下规则：
1. 只回答在「相关食谱信息」中有明确记载的内容
2. 如果信息不足或不确定，必须明确说「抱歉，我在当前食谱中没有找到相关信息」，不要编造
3. 不要添加食谱中没有的具体数据（如克数、时间、品牌等），除非食谱中明确给出
4. 不要声称「根据提供的食谱」却没有实际引用
5. 不要说「根据食谱 1」或「原文」，除非确实有多个明确来源

请提供详细、实用的回答。如果用户的问题涉及之前讨论的内容，请结合上下文回答。"""

        history_section = ""
        if history_str:
            history_section = f"之前的对话历史:\n{history_str}\n"

        prompt = ChatPromptTemplate.from_template(system_prompt)

        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "history_section": lambda _: history_section
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """
        格式化对话历史

        Args:
            history: 对话历史列表

        Returns:
            格式化后的对话历史字符串
        """
        if not history:
            return ""

        formatted = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"用户: {content}")
            else:
                formatted.append(f"助手: {content}")

        return "\n".join(formatted)

    def rewrite_query_with_history(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        根据对话历史重写查询（解决代词指代问题）

        Args:
            query: 原始查询
            conversation_history: 对话历史

        Returns:
            重写后的查询
        """
        if not conversation_history:
            return query

        history_str = self._format_conversation_history(conversation_history[-4:])

        prompt = PromptTemplate(
            template="""你是一个查询重写助手。请根据对话历史，将用户当前的问题重写为一个独立、完整的问题。

对话历史:
{history}

用户当前问题: {query}

重写规则：
1. 如果问题中包含代词（如"它"、"那道菜"、"刚才"等），请替换为具体的菜名或内容
2. 如果问题是对之前问题的追问，请补充完整上下文
3. 如果问题本身已经完整清晰，则保持原样
4. 只输出重写后的问题，不要解释

重写后的问题:""",
            input_variables=["history", "query"]
        )

        chain = (
            {"history": lambda _: history_str, "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            rewritten = chain.invoke(query).strip()
            if rewritten and rewritten != query:
                logger.info(f"查询已重写: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"查询重写失败: {e}")
            return query