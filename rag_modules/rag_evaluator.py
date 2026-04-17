"""
RAG 系统评估模块
使用大模型对 RAG 系统的回答进行多维度评估
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """单个评估结果"""
    question: str
    reference_answer: Optional[str] = None
    rag_answer: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    
    # 评分维度
    relevance_score: float = 0.0  # 相关性 (0-5)
    accuracy_score: float = 0.0   # 准确性 (0-5)
    completeness_score: float = 0.0  # 完整性 (0-5)
    clarity_score: float = 0.0   # 清晰度 (0-5)
    hallucination_score: float = 0.0  # 幻觉程度 (0-5，越低越好)
    
    overall_score: float = 0.0   # 综合得分
    
    feedback: str = ""  # 详细反馈
    evaluation_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(**data)


@dataclass
class EvaluationSummary:
    """评估汇总"""
    total_tests: int = 0
    avg_relevance: float = 0.0
    avg_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_clarity: float = 0.0
    avg_hallucination: float = 0.0
    avg_overall: float = 0.0
    results: List[EvaluationResult] = field(default_factory=list)
    summary_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "avg_relevance": self.avg_relevance,
            "avg_accuracy": self.avg_accuracy,
            "avg_completeness": self.avg_completeness,
            "avg_clarity": self.avg_clarity,
            "avg_hallucination": self.avg_hallucination,
            "avg_overall": self.avg_overall,
            "results": [r.to_dict() for r in self.results],
            "summary_timestamp": self.summary_timestamp
        }


class RAGEvaluator:
    """RAG 评估器"""
    
    def __init__(
        self,
        rag_system,
        llm_model: Optional[str] = None,
        eval_output_dir: str = "./evaluation_results"
    ):
        """
        初始化评估器
        
        Args:
            rag_system: RAG 系统实例
            llm_model: 用于评估的大模型（默认使用 RAG 系统的模型）
            eval_output_dir: 评估结果输出目录
        """
        self.rag_system = rag_system
        self.llm_model = llm_model or rag_system.config.llm_model
        self.eval_output_dir = Path(eval_output_dir)
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_evaluation_prompt(
        self,
        question: str,
        rag_answer: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_answer: Optional[str] = None
    ) -> str:
        """构建评估提示词"""
        
        docs_content = "\n\n".join([
            f"文档 {i+1}:\n{doc.get('content', '')[:500]}..."
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        reference_section = f"\n参考回答:\n{reference_answer}\n" if reference_answer else ""
        
        prompt = f"""你是一位专业的 RAG 系统评估专家。请根据以下信息对 RAG 系统的回答进行评估。

问题: {question}

检索到的文档:
{docs_content}

RAG 系统回答:
{rag_answer}
{reference_section}

请从以下维度进行评分（0-5分），并提供详细反馈：

1. 相关性 (Relevance): 回答是否与问题相关
2. 准确性 (Accuracy): 回答内容是否准确，是否基于检索文档
3. 完整性 (Completeness): 回答是否完整覆盖了问题的所有方面
4. 清晰度 (Clarity): 回答是否清晰易懂
5. 幻觉程度 (Hallucination): 回答中是否存在未在检索文档中出现的虚构内容（分数越低越好）

请以 JSON 格式返回，格式如下：
{{
    "relevance_score": 4.5,
    "accuracy_score": 4.0,
    "completeness_score": 3.5,
    "clarity_score": 4.0,
    "hallucination_score": 1.0,
    "overall_score": 3.8,
    "feedback": "详细的评估反馈..."
}}

只返回 JSON，不要其他内容。
"""
        return prompt
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """解析评估响应"""
        try:
            json_match = response.find('{')
            if json_match != -1:
                response = response[json_match:]
            json_end = response.rfind('}')
            if json_end != -1:
                response = response[:json_end + 1]
            return json.loads(response)
        except Exception as e:
            logger.warning(f"解析评估响应失败: {e}")
            return {
                "relevance_score": 0.0,
                "accuracy_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "hallucination_score": 5.0,
                "overall_score": 0.0,
                "feedback": f"解析失败: {str(e)}"
            }
    
    def evaluate_single(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        use_conversation: bool = False
    ) -> EvaluationResult:
        """
        评估单个问题
        
        Args:
            question: 问题
            reference_answer: 参考回答（可选）
            use_conversation: 是否使用对话记忆
            
        Returns:
            评估结果
        """
        logger.info(f"评估问题: {question}")
        
        result = EvaluationResult(
            question=question,
            reference_answer=reference_answer
        )
        
        try:
            rag_answer = self.rag_system.ask_question(
                question,
                stream=False,
                use_conversation=use_conversation
            )
            result.rag_answer = rag_answer
            
            if hasattr(self.rag_system, 'retrieval_module') and hasattr(self.rag_system, 'generation_module'):
                try:
                    retrieved_docs = self.rag_system.retrieval_module.search(question, top_k=3)
                    result.retrieved_docs = [
                        {"content": doc.page_content if hasattr(doc, 'page_content') else doc.get('content', ''), 
                         "score": doc.metadata.get('score', 0) if hasattr(doc, 'metadata') else doc.get('score', 0),
                         "dish_name": doc.metadata.get('dish_name', '未知') if hasattr(doc, 'metadata') else '未知'}
                        for doc in retrieved_docs
                    ]
                    logger.info(f"检索到 {len(result.retrieved_docs)} 个文档")
                except Exception as e:
                    logger.warning(f"获取检索文档失败: {e}")
            
            eval_prompt = self._build_evaluation_prompt(
                question, rag_answer, result.retrieved_docs, reference_answer
            )
            
            eval_response = self._call_llm_for_evaluation(eval_prompt)
            eval_data = self._parse_evaluation_response(eval_response)
            
            result.relevance_score = eval_data.get("relevance_score", 0.0)
            result.accuracy_score = eval_data.get("accuracy_score", 0.0)
            result.completeness_score = eval_data.get("completeness_score", 0.0)
            result.clarity_score = eval_data.get("clarity_score", 0.0)
            result.hallucination_score = eval_data.get("hallucination_score", 5.0)
            result.overall_score = eval_data.get("overall_score", 0.0)
            result.feedback = eval_data.get("feedback", "")
            
            logger.info(f"综合得分: {result.overall_score}")
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            result.feedback = f"评估过程出错: {str(e)}"
        
        return result
    
    def _call_llm_for_evaluation(self, prompt: str) -> str:
        """调用 LLM 进行评估"""
        try:
            if hasattr(self.rag_system, 'generation_module'):
                generation = self.rag_system.generation_module
                
                if hasattr(generation, 'llm'):
                    response = generation.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    return str(response)
                
                elif hasattr(generation, 'generate_basic_answer'):
                    return generation.generate_basic_answer(prompt, [])
            
            return '{"relevance_score": 0, "accuracy_score": 0, "completeness_score": 0, "clarity_score": 0, "hallucination_score": 5, "overall_score": 0, "feedback": "评估 LLM 不可用"}'
            
        except Exception as e:
            logger.error(f"调用评估 LLM 失败: {e}")
            return f'{{"relevance_score": 0, "accuracy_score": 0, "completeness_score": 0, "clarity_score": 0, "hallucination_score": 5, "overall_score": 0, "feedback": "调用 LLM 失败: {str(e)}"}}'
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        use_conversation: bool = False
    ) -> EvaluationSummary:
        """
        批量评估
        
        Args:
            test_cases: 测试用例列表，每个包含 question 和可选的 reference_answer
            use_conversation: 是否使用对话记忆
            
        Returns:
            评估汇总
        """
        summary = EvaluationSummary()
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"进度: {i+1}/{len(test_cases)}")
            
            result = self.evaluate_single(
                question=test_case.get("question", ""),
                reference_answer=test_case.get("reference_answer"),
                use_conversation=use_conversation
            )
            summary.results.append(result)
        
        summary.total_tests = len(summary.results)
        if summary.total_tests > 0:
            summary.avg_relevance = sum(r.relevance_score for r in summary.results) / summary.total_tests
            summary.avg_accuracy = sum(r.accuracy_score for r in summary.results) / summary.total_tests
            summary.avg_completeness = sum(r.completeness_score for r in summary.results) / summary.total_tests
            summary.avg_clarity = sum(r.clarity_score for r in summary.results) / summary.total_tests
            summary.avg_hallucination = sum(r.hallucination_score for r in summary.results) / summary.total_tests
            summary.avg_overall = sum(r.overall_score for r in summary.results) / summary.total_tests
        
        return summary
    
    def save_results(self, summary: EvaluationSummary, filename: Optional[str] = None):
        """保存评估结果"""
        if filename is None:
            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.eval_output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {filepath}")
        return filepath
    
    def print_summary(self, summary: EvaluationSummary):
        """打印评估汇总"""
        print("\n" + "="*60)
        print("RAG 系统评估汇总")
        print("="*60)
        print(f"总测试数: {summary.total_tests}")
        print(f"平均相关性: {summary.avg_relevance:.2f}/5")
        print(f"平均准确性: {summary.avg_accuracy:.2f}/5")
        print(f"平均完整性: {summary.avg_completeness:.2f}/5")
        print(f"平均清晰度: {summary.avg_clarity:.2f}/5")
        print(f"平均幻觉程度: {summary.avg_hallucination:.2f}/5 (越低越好)")
        print(f"综合得分: {summary.avg_overall:.2f}/5")
        print("="*60 + "\n")
