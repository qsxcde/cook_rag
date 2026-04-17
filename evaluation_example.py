"""
RAG 评估使用示例
展示如何使用评估模块
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DEFAULT_CONFIG
from main import RecipeRAGSystem
from rag_modules.rag_evaluator import RAGEvaluator


def example_single_evaluation():
    """单个问题评估示例"""
    print("\n" + "="*60)
    print("示例 1: 单个问题评估")
    print("="*60)
    
    rag = RecipeRAGSystem(DEFAULT_CONFIG)
    rag.initialize_system()
    rag.build_knowledge_base()
    
    evaluator = RAGEvaluator(rag)
    
    question = "白灼虾怎么做？"
    reference = "白灼虾的做法：鲜虾洗净，水烧开加姜葱料酒，放入虾煮2-3分钟，捞出配蘸料食用。"
    
    result = evaluator.evaluate_single(question, reference_answer=reference)
    
    print(f"\n问题: {result.question}")
    print(f"RAG 回答: {result.rag_answer[:100]}...")
    print(f"综合得分: {result.overall_score}/5")
    print(f"反馈: {result.feedback}")


def example_batch_evaluation():
    """批量评估示例"""
    print("\n" + "="*60)
    print("示例 2: 批量评估")
    print("="*60)
    
    rag = RecipeRAGSystem(DEFAULT_CONFIG)
    rag.initialize_system()
    rag.build_knowledge_base()
    
    evaluator = RAGEvaluator(rag)
    
    test_cases = [
        {"question": "白灼虾怎么做？"},
        {"question": "推荐几道简单的素菜"},
        {"question": "清蒸鲈鱼需要多长时间？"}
    ]
    
    summary = evaluator.evaluate_batch(test_cases)
    evaluator.print_summary(summary)
    evaluator.save_results(summary)


def example_custom_test_cases():
    """自定义测试用例示例"""
    print("\n" + "="*60)
    print("示例 3: 使用自定义测试用例")
    print("="*60)
    
    rag = RecipeRAGSystem(DEFAULT_CONFIG)
    rag.initialize_system()
    rag.build_knowledge_base()
    
    evaluator = RAGEvaluator(rag)
    
    custom_cases = [
        {
            "question": "红烧肉用什么肉？",
            "reference_answer": "红烧肉通常用带皮五花肉，肥瘦相间。"
        },
        {
            "question": "糖醋汁比例",
            "reference_answer": "糖醋汁常见比例：料酒1、酱油2、醋3、糖4、水5。"
        }
    ]
    
    summary = evaluator.evaluate_batch(custom_cases)
    evaluator.print_summary(summary)


if __name__ == "__main__":
    print("RAG 评估模块使用示例")
    print("请取消注释下面的示例来运行\n")
    
    # example_single_evaluation()
    # example_batch_evaluation()
    # example_custom_test_cases()
