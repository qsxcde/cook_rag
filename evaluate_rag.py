"""
RAG 系统评估脚本
使用大模型对 RAG 系统进行多维度评估
"""

import json
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DEFAULT_CONFIG
from main import RecipeRAGSystem
from rag_modules.rag_evaluator import RAGEvaluator


def load_test_cases(filepath: str = "evaluation_test_cases.json") -> list:
    """加载测试用例"""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"测试用例文件不存在: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("RAG 系统评估工具")
    print("="*60)
    
    logger.info("正在初始化 RAG 系统...")
    rag = RecipeRAGSystem(DEFAULT_CONFIG)
    rag.initialize_system()
    rag.build_knowledge_base()
    logger.info("RAG 系统初始化完成")
    
    evaluator = RAGEvaluator(
        rag_system=rag,
        eval_output_dir="./evaluation_results"
    )
    
    test_cases = load_test_cases("evaluation_test_cases.json")
    
    if not test_cases:
        logger.info("使用默认测试用例")
        test_cases = [
            {"question": "白灼虾怎么做？"},
            {"question": "推荐几道简单的素菜"},
            {"question": "清蒸鲈鱼需要多长时间？"}
        ]
    
    print(f"\n开始评估，共 {len(test_cases)} 个测试用例...\n")
    
    summary = evaluator.evaluate_batch(test_cases, use_conversation=False)
    
    evaluator.print_summary(summary)
    
    saved_path = evaluator.save_results(summary)
    
    print(f"\n详细结果已保存到: {saved_path}")
    print("\n评估完成！")


if __name__ == "__main__":
    main()
