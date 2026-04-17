"""
运行所有单元测试
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪  运行 RAG 项目单元测试  🧪")
    print("=" * 60)

    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).parent / "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出统计
    print("\n" + "=" * 60)
    print("📊  测试结果统计")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    # 返回是否全部通过
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
