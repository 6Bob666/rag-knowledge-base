# evaluate_rag.py
import json
import asyncio
from services.vector_store import VectorStore
from services.llm_service import ask_llm, rewrite_query

# 初始化向量存储
store = VectorStore()

# 定义测试用例（问题，参考答案）
test_cases = [
    {
        "question": "CNN 的全称是什么？",
        "ground_truth": "卷积神经网络（Convolutional Neural Network）"
    },
    {
        "question": "CNN 有哪些核心结构？",
        "ground_truth": "卷积层、池化层和全连接层"
    },
    {
        "question": "ResNet 解决了什么问题？",
        "ground_truth": "梯度消失问题"
    },
    {
        "question": "CNN 在哪些领域有应用？",
        "ground_truth": "图像分类、目标检测、图像分割、人脸识别"
    },
    {
        "question": "Transformer 是哪一年提出的？",
        "ground_truth": "2017年"
    }
]


async def run_rag_system(question: str):
    """模拟你的 RAG 系统，返回生成的答案和检索到的上下文列表"""
    # 查询改写（这里直接调用你的改写函数）
    try:
        optimized = rewrite_query(question)
    except:
        optimized = question

    # 混合检索
    contexts = store.hybrid_search(optimized, top_k=3)
    context_str = "\n".join(contexts)

    # 生成回答
    answer = ask_llm(question, context_str)

    return answer, contexts


async def collect_results():
    results = []
    for case in test_cases:
        answer, contexts = await run_rag_system(case["question"])
        results.append({
            "question": case["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": case["ground_truth"]
        })
    return results


if __name__ == "__main__":
    # 收集结果
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(collect_results())

    # 保存到文件供 RAGAS 使用
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("✅ 评估数据已保存到 eval_results.json")