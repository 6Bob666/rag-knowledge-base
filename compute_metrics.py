# compute_metrics.py
import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# 设置 DeepSeek 作为评估 LLM（RAGAS 内部会自动使用 OPENAI 兼容接口）
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 未配置，请先设置环境变量")
os.environ["OPENAI_API_KEY"] = _api_key
if base_url := os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = base_url

# 加载之前保存的结果
with open("eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_list([
    {
        "question": d["question"],
        "answer": d["answer"],
        "contexts": [
            context["text"] if isinstance(context, dict) else context
            for context in d["contexts"]
        ],
        "ground_truth": d["ground_truth"]
    }
    for d in data
])

# 计算指标
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print("📊 评估结果：")
print(result.to_pandas().to_string(index=False))

# 导出详细报告
df = result.to_pandas()
df.to_csv("rag_evaluation_report.csv", index=False)
print("✅ 详细报告已保存到 rag_evaluation_report.csv")
