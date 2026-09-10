# evaluate_rag_via_api.py
import json
import requests

# 你的 FastAPI 服务地址
API_URL = "http://127.0.0.1:8000/chat/ask"

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

results = []

for case in test_cases:
    resp = requests.post(API_URL, json={"question": case["question"]})
    if resp.status_code != 200:
        print(f"❌ 请求失败: {case['question']}, 状态码: {resp.status_code}")
        continue
    data = resp.json()
    results.append({
        "question": case["question"],
        "answer": data.get("answer", ""),
        "contexts": data.get("contexts", []),
        "ground_truth": case["ground_truth"]
    })
    print(f"✅ {case['question']} -> 已回答")

# 保存结果
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n✅ 评估数据已保存到 eval_results.json")