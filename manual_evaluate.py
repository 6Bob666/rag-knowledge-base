# manual_evaluate.py
import json
import os
from openai import OpenAI

# ===== 1. 初始化 DeepSeek 客户端 =====
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 未配置，请先设置环境变量")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ===== 2. 加载之前生成的评估结果 =====
with open("eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ===== 3. 调试：先测试 LLM 是否能正常调用 =====
def test_llm():
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "请只回答「是」或「否」：1+1=2 吗？"}],
            temperature=0.0,
            max_tokens=5
        )
        text = resp.choices[0].message.content.strip()
        print(f"[DEBUG] LLM 连通性测试返回: {repr(text)}")
        return True
    except Exception as e:
        print(f"[DEBUG] LLM 调用失败: {e}")
        return False

if not test_llm():
    print("❌ DeepSeek API 无法调用，请检查 API Key / Base URL / 网络")
    exit(1)

# ===== 4. 调用 LLM 判断（严格解析「是/否」）=====
def llm_judge(prompt: str) -> bool:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个严格的评估助手。请只回答「是」或「否」。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=5
    )
    answer = resp.choices[0].message.content.strip()
    # 只要开头包含「是」就认为 True
    return answer.startswith("是")

# ===== 5. 把一个段落拆成句子 =====
def split_sentences(text: str) -> list[str]:
    # 统一句号
    for ch in ["。", "！", "？", "!", "?"]:
        text = text.replace(ch, "。")
    parts = [s.strip() for s in text.split("。") if len(s.strip()) >= 4]
    return parts

# ===== 6. 逐条评估 =====
results = []

for item in data:
    question = item["question"]
    answer = item["answer"]
    contexts = item["contexts"]
    ground_truth = item["ground_truth"]

    context_text = "\n".join(
        context["text"] if isinstance(context, dict) else context
        for context in contexts
    )

    # --- Faithfulness：答案中的每个断言是否都被上下文支持 ---
    ans_sents = split_sentences(answer)
    faithful_count = 0
    for sent in ans_sents:
        prompt = f"""以下是参考资料：
{context_text}

请判断下面这句话的内容，是否可以由上述参考资料直接支持或推断出来。
只回答「是」或「否」，不要解释。
句子：{sent}"""
        if llm_judge(prompt):
            faithful_count += 1
    faithfulness = faithful_count / len(ans_sents) if ans_sents else 0.0

    # --- Context Recall：标准答案中的信息是否出现在上下文里 ---
    gt_sents = split_sentences(ground_truth)
    recall_count = 0
    for gt in gt_sents:
        prompt = f"""以下是参考资料：
{context_text}

请判断下面这条信息，是否在上述参考资料中被明确提及。
只回答「是」或「否」，不要解释。
信息：{gt}"""
        if llm_judge(prompt):
            recall_count += 1
    context_recall = recall_count / len(gt_sents) if gt_sents else 0.0

    results.append({
        "question": question,
        "answer_sentences": len(ans_sents),
        "faithful_supported": faithful_count,
        "faithfulness": round(faithfulness, 4),
        "gt_sentences": len(gt_sents),
        "recall_hit": recall_count,
        "context_recall": round(context_recall, 4)
    })

# ===== 7. 输出结果 =====
print("\n📊 评估结果：")
print(f"{'问题':<28} {'Fai':>5} {'Rec':>5} {'Fai%':>7} {'Rec%':>7}")
print("-" * 60)
tot_f = tot_r = 0.0
for r in results:
    print(f"{r['question']:<28} {r['faithful_supported']}/{r['answer_sentences']:<3} {r['recall_hit']}/{r['gt_sentences']:<3} {r['faithfulness']:>7.2f} {r['context_recall']:>7.2f}")
    tot_f += r["faithfulness"]
    tot_r += r["context_recall"]
n = len(results)
print("-" * 60)
print(f"{'平均值':<28} {'':>5} {'':>5} {tot_f/n:>7.2f} {tot_r/n:>7.2f}")

with open("manual_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n✅ 详细结果已保存到 manual_eval_results.json")
