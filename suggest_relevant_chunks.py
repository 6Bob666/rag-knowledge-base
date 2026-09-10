"""根据答案关键词，批量初筛可能相关的 chunk，辅助人工标注。"""

import sys

import chromadb

from config import settings


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


QUESTIONS = [
    {
        "question": "深度学习是什么？",
        "keywords": ["深度学习是机器学习", "多层神经网络", "模式识别"],
    },
    {
        "question": "自然语言处理包括哪些任务？",
        "keywords": ["文本分类", "情感分析", "机器翻译", "自然语言处理"],
    },
    {
        "question": "池化层的作用是什么？",
        "keywords": ["下采样", "降低特征图", "减少计算量", "平移不变性"],
    },
    {
        "question": "卷积层的作用是什么？",
        "keywords": ["提取特征", "卷积核", "边缘", "纹理"],
    },
    {
        "question": "全连接层的作用是什么？",
        "keywords": ["分类", "整合", "总结好的特征"],
    },
    {
        "question": "RNN 主要用于处理什么数据？",
        "keywords": ["序列数据", "循环神经网络", "文本和语音"],
    },
    {
        "question": "BERT 是什么？",
        "keywords": ["BERT", "预训练语言模型", "Google"],
    },
    {
        "question": "GPT 系列模型由谁开发？",
        "keywords": ["GPT", "OpenAI", "文本生成"],
    },
    {
        "question": "CNN 的训练通常需要什么？",
        "keywords": ["标注数据", "GPU", "计算资源"],
    },
    {
        "question": "池化层常用的有哪两种？",
        "keywords": ["最大池化", "平均池化", "Max Pooling", "Average Pooling"],
    },
    {
        "question": "卷积核是什么？",
        "keywords": ["卷积核", "加权平均", "权值"],
    },
    {
        "question": "全连接层占神经网络多少参数？",
        "keywords": ["80%", "参数"],
    },
    {
        "question": "Vision Transformer 是什么？",
        "keywords": ["Vision Transformer", "ViT", "Transformer 架构"],
    },
    {
        "question": "池化层引入了什么不变性？",
        "keywords": ["平移不变性", "翻转变形"],
    },
    {
        "question": "神经网络由什么组成？",
        "keywords": ["人工神经元", "大量的人工神经元"],
    },
]


def main() -> None:
    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(name="kb_docs")
    data = collection.get(include=["documents", "metadatas"])

    for item in QUESTIONS:
        print(f"\n### {item['question']}")
        found = False
        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            document = document.replace("\u200b", "").replace("\ufeff", "")
            for keyword in item["keywords"]:
                if keyword in document:
                    source = (metadata or {}).get("source", "?")
                    print(f"- {chunk_id} | {source} | {document[:90]}")
                    found = True
                    break
        if not found:
            print("  （无关键词命中）")


if __name__ == "__main__":
    main()
