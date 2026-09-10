# services/document.py
from fastapi import UploadFile
import os
import re
import aiofiles
from .retrieval import retriever

UPLOAD_DIR = "uploads"


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """按句子切分文本，并控制 chunk 大小和相邻 chunk 的重叠内容。"""
    if not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    sentences = re.findall(r".+?[。！？!?；;\n]|.+$", text, flags=re.S)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 超长句子没有自然边界时，按字符补充切分。
        while len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = current[-chunk_overlap:]
            chunks.append(sentence[:chunk_size].strip())
            sentence = sentence[chunk_size - chunk_overlap:]

        if current and len(current) + len(sentence) > chunk_size:
            chunks.append(current.strip())
            current = current[-chunk_overlap:]
        current += sentence

    if current.strip():
        chunks.append(current.strip())
    return chunks

async def save_document(file: UploadFile) -> dict:
    # 确保上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # 保存文件
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # 读取文本内容（假设是 txt 文件，如果是 pdf 需要额外解析）
    text_content = content.decode("utf-8")

    # 将文档添加到检索服务
    retriever.add_document(text_content)

    return {"filename": file.filename, "size": len(content), "status": "indexed"}
