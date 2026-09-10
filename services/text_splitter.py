import re


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
