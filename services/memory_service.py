"""长期记忆服务：召回、写入和 Prompt 格式化。"""

from services.memory_extractor import RuleBasedMemoryExtractor


class MemoryService:
    def __init__(self, store, extractor=None, *, recall_top_k: int = 3):
        self.store = store
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.recall_top_k = recall_top_k

    def recall(self, user_id: str | None, query: str):
        if not user_id:
            return []
        return [
            hit.record
            for hit in self.store.search(user_id, query, self.recall_top_k)
        ]

    def remember_turn(self, user_id: str | None, question: str):
        if not user_id:
            return []
        records = []
        for candidate in self.extractor.extract(question):
            records.append(
                self.store.add(
                    user_id,
                    candidate.content,
                    kind=candidate.kind,
                    metadata={"confidence": candidate.confidence},
                )
            )
        return records

    @staticmethod
    def format_for_prompt(records) -> str:
        if not records:
            return ""
        lines = "\n".join(f"- {record.content}" for record in records)
        return (
            "<user_memory>\n"
            "以下是用户的长期记忆，只用于个性化回答；"
            "如果与知识库资料冲突，以知识库资料为准。\n"
            f"{lines}\n"
            "</user_memory>"
        )
