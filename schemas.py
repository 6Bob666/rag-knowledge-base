from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, List   # 增加 List 导入

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    created_at: datetime


class QuestionRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="可选的会话标识；不传时按单轮问题处理",
    )
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="返回的上下文数量")
    enable_retry: bool = Field(
        default=True,
        description="改写后的查询无结果时，是否允许用原问题补救一次",
    )


class ContextResponse(BaseModel):
    text: str
    source: str = "unknown"
    chunk_id: str = "unknown"


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunk_count: int


class AnswerResponse(BaseModel):
    answer: str
    contexts: List[ContextResponse]
