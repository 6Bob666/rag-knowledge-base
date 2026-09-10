import pytest
from pydantic import ValidationError

from schemas import AnswerResponse, ContextResponse, QuestionRequest


def test_question_request_accepts_valid_input():
    request = QuestionRequest(question="什么是 CNN？", top_k=5)
    assert request.question == "什么是 CNN？"
    assert request.top_k == 5


def test_question_request_uses_default_top_k():
    request = QuestionRequest(question="什么是 CNN？")
    assert request.top_k == 3


def test_question_request_accepts_optional_conversation_id():
    request = QuestionRequest(
        conversation_id="demo-001",
        question="它有什么应用？",
    )
    assert request.conversation_id == "demo-001"


def test_question_request_defaults_enable_retry_true():
    request = QuestionRequest(question="什么是 CNN？")
    assert request.enable_retry is True


def test_question_request_accepts_enable_retry_false():
    request = QuestionRequest(
        question="什么是 CNN？",
        enable_retry=False,
    )
    assert request.enable_retry is False


def test_question_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QuestionRequest(question="")


def test_question_request_rejects_top_k_out_of_range():
    with pytest.raises(ValidationError):
        QuestionRequest(question="问题", top_k=0)
    with pytest.raises(ValidationError):
        QuestionRequest(question="问题", top_k=11)


def test_context_response_has_source_defaults():
    context = ContextResponse(text="片段")
    assert context.source == "unknown"
    assert context.chunk_id == "unknown"


def test_answer_response_parses_context_records():
    answer = AnswerResponse(
        answer="答案",
        contexts=[
            {"text": "片段", "source": "a.txt", "chunk_id": "a.txt#chunk-0"}
        ],
    )
    assert answer.answer == "答案"
    assert answer.contexts[0].source == "a.txt"
    assert answer.contexts[0].chunk_id == "a.txt#chunk-0"
