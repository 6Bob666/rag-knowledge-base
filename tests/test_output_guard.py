import pytest

from services.output_guard import UnsafeOutputError, validate_answer


def test_validate_answer_strips_surrounding_whitespace():
    assert validate_answer("  正常回答  ") == "正常回答"


def test_validate_answer_rejects_empty_answer():
    with pytest.raises(UnsafeOutputError):
        validate_answer("  ")


@pytest.mark.parametrize(
    "answer",
    [
        "api_key=sk-test-secret-value",
        "password: very-secret-password",
        "sk-1234567890abcdef",
    ],
)
def test_validate_answer_rejects_obvious_secrets(answer):
    with pytest.raises(UnsafeOutputError):
        validate_answer(answer)


def test_validate_answer_allows_normal_api_discussion():
    assert validate_answer("API 是应用程序编程接口。") == "API 是应用程序编程接口。"
