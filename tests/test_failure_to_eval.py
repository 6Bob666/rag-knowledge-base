from services.failure_to_eval import convert_labels_to_cases, write_draft


def test_convert_labels_to_cases_keeps_retrieval_and_knowledge_gap():
    records = [
        {
            "request_id": "a",
            "status": "no_documents",
            "question": "报销流程是什么？",
            "label": "retrieval",
        },
        {
            "request_id": "b",
            "status": "no_documents",
            "question": "违约金条款怎么解释？",
            "label": "knowledge_gap",
        },
        {
            "request_id": "c",
            "status": "llm_error",
            "question": "大模型坏了的问题",
            "label": "service_error",
        },
    ]

    cases = convert_labels_to_cases(records)

    assert len(cases) == 2
    assert cases[0]["failure_label"] == "retrieval"
    assert cases[1]["failure_label"] == "knowledge_gap"
    assert cases[0]["relevant_chunk_ids"] == []
    assert cases[1]["ground_truth_answer"] == ""


def test_convert_skips_blank_question():
    cases = convert_labels_to_cases(
        [
            {
                "request_id": "x",
                "status": "no_documents",
                "question": "   ",
                "label": "retrieval",
            }
        ]
    )
    assert cases == []


def test_write_draft_writes_json(tmp_path):
    cases = [
        {
            "id": "failure-001",
            "question": "报销流程是什么？",
            "relevant_chunk_ids": [],
        }
    ]
    output = tmp_path / "draft.json"

    write_draft(cases, output)

    import json

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded[0]["id"] == "failure-001"
