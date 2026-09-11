"""老库升级路径：create_all 不会补列，ensure_schema 要补上。"""

from sqlalchemy import create_engine, inspect, text

from database import ensure_schema


LEGACY_TABLE = """
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    chunk_count INTEGER,
    status VARCHAR(20),
    error_message TEXT,
    created_at DATETIME,
    updated_at DATETIME
)
"""


def legacy_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text(LEGACY_TABLE))
    return engine


def test_ensure_schema_adds_missing_content_hash_column(tmp_path):
    engine = legacy_engine(tmp_path)
    columns_before = {
        column["name"] for column in inspect(engine).get_columns("documents")
    }
    assert "content_hash" not in columns_before

    executed = ensure_schema(engine)

    assert executed == ["ALTER TABLE documents ADD COLUMN content_hash"]
    columns_after = {
        column["name"] for column in inspect(engine).get_columns("documents")
    }
    assert "content_hash" in columns_after


def test_ensure_schema_is_idempotent(tmp_path):
    engine = legacy_engine(tmp_path)
    ensure_schema(engine)

    # 第二次启动不能再执行一遍 ALTER，否则会因为列已存在而启动失败。
    assert ensure_schema(engine) == []


def test_ensure_schema_skips_missing_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")

    assert ensure_schema(engine) == []
