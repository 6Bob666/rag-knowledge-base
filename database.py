from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from config import settings


engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema(engine_) -> list[str]:
    """补齐 create_all 不会处理的列变更。

    `Base.metadata.create_all` 只建缺失的表，不会给已存在的表加字段：
    老库升级时 `documents.content_hash` 不会自动出现，查询会直接报错。
    这里做一次轻量迁移，返回真正执行过的语句，方便启动日志留痕。

    真实项目请换成 Alembic 迁移；这里保持零依赖，把机制讲清楚就够。
    """
    inspector = inspect(engine_)
    if "documents" not in inspector.get_table_names():
        return []

    existing = {column["name"] for column in inspector.get_columns("documents")}
    executed = []
    if "content_hash" not in existing:
        with engine_.begin() as connection:
            connection.execute(
                text("ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64)")
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_content_hash "
                    "ON documents (content_hash)"
                )
            )
        executed.append("ALTER TABLE documents ADD COLUMN content_hash")
    return executed
