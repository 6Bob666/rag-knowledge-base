import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from models import Document
from routers.documents import router
from services.dependencies import get_store


TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    bind=TEST_ENGINE, autocommit=False, autoflush=False
)


class FakeStore:
    def __init__(self):
        self.added_texts = []
        self.deleted_ids = []

    def add_texts(self, texts, metadatas=None, ids=None):
        self.added_texts.extend(texts)

    def delete_by_document_id(self, document_id):
        self.deleted_ids.append(document_id)
        return 1


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=TEST_ENGINE)
    Base.metadata.create_all(bind=TEST_ENGINE)

    app = FastAPI()
    app.include_router(router, prefix="/documents")
    app.dependency_overrides[get_db] = override_get_db
    # 所有请求共用同一个 FakeStore，才能断言"幂等命中时没有再写向量库"。
    fake_store = FakeStore()
    app.dependency_overrides[get_store] = lambda: fake_store

    with TestClient(app) as test_client:
        test_client.fake_store = fake_store
        yield test_client


def test_upload_document(client):
    content = "这是第一句。这是第二句。".encode("utf-8")
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["chunk_count"] == 1
    assert data["document_id"]


def test_list_documents_starts_empty(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["documents"] == []


def test_delete_document(client):
    content = "需要删除的内容".encode("utf-8")
    upload_response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", content, "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id


def upload(client, filename: str, text: str):
    return client.post(
        "/documents/upload",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )


def test_uploading_same_content_twice_is_idempotent(client):
    """同一份内容重复上传：不重复切分，也不产生新文档。"""
    first = upload(client, "same.txt", "这是同一份内容。")
    second = upload(client, "same.txt", "这是同一份内容。")

    assert first.json()["status"] == "indexed"
    assert second.json()["status"] == "unchanged"
    assert second.json()["document_id"] == first.json()["document_id"]
    assert second.json()["chunk_count"] == first.json()["chunk_count"]
    assert len(client.get("/documents").json()["documents"]) == 1


def test_uploading_changed_content_replaces_old_version(client):
    first = upload(client, "doc.txt", "第一版内容。")
    second = upload(client, "doc.txt", "第二版内容，已经改过。")

    assert second.json()["status"] == "indexed"
    assert second.json()["document_id"] != first.json()["document_id"]

    documents = client.get("/documents").json()["documents"]
    current_ids = [item["document_id"] for item in documents]
    assert current_ids == [second.json()["document_id"]]


def test_same_content_under_different_name_is_indexed_separately(client):
    """判据是内容 + 文件名：换个名字上传仍然视为新文档。"""
    first = upload(client, "a.txt", "共享内容。")
    second = upload(client, "b.txt", "共享内容。")

    assert second.json()["status"] == "indexed"
    assert len(client.get("/documents").json()["documents"]) == 2


def test_unchanged_upload_does_not_call_vector_store(client):
    """幂等命中时不能再去写向量库，否则等于白烧一次 embedding。"""
    upload(client, "same.txt", "内容不变。")
    client.fake_store.added_texts.clear()

    upload(client, "same.txt", "内容不变。")

    assert client.fake_store.added_texts == []
