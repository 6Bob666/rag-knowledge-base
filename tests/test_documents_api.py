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
    app.dependency_overrides[get_store] = lambda: FakeStore()

    with TestClient(app) as test_client:
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
