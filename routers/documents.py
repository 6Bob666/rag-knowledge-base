import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from database import get_db
from models import Document
from services.dependencies import get_store
from services.text_splitter import split_text
from services.upload_guard import compute_content_hash, upload_lock
from config import settings
from logging_config import logger

if TYPE_CHECKING:
    from services.vector_store import VectorStore

router = APIRouter(tags=["Documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db), store: "VectorStore" = Depends(get_store)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    content = await file.read()
    content_hash = compute_content_hash(content)

    # 临界区按文件名串行：并发上传同一个文件时，避免两次"先查后写"交错，
    # 导致同一份内容被重复入库、旧版本被误删。
    with upload_lock.acquire(file.filename):
        return await _index_document(
            file=file,
            content=content,
            content_hash=content_hash,
            db=db,
            store=store,
        )


async def _index_document(
    *,
    file: UploadFile,
    content: bytes,
    content_hash: str,
    db: Session,
    store: "VectorStore",
):
    """幂等入库：内容没变直接返回，内容变了才替换旧版本。"""
    existing_same_content = db.query(Document).filter(
        Document.filename == file.filename,
        Document.content_hash == content_hash,
        Document.status == "indexed",
    ).first()
    if existing_same_content is not None:
        logger.info(
            "文件内容未变化，跳过重复入库 filename=%s document_id=%s",
            file.filename,
            existing_same_content.document_id,
        )
        return {
            "document_id": existing_same_content.document_id,
            "filename": file.filename,
            "chunk_count": existing_same_content.chunk_count,
            "status": "unchanged",
        }

    document_id = str(uuid.uuid4())

    # 先保留旧版本，只有新版本成功后才清理它们。
    old_documents = db.query(Document).filter(
        Document.filename == file.filename,
        Document.status == "indexed",
    ).all()

    document = Document(
        document_id=document_id,
        filename=file.filename,
        file_size=len(content),
        content_hash=content_hash,
        status="processing",
    )
    db.add(document)
    db.commit()

    try:
        text = content.decode("utf-8")
        chunks = split_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        metadatas = [
            {
                "source": file.filename,
                "document_id": document_id,
                "chunk_index": index,
                "chunk_id": f"{document_id}#chunk-{index}",
            }
            for index in range(len(chunks))
        ]
        ids = [metadata["chunk_id"] for metadata in metadatas]
        await run_in_threadpool(store.add_texts, chunks, metadatas, ids)
        document.chunk_count = len(chunks)
        document.status = "indexed"
        db.commit()

        result = {
            "document_id": document_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "status": "indexed",
        }
    except Exception as exc:
        # 清理本次可能已经写入的部分 chunk，避免残留半成品。
        store.delete_by_document_id(document_id)
        document.status = "failed"
        document.error_message = str(exc)
        db.commit()
        raise HTTPException(status_code=400, detail="文档处理失败") from exc

    # 新版本已成功后再清理旧版本；清理失败不影响新版本继续服务。
    for old_document in old_documents:
        try:
            store.delete_by_document_id(old_document.document_id)
            db.delete(old_document)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("旧版本清理失败，请稍后重试 %s", old_document.document_id)

    return result


@router.get("")
async def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return {"documents": documents}


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db), store: "VectorStore" = Depends(get_store)):
    document = db.query(Document).filter(Document.document_id == document_id).first()
    if document is None:
        return {"document_id": document_id, "deleted_chunks": 0}
    deleted_count = store.delete_by_document_id(document_id)
    db.delete(document)
    db.commit()
    return {"document_id": document_id, "deleted_chunks": deleted_count}
