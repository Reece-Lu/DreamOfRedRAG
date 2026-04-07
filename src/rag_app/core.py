from dataclasses import dataclass
from pathlib import Path
import re

from chromadb import PersistentClient
from openai import OpenAI

from .config import Settings


# 这一组函数负责和 OpenAI 交互：创建客户端、批量做 embedding。
def get_openai_client(settings: Settings | None = None) -> OpenAI:
    if settings and settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key)
    return OpenAI()


def _batched_texts(
    texts: list[str],
    max_batch_chars: int = 8_000,
    max_batch_items: int = 32,
) -> list[list[str]]:
    # Embedding 接口对单次请求的长度和条数都有限制。
    # 这里按“总字符数 + 条数”双重条件分批，避免一次请求过大。
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_chars = 0

    for text in texts:
        text_len = len(text)
        if batch and (batch_chars + text_len > max_batch_chars or len(batch) >= max_batch_items):
            batches.append(batch)
            batch = []
            batch_chars = 0

        batch.append(text)
        batch_chars += text_len

    if batch:
        batches.append(batch)

    return batches


def embed_texts(texts: list[str], settings: Settings, on_batch=None) -> list[list[float]]:
    client = get_openai_client(settings)
    embeddings: list[list[float]] = []

    batches = _batched_texts(texts)
    for index, batch in enumerate(batches, 1):
        if on_batch:
            on_batch(index, len(batches), len(batch), sum(len(item) for item in batch))
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


# 这一组函数负责把原始文本读入，并切成适合检索的 chunk。
def load_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, max_len: int = 1500, overlap: int = 200) -> list[str]:
    # 直接按固定字符窗口切块，简单稳定。
    # 对中文长文本来说，这比依赖段落标点更容易控制 chunk 大小。
    chunks: list[str] = []
    start = 0
    text = text.strip()

    if not text:
        return chunks

    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


# 这一组函数负责 ChromaDB：创建 collection、写入 chunk、做向量检索。
def get_collection(settings: Settings, recreate: bool = False):
    client = PersistentClient(path=str(settings.chroma_dir))
    if recreate:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(name=settings.collection_name)


def ingest_corpus(settings: Settings) -> int:
    text = load_text(settings.text_file)
    chunks = chunk_text(text)
    collection = get_collection(settings, recreate=True)
    max_chunk_len = max((len(chunk) for chunk in chunks), default=0)
    print(f"共切出 {len(chunks)} 个分块，最长分块约 {max_chunk_len} 字符，开始写入 embedding...")

    def show_progress(batch_no: int, batch_total: int, batch_items: int, batch_chars: int) -> None:
        print(f"正在处理批次 {batch_no}/{batch_total}，{batch_items} 个分块，约 {batch_chars} 字符")

    embeddings = embed_texts(chunks, settings, on_batch=show_progress)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": settings.text_file.name, "chunk": i}
            for i in range(len(chunks))
        ],
    )
    print("写入完成")
    return len(chunks)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: int
    source: str
    text: str
    distance: float | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None


def _tokenize_for_keyword_match(text: str) -> list[str]:
    # 轻量中文关键词切分：提取连续中文片段和英文数字词。
    # 不是严格分词器，但对学习项目足够用于“关键词覆盖率”打分。
    cjk_parts = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    latin_parts = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    return cjk_parts + latin_parts


def _keyword_recall_score(question: str, document: str) -> float:
    question_tokens = set(_tokenize_for_keyword_match(question))
    if not question_tokens:
        return 0.0

    doc_text = document.lower()
    hit = sum(1 for token in question_tokens if token.lower() in doc_text)
    return hit / len(question_tokens)


def retrieve(question: str, settings: Settings, top_k: int = 3, candidate_k: int | None = None) -> list[RetrievedChunk]:
    collection = get_collection(settings, recreate=False)
    candidate_count = max(top_k, candidate_k or top_k * 4)
    query_embedding = embed_texts([question], settings)[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items: list[RetrievedChunk] = []
    for doc, meta, distance in zip(docs, metas, distances):
        distance_value = float(distance) if distance is not None else None
        semantic_score = 0.0 if distance_value is None else 1.0 / (1.0 + distance_value)
        keyword_score = _keyword_recall_score(question, str(doc))
        final_score = 0.75 * semantic_score + 0.25 * keyword_score
        items.append(
            RetrievedChunk(
                chunk=int(meta["chunk"]),
                source=str(meta["source"]),
                text=str(doc),
                distance=distance_value,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                final_score=final_score,
            )
        )

    items.sort(key=lambda item: item.final_score or 0.0, reverse=True)
    return items[:top_k]


# 这一组函数负责把检索结果组织成提示词，并调用聊天模型生成答案。
def format_context(chunks: list[RetrievedChunk], max_chars_per_chunk: int = 1000) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        snippet = chunk.text[:max_chars_per_chunk]
        parts.append(f"[{i}] source={chunk.source} chunk={chunk.chunk}\n{snippet}")
    return "\n\n".join(parts)


def answer_with_llm(question: str, chunks: list[RetrievedChunk], settings: Settings) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")

    client = get_openai_client(settings)
    context = format_context(chunks)
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个中文问答助手。你必须优先依据检索片段回答。"
                    "如果片段能直接支持答案，请给出简洁答案并标注依据片段编号。"
                    "如果片段不能直接回答但可合理推断，请明确写“根据片段推测”。"
                    "只有在片段完全没有相关信息时，才回答“不知道”。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索片段：\n{context}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
