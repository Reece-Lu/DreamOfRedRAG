import argparse

from .config import get_settings
from .core import answer_with_llm, ingest_corpus, retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="建立或重建向量库")

    ask_parser = subparsers.add_parser("ask", help="提问并生成答案")
    ask_parser.add_argument("question", nargs="?", default="")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--candidate-k", type=int, default=20, help="先召回再重排的候选数量")

    search_parser = subparsers.add_parser("search", help="只检索，不调用大模型")
    search_parser.add_argument("question", nargs="?", default="")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--candidate-k", type=int, default=20, help="先召回再重排的候选数量")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "ingest":
        count = ingest_corpus(settings)
        print(f"已写入 {count} 个分块到 ChromaDB")
        return

    question = getattr(args, "question", "").strip() or input("请输入问题：").strip()
    chunks = retrieve(
        question,
        settings,
        top_k=getattr(args, "top_k", 3),
        candidate_k=getattr(args, "candidate_k", None),
    )

    print("\n检索到的相关片段：")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] chunk={chunk.chunk} source={chunk.source}")
        print(
            "score="
            f"{(chunk.final_score or 0.0):.4f} "
            f"(semantic={(chunk.semantic_score or 0.0):.4f}, keyword={(chunk.keyword_score or 0.0):.4f})"
        )
        print(chunk.text[:400])

    if args.command == "search":
        return

    print("\n大模型回答：")
    try:
        print(answer_with_llm(question, chunks, settings))
    except RuntimeError as exc:
        print(f"{exc}")
        print("当前只完成了检索，没有调用大模型。请先设置 OPENAI_API_KEY。")
