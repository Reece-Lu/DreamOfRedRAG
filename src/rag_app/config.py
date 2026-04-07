from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    text_file: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    openai_api_key: str | None


def get_settings() -> Settings:
    text_file = Path(os.getenv("RAG_TEXT_FILE", BASE_DIR / "红楼梦.txt"))
    chroma_dir = Path(os.getenv("RAG_CHROMA_DIR", BASE_DIR / "data" / "chroma_db"))
    return Settings(
        text_file=text_file,
        chroma_dir=chroma_dir,
        collection_name=os.getenv("RAG_COLLECTION_NAME", "hongloumeng"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
