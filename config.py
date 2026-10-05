"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# --- LLM Provider: "openai" | "deepseek" ---
# Mặc định: openai nếu có OPENAI_API_KEY, ngược lại deepseek nếu có DEEPSEEK_API_KEY.
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER", "deepseek" if DEEPSEEK_API_KEY and not OPENAI_API_KEY else "openai"
).lower()

_PROVIDERS = {
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
}
if LLM_PROVIDER not in _PROVIDERS:
    raise ValueError(f"LLM_PROVIDER='{LLM_PROVIDER}' không hợp lệ, chọn: {list(_PROVIDERS)}")

LLM_API_KEY = _PROVIDERS[LLM_PROVIDER]["api_key"]
LLM_BASE_URL = _PROVIDERS[LLM_PROVIDER]["base_url"]
LLM_MODEL = _PROVIDERS[LLM_PROVIDER]["model"]


def get_llm_client():
    """OpenAI SDK client cho provider đang chọn (DeepSeek tương thích OpenAI API)."""
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def get_ragas_models():
    """(llm, embeddings) truyền vào ragas.evaluate().

    DeepSeek không có embeddings API → dùng OpenAI embeddings nếu có key,
    ngược lại dùng bge-m3 local (answer_relevancy cần embeddings).
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    if LLM_PROVIDER == "deepseek":
        llm = _single_completion(llm)
    if OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        embeddings = _shared_bge_m3_embeddings()
    return llm, embeddings


def _shared_bge_m3_embeddings():
    """LangChain Embeddings dùng chung bge-m3 (fp16) với DenseSearch — không load thêm bản thứ 2 lên GPU."""
    from langchain_core.embeddings import Embeddings
    from src.m2_search import get_encoder

    class SharedBgeM3Embeddings(Embeddings):
        def embed_documents(self, texts):
            return get_encoder().encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return get_encoder().encode(text, normalize_embeddings=True).tolist()

    return SharedBgeM3Embeddings()

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")


def _single_completion(chat_model):
    """Bọc chat model để RAGAS không gửi n>1 (DeepSeek chỉ hỗ trợ n=1).

    RAGAS gửi n completions trong 1 request nếu LLM là ChatOpenAI (answer_relevancy dùng n=3).
    Wrapper này không phải ChatOpenAI → RAGAS tự gửi n request riêng lẻ.
    """
    from langchain_core.language_models.chat_models import BaseChatModel

    class SingleCompletionChat(BaseChatModel):
        inner: BaseChatModel

        @property
        def _llm_type(self) -> str:
            return f"single-completion-{self.inner._llm_type}"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return SingleCompletionChat(inner=chat_model)
