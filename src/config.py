import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load variables from a local ".env" file into the process environment.
load_dotenv()


def _get_int(env_name: str, default: int) -> int:
    """Read an environment variable and safely convert it to int."""
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_name}='{raw_value}' is not a valid integer."
        )


def _get_list(env_name: str, default: List[str]) -> List[str]:
    """Read a comma-separated environment variable into a list of strings."""
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """A frozen (immutable) dataclass holding all project settings."""

    # --- LLM provider settings ---
    llm_provider: str          # "ollama" or "openai"
    ollama_model: str
    ollama_base_url: str
    openai_api_key: str
    openai_model: str

    # --- Embedding provider settings ---
    embedding_provider: str    # "huggingface" or "openai"
    hf_embedding_model: str

    # --- RAG tuning knobs ---
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # --- Storage paths ---
    chroma_persist_dir: str
    sqlite_db_path: str
    claims_data_dir: str

    # --- Bias testing ---
    bias_test_names: List[str] = field(default_factory=list)


def load_settings() -> Settings:
    """Reads all environment variables and returns a validated Settings object."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

    if llm_provider not in ("ollama", "openai"):
        raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'openai', got '{llm_provider}'")
    if embedding_provider not in ("huggingface", "openai"):
        raise ValueError(
            f"EMBEDDING_PROVIDER must be 'huggingface' or 'openai', got '{embedding_provider}'"
        )

    return Settings(
        llm_provider=llm_provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_provider=embedding_provider,
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 100),
        top_k=_get_int("TOP_K", 4),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", "./logs.db"),
        claims_data_dir=os.getenv("CLAIMS_DATA_DIR", "./data/claims"),
        bias_test_names=_get_list(
            "BIAS_TEST_NAMES",
            ["John Smith", "Maria Gonzalez", "Wei Chen", "Aaliyah Johnson"],
        ),
    )


# A single shared instance every other module imports.
settings = load_settings()


def get_llm():
    """
    Factory function that returns a LangChain chat model based on settings.
    Centralizing this means every other file just calls get_llm() and never
    needs to know whether we're using Ollama or OpenAI.
    """
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. Set it in .env."
            )
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )


def get_embeddings():
    """Factory function that returns a LangChain embeddings object based on settings."""
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    else:
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty. Set it in .env."
            )
        return OpenAIEmbeddings(api_key=settings.openai_api_key)


if __name__ == "__main__":
    print("Current configuration:")
    for field_name, value in settings.__dict__.items():
        print(f"  {field_name} = {value}")
