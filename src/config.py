"""Configuration management"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """Application configuration"""

    # LLM Provider Configuration
    # Set to "openai" or "deepseek"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # DeepSeek Configuration (alternative)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    # Embedding Provider Configuration
    # Set to "openai" or "sentence-transformers"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

    # OpenAI Embeddings
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Sentence Transformers (alternative)
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DATA_DIR = Path(__file__).parent.parent / "data"
    PROTOCOLS_DIR = DATA_DIR / "protocols"
    REQUIREMENTS_DIR = DATA_DIR / "requirements"

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration based on selected providers"""
        if cls.LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                return False
        elif cls.LLM_PROVIDER == "deepseek":
            if not cls.DEEPSEEK_API_KEY:
                return False
        else:
            return False

        # Validate embedding provider
        if cls.EMBEDDING_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                return False

        return True

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration based on provider"""
        if cls.LLM_PROVIDER == "openai":
            return {
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_MODEL,
                "base_url": None,
            }
        elif cls.LLM_PROVIDER == "deepseek":
            return {
                "api_key": cls.DEEPSEEK_API_KEY,
                "model": cls.DEEPSEEK_MODEL,
                "base_url": cls.DEEPSEEK_BASE_URL,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration based on provider"""
        if cls.EMBEDDING_PROVIDER == "openai":
            return {
                "provider": "openai",
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_EMBEDDING_MODEL,
            }
        elif cls.EMBEDDING_PROVIDER == "sentence-transformers":
            return {
                "provider": "sentence-transformers",
                "model": cls.SENTENCE_TRANSFORMER_MODEL,
            }
        else:
            raise ValueError(f"Unknown embedding provider: {cls.EMBEDDING_PROVIDER}")
