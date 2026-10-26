from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import SecretStr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):

    groq_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")

    groq_model: str = "openai/gpt-oss-20b"

    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768

    raw_data_dir: Path =  "data/raw"
    processed_data_dir: Path =  "data/processed"
    chroma_persist_dir: Path =  "data/indexes/chroma"

    pinecone_api_key: SecretStr = SecretStr("pcsk_2fVUAc_XUAykGrHDY5GvLrVAcVSpi3vpvwDTzvHGJuwjaZaj4Ee5Ai64pXsLGT3VerMFq")
    pinecone_index_name: str = "emplay-assignment"
    pinecone_namespace: str = "rfp-bids"

    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()