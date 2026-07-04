from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    yandex_api_key: str = ""
    yandex_iam_token: str = ""
    yandex_folder_id: str = ""
    yandex_model: str = "yandexgpt-lite"
    yandex_embed_doc_model: str = "text-embeddings-doc-v2"
    yandex_embed_query_model: str = "text-search-query"
    embedding_dims: int = 256
    role_vector_weight: float = 0.15
    unlimited_ocr_model: str = "baidu/Unlimited-OCR"
    ocr_dpi: int = 300
    ocr_device: str = "cpu"
    ocr_image_mode: str = "base"
    mock_yandex: bool = False
    default_ml_provider: str = "cloud"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_embed_model: str = "snowflake-arctic-embed2:latest"
    ollama_embedding_dims: int = 1024
    ollama_max_passages: int = 3
    ollama_passage_chars: int = 480
    ollama_num_predict: int = 512
    ollama_num_ctx: int = 4096


settings = Settings()
