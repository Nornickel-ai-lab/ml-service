from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gigachat_credentials: str = ""
    gigachat_client_id: str = ""
    gigachat_client_secret: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_api_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_llm_model: str = "GigaChat-2-Pro"
    gigachat_embed_model: str = "EmbeddingsGigaR"
    gigachat_embedding_dims: int = 2560
    gigachat_verify_ssl: bool = False
    unlimited_ocr_model: str = "baidu/Unlimited-OCR"
    ocr_dpi: int = 300
    ocr_device: str = "cpu"
    ocr_image_mode: str = "base"
    default_ml_provider: str = "gigachat"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_embed_model: str = "snowflake-arctic-embed2:latest"
    ollama_embedding_dims: int = 1024
    ollama_max_passages: int = 3
    ollama_passage_chars: int = 480
    ollama_num_predict: int = 512
    ollama_num_ctx: int = 4096
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_passages: int = 32
    rerank_passage_chars: int = 1500
    rerank_enabled: bool = False
    ocr_enabled: bool = True
    ollama_enabled: bool = True
    query_parse_llm: bool = False
    gigachat_max_passages: int = 4
    weak_relevance_ratio: float = 0.28
    weak_es_score_min: float = 0.35


settings = Settings()
