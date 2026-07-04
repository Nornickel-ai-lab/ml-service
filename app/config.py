from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    yandex_api_key: str = ""
    yandex_iam_token: str = ""
    yandex_folder_id: str = ""
    yandex_model: str = "yandexgpt-lite"
    yandex_embed_doc_model: str = "text-search-doc"
    yandex_embed_query_model: str = "text-search-query"
    embedding_dims: int = 256
    role_vector_weight: float = 0.15
    unlimited_ocr_model: str = "baidu/Unlimited-OCR"
    ocr_dpi: int = 300
    ocr_device: str = "cpu"
    ocr_image_mode: str = "base"
    mock_yandex: bool = False


settings = Settings()
