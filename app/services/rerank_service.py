import logging

from app.config import settings
from app.schemas.rerank import RerankPassageInput, RerankRequest, RerankResponse, RerankResultItem

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder

    logger.info("loading rerank model %s", settings.rerank_model)
    _model = CrossEncoder(settings.rerank_model, max_length=512)
    return _model


def model_loaded() -> bool:
    return _model is not None


def rerank(request: RerankRequest) -> RerankResponse:
    passages = request.passages[: settings.rerank_max_passages]
    if not settings.rerank_enabled:
        top_k = min(request.top_k, len(passages))
        return RerankResponse(
            results=[
                RerankResultItem(id=passage.id, score=1.0 - index * 0.001, rank=index + 1)
                for index, passage in enumerate(passages[:top_k])
            ],
        )
    pairs = [(request.query, passage.text[: settings.rerank_passage_chars]) for passage in passages]
    model = _load_model()
    raw_scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(passages, raw_scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_k = min(request.top_k, len(ranked))
    results = [
        RerankResultItem(id=passage.id, score=float(score), rank=index + 1)
        for index, (passage, score) in enumerate(ranked[:top_k])
    ]
    return RerankResponse(results=results)
