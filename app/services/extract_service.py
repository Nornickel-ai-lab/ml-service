from app.schemas.extract import ExtractRequest, ExtractResponse
from app.services import mock_extract


def extract(request: ExtractRequest) -> ExtractResponse:
    return mock_extract.extract_mock(
        text=request.text,
        document_id=request.document_id,
        title=request.title,
    )
