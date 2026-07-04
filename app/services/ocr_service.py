import tempfile
from pathlib import Path

import fitz

from app.config import settings

_model = None
_tokenizer = None


def parse_pdf_bytes(pdf_bytes: bytes) -> dict:
    with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as tmp:
        pdf_path = Path(tmp) / "input.pdf"
        pdf_path.write_bytes(pdf_bytes)
        image_paths = _pdf_to_images(pdf_path, Path(tmp))
        text = _run_ocr(image_paths, Path(tmp))
        pages = [{"page_num": index + 1, "text": text} for index in range(len(image_paths))]
        if len(image_paths) > 1 and len(text) > 0:
            pages = [{"page_num": 1, "text": text}]
        return {"pages": pages, "full_text": text}


def _pdf_to_images(pdf_path: Path, output_dir: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72)
    paths: list[str] = []
    for index, page in enumerate(doc):
        out = output_dir / f"page_{index + 1:04d}.png"
        page.get_pixmap(matrix=matrix).save(out)
        paths.append(str(out))
    doc.close()
    return paths


def _run_ocr(image_paths: list[str], output_dir: Path) -> str:
    model, tokenizer = _load_model()
    import torch

    device = settings.ocr_device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    output_path = str(output_dir / "ocr_out")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    result = model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_paths,
        output_path=output_path,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )
    if isinstance(result, str) and result.strip():
        return result.strip()
    if result is not None and not isinstance(result, str):
        text = str(result).strip()
        if text:
            return text
    return _read_output_files(Path(output_path))


def _read_output_files(output_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".mmd"}:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if settings.ocr_device == "cuda" else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(
        settings.unlimited_ocr_model,
        trust_remote_code=True,
    )
    _model = AutoModel.from_pretrained(
        settings.unlimited_ocr_model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    _model = _model.eval()
    return _model, _tokenizer


def model_loaded() -> bool:
    return _model is not None
