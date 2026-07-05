import re
import tempfile
from pathlib import Path

import fitz

from app.config import settings

_model = None
_tokenizer = None


def parse_pdf_bytes(pdf_bytes: bytes) -> dict:
    if not settings.ocr_enabled:
        raise RuntimeError("OCR disabled (OCR_ENABLED=false). Use text PDFs or enable OCR in .env.")
    with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "input.pdf"
        pdf_path.write_bytes(pdf_bytes)
        image_paths = _pdf_to_images(pdf_path, tmp_path)
        output_dir = tmp_path / "ocr_out"
        text = _run_ocr(image_paths, output_dir)
        page_texts = _page_texts_from_output(output_dir, len(image_paths), text)
        pages = [{"page_num": index + 1, "text": page_text} for index, page_text in enumerate(page_texts)]
        full_text = "\n\n".join(part.strip() for part in page_texts if part.strip())
        return {"pages": pages, "full_text": full_text or text}


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


def _page_texts_from_output(output_dir: Path, page_count: int, fallback_text: str) -> list[str]:
    per_page: list[str] = []
    for index in range(page_count):
        page_no = index + 1
        candidates = sorted(output_dir.glob(f"*page_{page_no:04d}*"))
        candidates.extend(sorted(output_dir.glob(f"*{page_no}*.txt")))
        candidates.extend(sorted(output_dir.glob(f"*{page_no}*.md")))
        candidates.extend(sorted(output_dir.glob(f"*{page_no}*.mmd")))
        text = ""
        for candidate in candidates:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    break
        per_page.append(text)
    if any(part.strip() for part in per_page):
        return per_page
    if fallback_text.strip():
        return _split_text_across_pages(fallback_text, page_count)
    return [""] * page_count


def _split_text_across_pages(text: str, page_count: int) -> list[str]:
    if page_count <= 1:
        return [text.strip()]
    blocks = [block.strip() for block in re.split(r"\n{2,}", text.strip()) if block.strip()]
    if len(blocks) >= page_count:
        chunk_size = max(1, len(blocks) // page_count)
        pages: list[str] = []
        cursor = 0
        for index in range(page_count):
            if index == page_count - 1:
                pages.append("\n\n".join(blocks[cursor:]))
            else:
                pages.append("\n\n".join(blocks[cursor : cursor + chunk_size]))
                cursor += chunk_size
        return pages
    return [text.strip()] + [""] * (page_count - 1)


def _run_ocr(image_paths: list[str], output_dir: Path) -> str:
    model, tokenizer = _load_model()
    import torch

    device = settings.ocr_device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir)
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
    return _read_output_files(output_dir)


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
