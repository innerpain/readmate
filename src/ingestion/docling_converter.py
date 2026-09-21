"""Docling PDF conversion for the business ingestion path.

Quality contract (see docs/Docling解析开关与注意点.md):
  table ACCURATE, formula/code enrichment, scale=3.0, picture export REFERENCED.
OCR: quality baseline is ON; runtime may set DOCLING_DO_OCR=false when container
RapidOCR has no GPU (CPU path is extremely slow on born-digital papers).
"""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode


def build_converter(*, do_ocr: bool | None = None) -> DocumentConverter:
    """Build Docling converter.

    Born-digital academic PDFs should keep OCR off: container RapidOCR is CPU-only
    without onnxruntime-gpu and dominates wall time.
    """

    import os

    if do_ocr is None:
        do_ocr = os.getenv("DOCLING_DO_OCR", "true").lower() in {"1", "true", "yes"}

    opts = PdfPipelineOptions()
    opts.do_ocr = bool(do_ocr)
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.do_formula_enrichment = True
    opts.do_code_enrichment = True
    opts.generate_picture_images = True
    # Mutating scale in place avoids a pydantic circular-ref when hashing options.
    opts.code_formula_options.scale = 3.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_pdf(pdf_path: Path | str, *, artifacts_dir: Path, do_ocr: bool | None = None) -> DoclingDocument:
    """Convert one PDF and materialize referenced picture files under artifacts_dir."""

    import os

    pdf_path = Path(pdf_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    converter = build_converter(do_ocr=do_ocr)
    if do_ocr is None:
        do_ocr = os.getenv("DOCLING_DO_OCR", "true").lower() in {"1", "true", "yes"}
    PARSER_OPTIONS["do_ocr"] = bool(do_ocr)
    result = converter.convert(str(pdf_path))
    doc = result.document
    doc.save_as_json(
        artifacts_dir / "docling.json",
        image_mode=ImageRefMode.REFERENCED,
    )
    saved = DoclingDocument.model_validate_json(
        (artifacts_dir / "docling.json").read_text(encoding="utf-8")
    )
    return saved


PARSER_OPTIONS = {
    # Runtime OCR is env-controlled; recorded value should reflect the run.
    "do_ocr": None,  # filled at convert time from env / build_converter
    "do_table_structure": True,
    "table_mode": "accurate",
    "do_formula_enrichment": True,
    "do_code_enrichment": True,
    "generate_picture_images": True,
    "image_export_mode": "referenced",
    "formula_scale": 3.0,
}
