"""Docling conversion for the business ingestion path (D61: multi-format).

Quality contract (see docs/Docling解析开关与注意点.md):
  table ACCURATE, formula/code enrichment, scale=3.0, picture export REFERENCED.
OCR: quality baseline is ON; runtime may set DOCLING_DO_OCR=false when container
RapidOCR has no GPU (CPU path is extremely slow on born-digital papers).

D61: the converter now registers the accepted office/text formats next to PDF.
``PdfPipelineOptions`` only configures the PDF pipeline -- Docling's office/text
backends are deterministic structure extraction without an OCR/table/layout stage,
so the enrichment knobs above apply to PDF *only*.  ``PARSER_OPTIONS`` records the
per-document truth (``pipeline`` is ``pdf`` or ``backend``) so an artifact can
never claim knobs that were not applied to it.
"""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

# D61: file_type (extension, with dot) -> Docling InputFormat.  Mirrors the
# registry whitelist (src/storage/document_registry.py); the two must not drift,
# which is asserted in tests/test_registry_multi_format.py.
FORMAT_BY_EXTENSION: dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".xlsx": InputFormat.XLSX,
    ".md": InputFormat.MD,
    ".html": InputFormat.HTML,
    ".htm": InputFormat.HTML,
    ".csv": InputFormat.CSV,
}


def _pdf_pipeline_options(*, do_ocr: bool) -> PdfPipelineOptions:
    opts = PdfPipelineOptions()
    opts.do_ocr = bool(do_ocr)
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.do_formula_enrichment = True
    opts.do_code_enrichment = True
    opts.generate_picture_images = True
    # Mutating scale in place avoids a pydantic circular-ref when hashing options.
    opts.code_formula_options.scale = 3.0
    return opts


def build_converter(*, do_ocr: bool | None = None) -> DocumentConverter:
    """Build the Docling converter for every accepted format.

    Born-digital academic PDFs should keep OCR off: container RapidOCR is CPU-only
    without onnxruntime-gpu and dominates wall time.  Office/text formats are not
    affected by this switch -- they have no OCR stage.
    """

    import os

    if do_ocr is None:
        do_ocr = os.getenv("DOCLING_DO_OCR", "true").lower() in {"1", "true", "yes"}

    opts = _pdf_pipeline_options(do_ocr=do_ocr)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_document(
    source_path: Path | str,
    *,
    file_type: str,
    artifacts_dir: Path,
    do_ocr: bool | None = None,
) -> DoclingDocument:
    """Convert one document and materialize referenced pictures under artifacts_dir.

    ``file_type`` selects the Docling backend explicitly (the source path keeps
    its own extension, which may be a transcoded UTF-8 copy).  Returns the
    re-loaded document, so picture references resolve against the on-disk
    artifacts exactly as the JSON round-trip downstream does.
    """

    import os

    source_path = Path(source_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fmt = FORMAT_BY_EXTENSION.get((file_type or "").lower())
    if fmt is None:
        raise ValueError(f"unsupported file_type: {file_type!r}")

    if do_ocr is None:
        do_ocr = os.getenv("DOCLING_DO_OCR", "true").lower() in {"1", "true", "yes"}
    PARSER_OPTIONS["do_ocr"] = bool(do_ocr) if fmt is InputFormat.PDF else False
    PARSER_OPTIONS["pipeline"] = "pdf" if fmt is InputFormat.PDF else "backend"

    converter = build_converter(do_ocr=do_ocr)
    result = converter.convert(source_path)
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
    # D61: which pipeline these knobs were applied to.  ``pdf`` = the full
    # layout/table/OCR pipeline above; ``backend`` = the deterministic office/
    # text backend, where table/formula/OCR options are *not* in play.
    "pipeline": "pdf",
}
