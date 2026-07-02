"""
Phase 1.1 — Document extraction and cleaning pipeline (pymupdf4llm backend).

Orchestrates extraction into a single pipeline per PDF:
  1. pymupdf4llm converts each page to markdown, handling column ordering,
     table linearisation, and figure-region detection via its internal ML
     layout model — replacing all hand-rolled coordinate heuristics.
  2. Per-page post-processing: unwrap picture-text blocks, strip running
     headers/footers, normalise symbol encoding, fix compound-word artifacts.
  3. Document-level: apply config-driven section exclusion.
  4. Quality gate before writing to data/cleaned/<stem>.txt.

Usage
-----
Process all PDFs in data/raw/ (default):
    python -m src.ingestion.extractor

Process specific PDF(s):
    python -m src.ingestion.extractor data/raw/my_guideline.pdf
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pymupdf4llm
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from gates.base import GateFailure
from gates.p1_1_extraction import run as run_extraction_gate
from src.ingestion.section_filter import filter_excluded_sections

logger = logging.getLogger(__name__)

# ── Filesystem layout ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "extraction.yaml"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


# ── Picture-text block processing ────────────────────────────────────────────


def _process_picture_text_blocks(md: str) -> str:
    """
    pymupdf4llm wraps figure/flowchart label text in HTML comment markers:

        <!-- Start of picture text -->
        label one<br>label two<br>...
        <!-- End of picture text -->

    This function converts those blocks to a labelled prose marker so the
    content (flowchart labels, diagram text) is preserved for RAG retrieval
    but clearly separated from body paragraphs.
    """
    def _replace(m: re.Match) -> str:
        content = m.group(1)
        content = re.sub(r"<br\s*/?>", "\n", content)
        content = content.strip()
        return f"[Figure content: {content}]"

    return re.sub(
        r"<!--\s*Start of picture text\s*-->(.*?)<!--\s*End of picture text\s*-->",
        _replace,
        md,
        flags=re.DOTALL,
    )


# ── Running-header / footer stripping ─────────────────────────────────────────


# Patterns for text that appears verbatim on every page of the KDIGO PDF
# (journal running title, author-line footer, journal citation footer).
# These are stripped as part of per-page post-processing so they never
# reach the section filter or the quality gate.
_RUNNING_HEADER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^KDIGO executive conclusions\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^J Floege et al\..*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^www\.kidney-international\.org\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Kidney International \(\d{4}\) \d+, \d+[–\-]\d+\s*$", re.MULTILINE),
    # Standalone journal-page numbers (e.g. "548", "549")
    re.compile(r"^\d{3,4}\s*$", re.MULTILINE),
]


def _strip_running_headers(md: str) -> str:
    for pat in _RUNNING_HEADER_PATTERNS:
        md = pat.sub("", md)
    return md


# ── Compound-word artifact repair ─────────────────────────────────────────────


# PDF soft-hyphens at line breaks sometimes produce run-together compound
# words in the pymupdf4llm output.  Fix the known occurrences for this
# document family; extend the list as new documents are ingested.
_COMPOUND_WORD_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\breninangiotensin\b", re.IGNORECASE), "renin-angiotensin"),
    (re.compile(r"\bsodiumglucose\b", re.IGNORECASE), "sodium-glucose"),
    (re.compile(r"\bplacebocontrolled\b", re.IGNORECASE), "placebo-controlled"),
]


def _fix_compound_words(text: str) -> str:
    for pat, replacement in _COMPOUND_WORD_FIXES:
        text = pat.sub(replacement, text)
    return text


# ── Per-page cleaning ─────────────────────────────────────────────────────────


def _clean_page_markdown(md: str) -> str:
    """Apply all per-page post-processing steps to one pymupdf4llm page chunk."""
    md = _process_picture_text_blocks(md)
    md = _strip_running_headers(md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ── Core document pipeline ────────────────────────────────────────────────────


def extract_document(
    pdf_path: Path,
    config: dict,
    openai_client: OpenAI | None = None,
) -> str:
    """
    Run the full extraction pipeline for a single PDF.

    Uses pymupdf4llm for base extraction (column ordering, table
    linearisation, figure-region separation via ML layout model) then
    applies document-level post-processing and section exclusion.

    The ``openai_client`` parameter is accepted for API compatibility with
    the batch runner but is not currently used; vision-LLM figure
    descriptions will be re-introduced as an optional enhancement once the
    base pipeline is stable.

    Returns:
        Cleaned text/markdown string ready for Phase 1.2 chunking.
    """
    logger.info("Extracting: %s", pdf_path.name)

    # Step 1 — per-page markdown via pymupdf4llm
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    total_pages = len(page_chunks)

    page_texts: list[str] = []
    for i, chunk in enumerate(page_chunks):
        logger.info("  Page %d / %d", i + 1, total_pages)
        page_md = _clean_page_markdown(chunk["text"])
        if page_md:
            page_texts.append(page_md)

    full_text = "\n\n".join(page_texts)

    # Step 2 — fix known PDF font-encoding substitutions for math symbols
    # Some PDFs encode ≥ as $ and ≤ as # in custom symbol fonts; fix these
    # only when immediately followed by a digit (safe for clinical content).
    full_text = re.sub(r"\$(\d)", r"≥\1", full_text)
    full_text = re.sub(r"#(\d)", r"≤\1", full_text)

    # Step 3 — fix compound-word PDF hyphenation artifacts
    full_text = _fix_compound_words(full_text)

    # Step 4 — drop noise sections
    excluded_headers = config["extraction"]["exclude_sections"]
    full_text = filter_excluded_sections(full_text, excluded_headers)

    logger.info(
        "Done. %d page(s), %d chars after cleaning.",
        total_pages, len(full_text),
    )
    return full_text


# ── Output ────────────────────────────────────────────────────────────────────


def write_cleaned_output(
    text: str,
    pdf_path: Path,
    output_dir: Path = CLEANED_DIR,
) -> Path:
    """Write cleaned text to data/cleaned/<pdf_stem>.txt; return the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Written → %s  (%d chars)", out_path.relative_to(PROJECT_ROOT), len(text))
    return out_path


# ── Batch runner ──────────────────────────────────────────────────────────────


def run_extraction_pipeline(pdf_paths: list[Path] | None = None) -> None:
    """
    Run the extraction pipeline for one or more PDFs.

    If ``pdf_paths`` is None, processes every *.pdf found in data/raw/.
    Skips paths that are not files or not PDFs.
    """
    load_dotenv()
    config = load_config()
    client = OpenAI()

    if pdf_paths is None:
        pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDFs found in %s", RAW_DIR)
            return

    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            logger.error("Not a file: %s — skipping", pdf_path)
            continue
        if pdf_path.suffix.lower() != ".pdf":
            logger.error("Not a PDF: %s — skipping", pdf_path)
            continue

        cleaned_text = extract_document(pdf_path, config, client)

        # Quality gate: validate the cleaned text before writing to disk.
        # If any hard-fail check fails, GateFailure is raised here and
        # the output file is never created.
        try:
            report = run_extraction_gate(cleaned_text, pdf_path.stem, config)
            report.raise_if_failed()
        except GateFailure as exc:
            logger.error("Gate rejected %s: %s", pdf_path.name, exc)
            continue

        write_cleaned_output(cleaned_text, pdf_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Extract and clean PDF documents (Phase 1.1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.ingestion.extractor\n"
            "  python -m src.ingestion.extractor data/raw/guideline.pdf"
        ),
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="PDF file(s) to process.  Defaults to all PDFs in data/raw/.",
    )
    args = parser.parse_args()

    run_extraction_pipeline(args.pdfs or None)


if __name__ == "__main__":
    main()
