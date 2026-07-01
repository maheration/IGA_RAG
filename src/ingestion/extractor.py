"""
Phase 1.1 — Document extraction and cleaning pipeline.

Orchestrates the three sub-steps into a single pipeline per PDF:
  1. Per-page: extract body text as (y, text) blocks with tables linearized inline.
  2. Per-page: detect figure regions and call vision LLM for prose descriptions.
  3. Per-page: merge all blocks by y-position → correct top-to-bottom ordering.
  4. Document-level: apply config-driven section exclusion.
  5. Write cleaned text to data/cleaned/<stem>.txt

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

import fitz  # pymupdf
import pdfplumber
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from gates.base import GateFailure
from gates.p1_1_extraction import run as run_extraction_gate
from src.ingestion.figure_handler import FigureDescription, describe_page_figures, detect_figure_regions
from src.ingestion.section_filter import filter_excluded_sections
from src.ingestion.table_handler import detect_column_boundary, get_page_content_blocks

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


# ── Per-page assembly ─────────────────────────────────────────────────────────


def _assemble_page_text(
    text_blocks: list[tuple[float, str]],
    figure_descriptions: list[FigureDescription],
    col_boundary: float | None,
    page_height: float,
) -> str:
    """
    Merge body-text/table blocks and figure descriptions into a single
    page string, sorted by their adjusted y-coordinate.

    For two-column pages, figure descriptions are assigned to their
    column by their horizontal center: left-column figures keep their
    original top_y; right-column figures get top_y + page_height so they
    sort after all left-column content.
    """
    all_items: list[tuple[float, str]] = list(text_blocks)

    for fig in figure_descriptions:
        if col_boundary is not None:
            fig_center_x = (fig.x0 + fig.x1) / 2
            y_adj = fig.top_y if fig_center_x < col_boundary else page_height + fig.top_y
        else:
            y_adj = fig.top_y
        all_items.append((y_adj, fig.description))

    all_items.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in all_items if text.strip())


# ── Core document pipeline ────────────────────────────────────────────────────


def extract_document(pdf_path: Path, config: dict, openai_client: OpenAI) -> str:
    """
    Run the full extraction pipeline for a single PDF.

    Opens the file once with pdfplumber (structural analysis) and once with
    pymupdf (page rendering for vision LLM) to avoid reopening per page.

    Returns:
        Cleaned text string ready for Phase 1.2 chunking.
    """
    logger.info("Extracting: %s", pdf_path.name)

    page_texts: list[str] = []
    figure_counter = [0]  # mutable int shared across pages

    with pdfplumber.open(pdf_path) as plumber_doc, fitz.open(str(pdf_path)) as fitz_doc:
        total_pages = len(plumber_doc.pages)

        for page_index, plumber_page in enumerate(plumber_doc.pages):
            logger.info("  Page %d / %d", page_index + 1, total_pages)
            fitz_page = fitz_doc[page_index]

            # Step 1 — detect figure regions first (no LLM yet)
            # This must happen before text extraction so figure bboxes can be
            # excluded from body-text word collection, preventing flowchart
            # labels from leaking into the extracted prose.
            fig_regions = detect_figure_regions(plumber_page, config)

            # Pad figure bboxes to capture overhanging labels and captions
            # (text that sits just outside the graphical boundary in the PDF).
            pad = config["figure_detection"].get("bbox_padding_pts", 0)
            padded_fig_regions = [
                (x0, max(0, top - pad), x1, min(plumber_page.height, bottom + pad))
                for x0, top, x1, bottom in fig_regions
            ]

            # Step 2 — body text + linearized tables as (y, text) blocks,
            # with figure regions excluded from word collection.
            header_strip_top = config["extraction"].get("running_header_top_pts", 0)
            col_boundary = detect_column_boundary(plumber_page)
            text_blocks = get_page_content_blocks(
                plumber_page,
                figure_bboxes=padded_fig_regions,
                header_strip_top=header_strip_top,
            )

            # Step 3 — call vision LLM for each detected figure region
            figure_descriptions = describe_page_figures(
                pdfplumber_page=plumber_page,
                fitz_page=fitz_page,
                config=config,
                openai_client=openai_client,
                figure_counter=figure_counter,
                predetected_regions=fig_regions,
            )

            # Step 4 — merge all blocks into correct vertical order (column-aware)
            page_text = _assemble_page_text(
                text_blocks, figure_descriptions,
                col_boundary, plumber_page.height,
            )
            page_texts.append(f"[Page {page_index + 1}]\n{page_text}")

    # Step 4 — concatenate all pages
    full_text = "\n\n".join(page_texts)

    # Step 5 — drop noise sections
    excluded_headers = config["extraction"]["exclude_sections"]
    cleaned = filter_excluded_sections(full_text, excluded_headers)

    # Step 6 — fix known PDF font-encoding substitutions for math symbols
    # Some PDFs encode ≥ as $ and ≤ as # in custom symbol fonts; fix these
    # only when immediately followed by a digit (safe for clinical content).
    cleaned = re.sub(r"\$(\d)", r"≥\1", cleaned)
    cleaned = re.sub(r"#(\d)", r"≤\1", cleaned)

    logger.info(
        "Done. %d page(s), %d figure(s), %d chars after cleaning.",
        total_pages, figure_counter[0], len(cleaned),
    )
    return cleaned


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
