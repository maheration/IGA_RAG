"""
Table detection, linearization, and column-aware text extraction.

Public API
----------
detect_column_boundary(page) -> float | None
    Returns the x-coordinate of the gutter between two text columns, or None.

get_page_content_blocks(
    plumber_page,
    fitz_page,
    figure_bboxes=None,
    header_strip_top=0.0,
    max_header_cell_len=80,
    dedup_min_len=20,
) -> list[tuple[float, str]]
    Returns (top_y, text) pairs for body text and linearized tables, in
    correct top-to-bottom reading order, with figure and table regions
    excluded from body text.

Design notes — body text extraction
--------------------------------------
Body text is extracted using pymupdf (fitz) word-level extraction:
    fitz_page.get_text("words") → list of (x0, y0, x1, y1, word, ...)

pymupdf derives word boundaries from the font program's advance widths and the
kerning/glyph-positioning data embedded in the PDF.  Inter-character gaps are
compared to the font's own advance-width metric, so a word break is inserted
whenever the gap exceeds ~0.3 × char_width.  This is far more aggressive (and
more accurate) than pdfminer's default char_margin=2.0 and reliably separates
consecutive words stored as a single glyph stream — the root cause of
word-glue artefacts like "Thegoaloftreatmentisto...".

Crucially, this approach reconstructs text from a word list (filtered by column
x-range and exclude-bbox, then sorted by (y, x)), not from a fixed-width
character grid.  There is no grid truncation at crop boundaries, so the
regression introduced by pdfminer LAParams is avoided entirely.

Design notes — table linearization
-----------------------------------
Tables are detected with pdfplumber (superior table-structure detection).
pdfplumber forward-fills merged cell content across all rows that share the
merge.  We detect this with whitespace-normalized comparison: if a cell's
normalized value matches the previous row's normalized cell value AND the raw
string is longer than dedup_min_len characters, we suppress it as a
forward-fill artifact.

Header-row detection:
The linearizer promotes the first non-empty row to column headers ONLY if every
cell is at most max_header_cell_len characters long.  Longer first rows are
actual data; auto-generated labels ("Col 1", "Col 2", …) are used instead and
the row is kept in the data output.
"""

from __future__ import annotations

from collections import Counter

import fitz  # pymupdf
import pdfplumber

Bbox = tuple[float, float, float, float]  # (x0, top, x1, bottom)

_DEFAULT_DEDUP_MIN_LEN = 20
_DEFAULT_MAX_HEADER_CELL_LEN = 80
_DEFAULT_LINE_Y_TOL = 4.0    # words within this many pts of y share a line


# ── Public API ────────────────────────────────────────────────────────────────


def detect_column_boundary(page: pdfplumber.page.Page) -> float | None:
    """
    Return the x-coordinate of the column gutter if the page is two-column,
    or None for single-column pages.

    Algorithm: find the dominant word-start x-position in the right half of
    the page — in a two-column layout this corresponds to the right column's
    left margin, and the boundary is set just to its left.
    """
    words = page.extract_words()
    if len(words) < 20:
        return None

    pw = page.width
    right_x0s = [w["x0"] for w in words if w["x0"] > pw * 0.45]

    if len(right_x0s) < 10:
        return None

    bins: Counter[int] = Counter(int(x // 10) * 10 for x in right_x0s)
    top_count = max(bins.values())

    if top_count < 15:
        return None

    left_count = sum(1 for w in words if w["x0"] < pw * 0.45)
    if left_count < 15:
        return None

    top_bins = [x for x, c in bins.items() if c >= top_count * 0.70]
    right_col_margin = float(min(top_bins))

    boundary = right_col_margin - 5.0
    if not (pw * 0.40 <= boundary <= pw * 0.65):
        return None

    return boundary


def get_page_content_blocks(
    plumber_page: pdfplumber.page.Page,
    fitz_page: fitz.Page,
    figure_bboxes: list[Bbox] | None = None,
    header_strip_top: float = 0.0,
    max_header_cell_len: int = _DEFAULT_MAX_HEADER_CELL_LEN,
    dedup_min_len: int = _DEFAULT_DEDUP_MIN_LEN,
) -> list[tuple[float, str]]:
    """
    Extract all text content from a page as (top_y, text) blocks.

    Body text is extracted via pymupdf word-level extraction (see module
    docstring).  Tables are detected and linearized via pdfplumber.

    Steps:
    1. Find tables (pdfplumber); collect their bboxes for exclusion.
    2. Build the full exclude-bbox list (tables + figures).
    3. Detect two-column layout.
    4. For each column, collect fitz words inside the column, outside all
       exclude bboxes, sort by (y, x), group into lines, join with spaces.
    5. Offset right-column block by page.height so it sorts after left column.
    6. Append linearized table blocks at their natural y-positions.

    Args:
        plumber_page:
            pdfplumber Page object (used for table detection and column
            boundary detection only).
        fitz_page:
            pymupdf Page object (used for word-level body text extraction).
        figure_bboxes:
            Padded figure bboxes from detect_figure_regions().  Words inside
            these regions are excluded from body-text blocks.
        header_strip_top:
            Words whose top-edge y is below this value are treated as running
            headers and stripped.
        max_header_cell_len:
            Max cell length for a table row to be treated as column headers.
        dedup_min_len:
            Min length for a repeated cell to be suppressed as forward-fill.
    """
    tables = plumber_page.find_tables()
    table_bboxes: list[Bbox] = [t.bbox for t in tables]
    exclude_bboxes: list[Bbox] = table_bboxes + (figure_bboxes or [])

    col_boundary = detect_column_boundary(plumber_page)
    blocks: list[tuple[float, str]] = []

    if col_boundary is None:
        text = _extract_body_text_fitz(
            fitz_page,
            col_x0=0.0, col_x1=plumber_page.width,
            y_min=header_strip_top, y_max=plumber_page.height,
            exclude_bboxes=exclude_bboxes,
        )
        if text.strip():
            blocks.append((0.0, text.strip()))
    else:
        left_text = _extract_body_text_fitz(
            fitz_page,
            col_x0=0.0, col_x1=col_boundary,
            y_min=header_strip_top, y_max=plumber_page.height,
            exclude_bboxes=exclude_bboxes,
        )
        if left_text.strip():
            blocks.append((0.0, left_text.strip()))

        right_text = _extract_body_text_fitz(
            fitz_page,
            col_x0=col_boundary, col_x1=plumber_page.width,
            y_min=header_strip_top, y_max=plumber_page.height,
            exclude_bboxes=exclude_bboxes,
        )
        if right_text.strip():
            # Offset so right-column content always sorts after left-column.
            blocks.append((plumber_page.height, right_text.strip()))

    # Linearized table blocks at their natural y-position.
    # fitz_page is passed so cell text is re-extracted via fitz word extraction,
    # eliminating word-glue artefacts that come from pdfminer's table.extract().
    for table in sorted(tables, key=lambda t: t.bbox[1]):
        linearized = _linearize_table(
            table, max_header_cell_len, dedup_min_len, fitz_page=fitz_page
        )
        if linearized:
            blocks.append((table.bbox[1], linearized))

    return sorted(blocks, key=lambda b: b[0])


# ── fitz-based body text extraction ──────────────────────────────────────────


def _extract_body_text_fitz(
    fitz_page: fitz.Page,
    col_x0: float,
    col_x1: float,
    y_min: float,
    y_max: float,
    exclude_bboxes: list[Bbox],
    line_y_tol: float = _DEFAULT_LINE_Y_TOL,
) -> str:
    """
    Extract body text from a page region using pymupdf word-level extraction.

    pymupdf uses the font program's advance widths to determine word
    boundaries, which reliably separates words stored as dense glyph streams.
    Text is reconstructed from the word list (not a layout grid), so there is
    no right-edge truncation at crop boundaries.

    Args:
        fitz_page:    pymupdf Page object for the whole page.
        col_x0/x1:   Horizontal bounds of the column to extract.
        y_min/y_max:  Vertical bounds (header strip to page bottom).
        exclude_bboxes: (x0, top, x1, bottom) regions to skip (figures, tables).
        line_y_tol:   Words within this many pts in y are on the same line.
    """
    raw_words = fitz_page.get_text("words")
    # Each entry: (x0, y0, x1, y1, word_text, block_no, line_no, word_no)

    filtered: list[tuple[float, float, str]] = []
    for entry in raw_words:
        wx0, wy0, wx1, wy1, word = entry[0], entry[1], entry[2], entry[3], entry[4]

        # Column filter: word must START within this column (±2 pt tolerance)
        if wx0 < col_x0 - 2 or wx0 >= col_x1:
            continue
        # Y-range filter
        if wy1 <= y_min or wy0 >= y_max:
            continue
        # Exclude-bbox filter: skip words that overlap any excluded region
        excluded = False
        for ex0, ey0, ex1, ey1 in exclude_bboxes:
            if wx0 < ex1 and wx1 > ex0 and wy0 < ey1 and wy1 > ey0:
                excluded = True
                break
        if excluded:
            continue

        filtered.append((wy0, wx0, word))

    if not filtered:
        return ""

    # Sort top-to-bottom, then left-to-right within each line
    filtered.sort()

    # Group words with similar y-positions into the same line
    lines: list[str] = []
    current_line_words: list[tuple[float, str]] = []
    current_y: float | None = None

    for y, x, word in filtered:
        if current_y is None or y - current_y > line_y_tol:
            if current_line_words:
                lines.append(
                    " ".join(w for _, w in sorted(current_line_words))
                )
            current_line_words = [(x, word)]
            current_y = y
        else:
            current_line_words.append((x, word))

    if current_line_words:
        lines.append(" ".join(w for _, w in sorted(current_line_words)))

    return "\n".join(lines)


# ── Table linearization ───────────────────────────────────────────────────────


def _normalize_cell(text: str) -> str:
    """Collapse all whitespace runs to single spaces for robust comparison."""
    return " ".join(text.split())


def _is_header_row(row: list[str], max_header_cell_len: int) -> bool:
    """
    Return True if this row looks like genuine column labels.

    A real header row has short, simple strings ("Question", "Dose", "Notes").
    A data row has long strings — questions, sentences, clinical text.
    If any cell exceeds max_header_cell_len characters, the row is data.
    """
    return all(len(cell) <= max_header_cell_len for cell in row)


def _linearize_table(
    table: pdfplumber.table.Table,
    max_header_cell_len: int = _DEFAULT_MAX_HEADER_CELL_LEN,
    dedup_min_len: int = _DEFAULT_DEDUP_MIN_LEN,
    fitz_page: fitz.Page | None = None,
) -> str:
    """
    Serialize a pdfplumber Table into a self-contained string.

    Format — each data row becomes one line:
        Drug: Nefecon | Dose: 16 mg/day | Duration: 9 months

    Header-row detection:
        The first non-empty row is used as column headers ONLY if every cell is
        at most max_header_cell_len characters long.  If any cell is longer, the
        row is actual data; auto-generated labels ("Col 1", "Col 2", …) are used
        instead and the row is kept in the data output.

    Forward-fill deduplication:
        pdfplumber repeats the content of merged cells in every row they span.
        We detect this with whitespace-normalized comparison: if a cell's
        normalized value matches the previous row's normalized cell value AND
        the raw string is longer than dedup_min_len characters, we suppress it.

    fitz_page (optional):
        When provided, each cell's text is re-extracted by cropping the fitz
        page to the cell's bbox and running fitz word-level extraction.  This
        applies the same word-spacing logic used for body text, eliminating
        word-glue artefacts in table cells (e.g. "Age:Inthetrials..." →
        "Age: In the trials...").  Falls back to the pdfplumber string when
        the cell bbox is unavailable or fitz returns empty text.
    """
    rows = table.extract()
    if not rows:
        return ""

    # Build (row_idx, col_idx) → cell bbox lookup from pdfplumber Row objects.
    # table.rows[i].cells[j] is (x0, top, x1, bottom) or None for missing cells.
    cell_bboxes: list[list[Bbox | None]] = []
    if fitz_page is not None:
        try:
            for plumb_row in table.rows:
                row_cbs: list[Bbox | None] = []
                for cb in plumb_row.cells:
                    if cb is not None and len(cb) == 4 and cb[2] > cb[0] and cb[3] > cb[1]:
                        row_cbs.append(
                            (float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))
                        )
                    else:
                        row_cbs.append(None)
                cell_bboxes.append(row_cbs)
        except (AttributeError, TypeError):
            # Unexpected pdfplumber internal structure — fall back gracefully.
            cell_bboxes = []

    def _get_cell_text(row_idx: int, col_idx: int, raw: str) -> str:
        """Return fitz-extracted cell text, or raw pdfplumber text as fallback."""
        if not cell_bboxes or fitz_page is None:
            return raw
        if row_idx >= len(cell_bboxes):
            return raw
        row_cbs = cell_bboxes[row_idx]
        if col_idx >= len(row_cbs) or row_cbs[col_idx] is None:
            return raw
        bbox = row_cbs[col_idx]
        fitz_text = _extract_body_text_fitz(
            fitz_page,
            col_x0=bbox[0],
            col_x1=bbox[2],
            y_min=bbox[1],
            y_max=bbox[3],
            exclude_bboxes=[],
        ).strip()
        return fitz_text if fitz_text else raw

    # Find first non-empty row
    first_nonempty_idx = None
    for i, row in enumerate(rows):
        if any(cell for cell in row if cell):
            first_nonempty_idx = i
            break

    if first_nonempty_idx is None:
        return ""

    first_row_cells = [
        _get_cell_text(first_nonempty_idx, j, str(cell).strip() if cell else "")
        for j, cell in enumerate(rows[first_nonempty_idx])
    ]

    if _is_header_row(first_row_cells, max_header_cell_len):
        # First row is genuine column labels — use as headers, skip in data.
        headers = first_row_cells
        data_start = first_nonempty_idx + 1
    else:
        # First row is data — generate labels, include this row in data output.
        headers = [f"Col {i + 1}" for i in range(len(first_row_cells))]
        data_start = first_nonempty_idx

    if not headers:
        return ""

    lines = ["[TABLE]"]
    prev_norm: list[str] = [""] * len(headers)

    for row_idx, row in enumerate(rows):
        if row_idx < data_start:
            continue
        if not any(cell for cell in row if cell):
            continue

        cells = [
            _get_cell_text(row_idx, col_idx, str(cell).strip() if cell else "")
            for col_idx, cell in enumerate(row)
        ]

        parts: list[str] = []
        for h, c, prev_n in zip(headers, cells, prev_norm):
            if not (h or c):
                continue
            # Suppress forward-filled cells: normalized comparison makes this
            # robust to minor whitespace differences across merged-cell rows.
            c_norm = _normalize_cell(c)
            if c and c_norm == prev_n and len(c) >= dedup_min_len:
                continue
            parts.append(f"{h}: {c}")

        if parts:
            lines.append(" | ".join(parts))

        prev_norm = [_normalize_cell(c) for c in cells]

    return "\n".join(lines) if len(lines) > 1 else ""
