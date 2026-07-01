"""
Figure detection and vision-LLM description for PDF pages.

Public API
----------
describe_page_figures(pdfplumber_page, fitz_page, config, openai_client, figure_counter)
    -> list[FigureDescription]

Detects figure regions by:
  1. Collecting embedded raster images (page.images)
  2. Clustering qualifying vector rectangles (flowcharts, diagrams)

Each detected region is rendered to PNG via pymupdf and described by a
vision-capable LLM.  Descriptions are returned with their top_y coordinate
so the caller can merge them with body text blocks in vertical order.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import fitz  # pymupdf
import pdfplumber
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class FigureDescription:
    """Vision-LLM description of one detected figure region."""
    x0: float          # left edge of the figure (pts)
    x1: float          # right edge of the figure (pts)
    top_y: float       # top edge of the figure (pts, from page top)
    bottom_y: float    # bottom edge of the figure
    label: str         # e.g. "Figure 1", auto-assigned sequentially per document
    description: str   # formatted prose description ready to insert into text


def detect_figure_regions(
    page: pdfplumber.page.Page,
    config: dict,
) -> list[tuple[float, float, float, float]]:
    """
    Return (x0, top, x1, bottom) bounding boxes for all figure regions on a page.

    Public entry point for callers that need the bboxes before the vision LLM
    is called (e.g. to exclude figure areas from body-text extraction).
    """
    return _detect_figure_regions(page, config["figure_detection"])


def describe_page_figures(
    pdfplumber_page: pdfplumber.page.Page,
    fitz_page: fitz.Page,
    config: dict,
    openai_client: OpenAI,
    figure_counter: list[int],  # single-element mutable counter shared across pages
    predetected_regions: list[tuple[float, float, float, float]] | None = None,
) -> list[FigureDescription]:
    """
    Detect figures on one page and return vision-LLM descriptions with y-positions.

    Args:
        pdfplumber_page: used for structural analysis (rects, images, tables)
        fitz_page: used for rendering figure crops to PNG
        config: full extraction config dict
        openai_client: initialized OpenAI client
        figure_counter: ``[n]`` — incremented for each figure found across the document

    Returns:
        List of FigureDescription objects sorted by top_y.
    """
    fig_cfg = config["figure_detection"]
    regions = predetected_regions if predetected_regions is not None else _detect_figure_regions(pdfplumber_page, fig_cfg)

    if not regions:
        return []

    results: list[FigureDescription] = []
    for x0, top, x1, bottom in regions:
        figure_counter[0] += 1
        label = f"Figure {figure_counter[0]}"

        try:
            img_bytes = _render_region(fitz_page, x0, top, x1, bottom, fig_cfg["render_scale"])
            prose = _call_vision_llm(img_bytes, label, config, openai_client)
        except Exception:
            logger.exception("Vision LLM call failed for %s", label)
            prose = "[description unavailable]"

        results.append(FigureDescription(
            x0=x0,
            x1=x1,
            top_y=top,
            bottom_y=bottom,
            label=label,
            description=f"[{label} description: {prose}]",
        ))

    return sorted(results, key=lambda f: f.top_y)


# ── Figure region detection ───────────────────────────────────────────────────


def _detect_figure_regions(
    page: pdfplumber.page.Page,
    fig_cfg: dict,
) -> list[tuple[float, float, float, float]]:
    """
    Return (x0, top, x1, bottom) bounding boxes for detected figure regions.

    Sources:
    - Embedded raster images  (page.images)
    - Vector graphic clusters (rects, curves, and lines — handles both
      traditional box-and-border figures and flowcharts drawn with rounded
      bezier boxes and arrow connectors)
    """
    regions: list[tuple[float, float, float, float]] = []

    # — Raster images —
    for img in page.images:
        bbox = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area >= fig_cfg["min_cluster_area_pts2"]:
            regions.append(bbox)

    # — Vector graphic clusters (flowcharts, diagrams) —
    # Collect rects + curves + lines; cluster-level thresholds do the filtering.
    table_bboxes = [t.bbox for t in page.find_tables()]
    elements = _qualifying_graphic_elements(page, fig_cfg["min_rect_dimension"], table_bboxes)
    clusters = _cluster_rects(elements, fig_cfg["cluster_proximity_pts"])

    for cluster in clusters:
        if len(cluster) < fig_cfg["min_rects_in_cluster"]:
            continue
        bbox = _union_bbox(cluster)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < fig_cfg["min_cluster_area_pts2"]:
            continue
        # Avoid duplicating a region already captured as a raster image
        if not _overlaps_any(bbox, regions, iou_threshold=0.5):
            regions.append(bbox)

    return regions


def _qualifying_graphic_elements(
    page: pdfplumber.page.Page,
    min_rect_dim: float,
    exclude_bboxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Collect bounding boxes for all graphical primitives that could belong to
    a figure region:

    - Rects: must exceed min_rect_dim in both dimensions and not span the
      full page width (horizontal rule / page border).
    - Curves: included at any size — flowchart rounded-rect boxes and
      arrowheads are both curves; the cluster-level filters weed out noise.
    - Lines: included at any size — connectors between flowchart boxes.

    All primitives that overlap substantially with a detected table region
    are excluded.
    """
    page_width = page.width
    result: list[tuple[float, float, float, float]] = []

    def _bbox(obj: dict) -> tuple[float, float, float, float]:
        return float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])

    # Rectangles
    for rect in page.rects:
        w = float(rect.get("width", 0))
        h = float(rect.get("height", 0))
        if w < min_rect_dim or h < min_rect_dim:
            continue
        if w >= page_width * 0.9:
            continue
        bbox = _bbox(rect)
        if not _overlaps_any(bbox, exclude_bboxes, iou_threshold=0.3):
            result.append(bbox)

    # Curves (rounded-rect boxes, arrowheads, decorative shapes)
    for curve in page.curves:
        bbox = _bbox(curve)
        if not _overlaps_any(bbox, exclude_bboxes, iou_threshold=0.3):
            result.append(bbox)

    # Lines (connectors / arrows between diagram nodes)
    for line in page.lines:
        bbox = _bbox(line)
        if not _overlaps_any(bbox, exclude_bboxes, iou_threshold=0.3):
            result.append(bbox)

    return result


def _cluster_rects(
    rects: list[tuple[float, float, float, float]],
    proximity: float,
) -> list[list[tuple[float, float, float, float]]]:
    """
    Greedy single-linkage clustering: a rect joins an existing cluster if it
    is within ``proximity`` pts of any rect already in that cluster.
    """
    clusters: list[list[tuple[float, float, float, float]]] = []

    for rect in rects:
        placed = False
        for cluster in clusters:
            if any(_gap_between(rect, existing) <= proximity for existing in cluster):
                cluster.append(rect)
                placed = True
                break
        if not placed:
            clusters.append([rect])

    return clusters


def _gap_between(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Minimum Euclidean gap between two axis-aligned rectangles (0 if overlapping)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return (dx ** 2 + dy ** 2) ** 0.5


def _union_bbox(
    rects: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Return the bounding box that encloses all rects in the list."""
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _overlaps_any(
    bbox: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> bool:
    """Return True if bbox overlaps with any bbox in ``others`` by ≥ iou_threshold of its area."""
    x0, y0, x1, y1 = bbox
    self_area = max((x1 - x0) * (y1 - y0), 1e-9)

    for ox0, oy0, ox1, oy1 in others:
        ix = max(0.0, min(x1, ox1) - max(x0, ox0))
        iy = max(0.0, min(y1, oy1) - max(y0, oy0))
        if (ix * iy) / self_area >= iou_threshold:
            return True
    return False


# ── Rendering & vision LLM ────────────────────────────────────────────────────


def _render_region(
    fitz_page: fitz.Page,
    x0: float,
    top: float,
    x1: float,
    bottom: float,
    scale: float,
) -> bytes:
    """Render a page region to PNG bytes at the given resolution scale."""
    clip = fitz.Rect(x0, top, x1, bottom)
    mat = fitz.Matrix(scale, scale)
    pix = fitz_page.get_pixmap(matrix=mat, clip=clip)
    return pix.tobytes("png")


def _call_vision_llm(
    img_bytes: bytes,
    label: str,
    config: dict,
    client: OpenAI,
) -> str:
    """Base64-encode the image and call the vision LLM; return the prose description."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt: str = config["vision"]["prompt"]
    model: str = config["vision"]["model"]
    max_tokens: int = config["vision"]["max_tokens"]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\nThis is {label}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
