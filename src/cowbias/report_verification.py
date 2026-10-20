"""Check report text, table cells, embedded genuine figures, hyperlinks and geometry."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

from cowbias.evidence import sha256


def normalized(text):
    return re.sub(r"\s+", "", text)


def verify_pdf(pdf, manifest):
    if sha256(pdf) != manifest["pdf_sha256"]:
        raise ValueError("Report PDF hash mismatch")
    document = pymupdf.open(pdf)
    if len(document) != manifest["expected_pages"]:
        raise ValueError(
            f"Expected {manifest['expected_pages']} readable report pages, got {len(document)}"
        )
    texts = [p.get_text() for p in document]
    if any(len(t.strip()) < 100 for t in texts):
        raise ValueError("Empty or nonselectable report page")
    tables, figures = 0, 0
    for entry in manifest["entries"]:
        page = document[entry["report_page"] - 1]
        text = normalized(texts[entry["report_page"] - 1])
        if normalized(entry["id"]) not in text:
            raise ValueError(f"Report page mapping incorrect: {entry['id']}")
        if entry["kind"] == "derived_table":
            for row in entry["cells"]:
                for value in row:
                    if normalized(str(value)) not in text:
                        raise ValueError(f"Table cell absent from PDF: {entry['id']}: {value}")
            tables += 1
        else:
            matches = []
            for img in page.get_images(full=True):
                pix = pymupdf.Pixmap(document, img[0])
                if pix.alpha:
                    pix = pymupdf.Pixmap(pix, 0)
                if pix.colorspace.n != 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                matches.append(hashlib.sha256(pix.samples).hexdigest())
            if entry["rgb_pixels_sha256"] not in matches:
                raise ValueError(f"Embedded figure pixels do not match source: {entry['id']}")
            figures += 1
    links = [link["uri"] for page in document for link in page.get_links() if "uri" in link]
    if len(links) < 8 or any(not u.startswith("https://") for u in links):
        raise ValueError("Missing/invalid repository, reference or evidence hyperlinks")
    # Bounds detect clipped text; visual PNG inspection remains a separate acceptance step.
    for page in document:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    rect = pymupdf.Rect(span["bbox"])
                    if not page.rect.contains(rect):
                        raise ValueError(f"Text outside page bounds, page {page.number + 1}")
                    if "\ufffd" in span["text"] or "\u25a0" in span["text"]:
                        raise ValueError("Missing font glyph in report")
    text_digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    return {
        "status": "PASS",
        "pages": len(document),
        "tables": tables,
        "figures": figures,
        "hyperlinks": len(links),
        "table_cells": "all present on declared pages",
        "figure_pixels": "all match original/derived source raster pixels",
        "text_bounds": "PASS",
        "selectable_text_sha256": text_digest,
        "visual_inspection": "Requires rendered-page human/agent inspection; not inferred here",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--render-dir", type=Path)
    p.add_argument(
        "--compare-manifest", type=Path, help="Compare extracted content and figure provenance"
    )
    args = p.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        result = verify_pdf(args.pdf, manifest)
        if args.compare_manifest:
            other = json.loads(args.compare_manifest.read_text())
            for key in (
                "run_id",
                "report_sources",
                "scientific_source_hashes",
                "entries",
                "archives",
            ):
                if manifest[key] != other[key]:
                    raise ValueError(f"Generated report differs from tracked report: {key}")
            if result["selectable_text_sha256"] != other["pdf_checks"]["selectable_text_sha256"]:
                raise ValueError("Generated PDF content differs from tracked report")
            result["tracked_report_comparison"] = "PASS"
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=False)
            with pymupdf.open(args.pdf) as pdf:
                for i, page in enumerate(pdf, 1):
                    page.get_pixmap(dpi=120, alpha=False).save(
                        args.render_dir / f"page-{i:02d}.png"
                    )
            result["rendered_pages"] = len(pdf) if not pdf.is_closed else result["pages"]
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Report verification failed: {exc}")
        return 1
