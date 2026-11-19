"""Render PDF proofs, verify raster/vector geometry and compare PNG/PDF content."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as XML

import numpy as np
from PIL import Image, ImageFilter


REPORT = Path(__file__).resolve().parents[1]
PLOTS = REPORT / "plots"
PREVIEWS = REPORT / "rendered_previews"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def raster_pdf(path, pdftoppm):
    destination = PREVIEWS / path.stem
    subprocess.run([pdftoppm, "-r", "80", "-singlefile", "-png",
                    str(path), str(destination)], check=True)
    return destination.with_suffix(".png")


def contrast_mask(image):
    return np.asarray(image.convert("L")) < 215


def qa_one(png_path, pdftoppm):
    pdf_path = png_path.with_suffix(".pdf")
    svg_path = png_path.with_suffix(".svg")
    assert b"/FontFile2" in pdf_path.read_bytes(), "expected embedded TrueType outlines"
    with Image.open(png_path) as raster:
        assert raster.size == (4000, 2250)
        png = raster.convert("RGB")
    svg = XML.parse(svg_path).getroot()
    _, _, width, height = map(float, re.split(r"[ ,]+", svg.attrib["viewBox"]))
    assert abs(width / height - 16 / 9) < 1e-5
    svg_namespace = "{http://www.w3.org/2000/svg}"
    assert sum(1 for _ in svg.iter(svg_namespace + "path")) > 10
    assert not any(svg.iter(svg_namespace + "image")), "plot should be vector"
    proof_path = raster_pdf(pdf_path, pdftoppm)
    with Image.open(proof_path) as proof:
        assert proof.size == (1280, 720), proof.size
        ref = proof.convert("RGB")
    scaled = png.resize(ref.size, Image.Resampling.LANCZOS)
    left = contrast_mask(ref)
    right = contrast_mask(scaled)
    overlap = left & right
    union = left | right
    iou = np.count_nonzero(overlap) / np.count_nonzero(union)
    left_wide = np.asarray(Image.fromarray(left).convert("L").filter(ImageFilter.MaxFilter(3))) > 0
    right_wide = np.asarray(Image.fromarray(right).convert("L").filter(ImageFilter.MaxFilter(3))) > 0
    agreement = min((left_wide & right).sum() / right.sum(), (right_wide & left).sum() / left.sum())
    a = np.asarray(ref.filter(ImageFilter.GaussianBlur(0.75)), dtype=float)
    b = np.asarray(scaled.filter(ImageFilter.GaussianBlur(0.75)), dtype=float)
    mae = np.abs(a - b).mean()
    # Poppler and Agg anti-alias differently at 1280 px. Use both foreground
    # spatial overlap and whole-canvas pixel error to catch render differences.
    assert agreement > 0.93 and mae < 3.0, (png_path.name, agreement, mae)
    ink = np.argwhere(union)
    assert len(ink) and (ink.min(axis=0) > 8).all()
    assert ((np.array([720, 1280]) - ink.max(axis=0)) > 8).all()
    return {
        "stem": png_path.stem, "png_pixels": list(png.size),
        "pdf_proof_pixels": list(ref.size), "svg_viewbox_pt": [width, height],
        "png_pdf_foreground_iou": round(iou, 5), "foreground_agreement_within_one_pixel": round(agreement, 5), "png_pdf_blurred_pixel_mae_0_to_255": round(mae, 5),
        "sha256": {path.suffix[1:]: sha256(path) for path in [png_path, svg_path, pdf_path, proof_path]
                   if path != proof_path},
        "pdf_proof_sha256": sha256(proof_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--partial", action="store_true", help="allow previewing files as they arrive")
    args = parser.parse_args()
    pngs = sorted(PLOTS.glob("*.png"))
    assert args.partial or len(pngs) == 9, len(pngs)
    proofs = [qa_one(path, args.pdftoppm) for path in pngs]
    if not args.partial:
        assert len(list(PLOTS.glob("*.svg"))) == len(list(PLOTS.glob("*.pdf"))) == 9
        (REPORT / "sources" / "visual_qa.json").write_text(
            json.dumps(proofs, indent=2) + "\n", encoding="utf-8")
    for proof in proofs:
        print(proof["stem"], proof["png_pdf_foreground_iou"], proof["png_pdf_blurred_pixel_mae_0_to_255"])


if __name__ == "__main__":
    main()
