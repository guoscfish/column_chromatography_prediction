"""Combine the 13 vector pages into a review catalog using pypdf."""
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

def merge(directory):
    directory=Path(directory)
    files=sorted(directory.glob("F[0-9][0-9]_*.pdf"))+sorted(directory.glob("R[0-9][0-9]_*.pdf"))
    if len(files)!=13:
        raise ValueError(f"Expected 13 PDFs, found {len(files)}")
    writer=PdfWriter()
    for path in files:
        page=PdfReader(path)
        if len(page.pages)!=1:
            raise ValueError(path)
        writer.append(page,outline_item=path.stem)
    writer.add_metadata({"/Title":"柱色谱主动学习科研图集", "/Author":"Research visualization",
                         "/Subject":"9 method diagrams and 4 recorded result figures"})
    with (directory/"all_figures_review.pdf").open("wb") as f:
        writer.write(f)
    print("PDF review catalog: 13 pages")

if __name__=="__main__":
    merge(sys.argv[1])
