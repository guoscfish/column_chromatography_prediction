"""Check SVG editability, PNG size/DPI, palette; render every PDF via Poppler."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from style import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pdf-python",default=str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"))
    args=parser.parse_args()
    names=sorted(p.stem for p in FIGURES.glob("F[0-9][0-9]_*.svg"))+sorted(p.stem for p in FIGURES.glob("R[0-9][0-9]_*.svg"))
    assert len(names)==13
    reports=[]
    for name in names:
        source=(FIGURES/f"{name}.svg").read_text(encoding="utf-8")
        tree=ET.fromstring(source)
        tags=[element.tag.rsplit("}",1)[-1] for element in tree.iter()]
        assert tags.count("text")>=5,name
        assert "image" not in tags,name
        assert "Microsoft YaHei" in source,name
        colors={x.upper() for x in re.findall(r"#[0-9a-fA-F]{6}\b",source)}
        assert colors<=set(PALETTE),(name,colors-set(PALETTE))
        img=Image.open(FIGURES/f"{name}.png")
        assert img.size==(4800,2700),name
        assert all(abs(float(x)-300)<.01 for x in img.info["dpi"]),name
        assert img.convert("RGB").getpixel((0,0))==(255,255,255),name
        reports.append({"figure":name,"editable_svg_text_elements":tags.count("text"),
                        "raster_images_in_svg":0,"colors":sorted(colors),"png_size":list(img.size),
                        "png_dpi":list(img.info["dpi"]),"font":"Microsoft YaHei"})
    (QA/"asset_structure_audit.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
    subprocess.run([args.pdf_python,str(OUT/"src/inspect_pdfs.py"),str(FIGURES)],check=True)
    binary=shutil.which("pdftoppm")
    if not binary:
        raise RuntimeError("Poppler pdftoppm is required for final rendering QA")
    dest=QA/"pdf_render"
    dest.mkdir(parents=True,exist_ok=True)
    for name in names:
        subprocess.run([binary,"-png","-singlefile","-scale-to","1600",
                        str(FIGURES/f"{name}.pdf"),str(dest/name)],check=True,stdout=subprocess.DEVNULL)
    # PDF renders, not the original PNGs, feed this verification contact sheet.
    for group,subset,cols in [("F01_F03",names[:3],1),("F04_F09",names[3:9],2),("R01_R04",names[9:],2)]:
        thumb=(1280,720) if cols==1 else (960,540)
        rows=(len(subset)+cols-1)//cols
        sheet=Image.new("RGB",(cols*thumb[0],rows*thumb[1]),WHITE)
        for i,name in enumerate(subset):
            img=Image.open(dest/f"{name}.png").convert("RGB")
            img=ImageOps.contain(img,thumb,Image.Resampling.LANCZOS)
            sheet.paste(img,((i%cols)*thumb[0],(i//cols)*thumb[1]))
        sheet.save(QA/f"pdf_preview_{group}.png")
    print("SVG/PNG checks passed; all 13 vector PDFs rendered with Poppler")

if __name__=="__main__":
    main()
