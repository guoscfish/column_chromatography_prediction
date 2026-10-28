"""Read-only vector PDF validation; requires pypdf."""
import json
import sys
from pathlib import Path
from pypdf import PdfReader

def main(directory):
    directory=Path(directory)
    reports=[]
    for path in sorted(directory.glob("*.pdf")):
        reader=PdfReader(path)
        expected=13 if path.stem=="all_figures_review" else 1
        assert len(reader.pages)==expected,path
        pages=[]
        for page in reader.pages:
            assert abs(float(page.mediabox.width)/float(page.mediabox.height)-16/9)<1e-8,path
            fonts=[]
            font_resources=page.get("/Resources").get_object().get("/Font")
            for value in font_resources.get_object().values():
                font=value.get_object()
                if "/DescendantFonts" in font:
                    child=font["/DescendantFonts"][0].get_object()
                else:
                    child=font
                desc=child.get("/FontDescriptor")
                desc=desc.get_object() if desc else {}
                fonts.append({"name":str(font.get("/BaseFont","")),
                              "embedded":any(k in desc for k in ["/FontFile","/FontFile2","/FontFile3"])})
            assert any("yahei" in f["name"].lower() for f in fonts),path
            assert all(f["embedded"] for f in fonts),path
            extracted=page.extract_text()
            assert len(extracted)>50,path
            assert any("\u4e00"<=ch<="\u9fff" for ch in extracted),path
            pages.append({"width_points":float(page.mediabox.width),"height_points":float(page.mediabox.height),
                          "fonts":fonts,"extractable_text_characters":len(extracted)})
        reports.append({"file":path.name,"page_count":len(reader.pages),"pages":pages})
    assert len(reports)==14
    target=directory.parent/"qa/pdf_structure_audit.json"
    target.write_text(json.dumps(reports,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("PDF structure passed: 14 files, 26 pages; YaHei and math fonts embedded, text extractable")

if __name__=="__main__":
    main(sys.argv[1])
