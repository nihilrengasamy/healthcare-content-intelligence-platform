import fitz
from pathlib import Path
pdf = Path(r"C:\Users\Nihil Rengasamy\Documents\Codex\2026-06-27\hi\outputs\healthcare-content-intelligence\report_artifacts\rendered\Cotiviti_AI_Healthcare_Content_Intelligence_Report.pdf")
out_dir = Path(r"C:\Users\Nihil Rengasamy\Documents\Codex\2026-06-27\hi\outputs\healthcare-content-intelligence\report_artifacts\rendered\pages")
out_dir.mkdir(parents=True, exist_ok=True)
doc = fitz.open(pdf)
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(out_dir / f"page-{i+1}.png")
print(len(doc))
