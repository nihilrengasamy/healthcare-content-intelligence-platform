import fitz
from pathlib import Path
pdf = Path(r"C:\Users\Nihil Rengasamy\Documents\Codex\2026-06-27\hi\outputs\healthcare-content-intelligence\report_artifacts\rendered\Cotiviti_AI_Healthcare_Content_Intelligence_Report.pdf")
doc = fitz.open(pdf)
print(len(doc))
for i, page in enumerate(doc, start=1):
    text = page.get_text("text").strip().replace("\n", " ")
    print(f"PAGE {i}: {text[:180]}")
