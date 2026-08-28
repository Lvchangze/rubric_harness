"""Extract the RaR paper PDF into plain text for close reading.

Writes analysis/out/paper_text.txt with explicit page markers so that quotes can
be traced back to a page number.
"""

import pathlib

import fitz

REPO = pathlib.Path(__file__).resolve().parent.parent
PDF = REPO / "rubric_as_reward.pdf"
OUT_DIR = REPO / "analysis" / "out"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    chunks = []
    for page_index, page in enumerate(doc, start=1):
        chunks.append(f"\n\n===== PAGE {page_index} =====\n")
        chunks.append(page.get_text("text"))
    text = "".join(chunks)
    (OUT_DIR / "paper_text.txt").write_text(text, encoding="utf-8")
    print(f"pages={doc.page_count} chars={len(text)} lines={text.count(chr(10))}")


if __name__ == "__main__":
    main()
