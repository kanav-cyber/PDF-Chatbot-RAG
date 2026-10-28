from pypdf import PdfReader
from pathlib import Path

def load_pdf(file_path:str)->list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Pdf not found:{path}")

    reader = PdfReader(str(path))

    doc = []

    for pg_no,page in enumerate(reader.pages,start=1):
        text = page.extract_text() or ""

        if text.strip():
            doc.append({
                "text" : text,
                "source" : path.name,
                "page" : pg_no
            })
    return doc

if __name__ == "__main__":
    pdf_path = "data/deep_lr.pdf"

    pages = load_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages")

    for page in pages:
        print(f"\nsource : {page['source']}")
        print(f"Page : {page['page']}")
        print(page["text"][:500])
