import fitz


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number,
            "source": pdf_path
        })

    doc.close()

    return pages


if __name__ == "__main__":
    pdf_path = "documents/TOURISM IN INDIA.pdf"

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:1000])