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


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    pdf_path = "documents/TOURISM IN INDIA.pdf"

    pages = extract_text_from_pdf(pdf_path)

    total_chunks = 0

    for page in pages:

        chunks = chunk_text(page["text"])

        for i, chunk in enumerate(chunks, start=1):

            print("\n-------------------------")
            print(f"Page: {page['page']}")
            print(f"Source: {page['source']}")
            print(f"Chunk: {i}")
            print("-------------------------")
            print(chunk[:500])

            total_chunks += 1

    print("\n=========================")
    print(f"Total chunks: {total_chunks}")
    print("=========================")