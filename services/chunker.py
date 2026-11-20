
def chunk_pages(pages, chunk_size=1200, overlap=200):
    """
    Split extracted PDF pages into smaller chunks.
    Preserve page number and section metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    all_chunks = []

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        # Use the first non-empty line as a section label
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        section_title = lines[0][:120] if lines else f"Page {page_number}"

        start = 0

        while start < len(page_text):

            end = start + chunk_size
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_number,
                        "section": section_title
                    }
                })

            # Move forward while keeping overlapping text
            start += chunk_size - overlap

    return all_chunks

