
import fitz


def extract_pdf_content(pdf_path):
    """
    Extract text and positioned blocks from each PDF page.
    Preserve original page numbers.
    """

    extracted_pages = []

    with fitz.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text", sort=True)
            blocks = page.get_text("blocks", sort=True)

            page_blocks = []

            for block in blocks:

                if len(block) >= 5 and isinstance(block[4], str):

                    block_text = block[4].strip()

                    if block_text:
                        page_blocks.append({
                            "text": block_text,
                            "bbox": block[:4]
                        })

            if not page_text.strip():
                continue

            extracted_pages.append({
                "page_number": page_number,
                "text": page_text,
                "blocks": page_blocks,
                "tables": []
            })

    return extracted_pages
