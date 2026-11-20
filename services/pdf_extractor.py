
import fitz


def extract_pdf_content(pdf_path):
    """
    Extract PDF text, positioned blocks and tables.
    Preserve the original page numbers.
    """

    document = fitz.open(pdf_path)
    extracted_pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text", sort=True)
            blocks = page.get_text("blocks", sort=True)

            page_content = {
                "page_number": page_number,
                "text": page_text,
                "blocks": [],
                "tables": []
            }

            # Extract positioned text blocks
            for block in blocks:

                if len(block) >= 5 and isinstance(block[4], str):

                    block_text = block[4].strip()

                    if block_text:
                        page_content["blocks"].append({
                            "text": block_text,
                            "bbox": block[:4]
                        })

            # Detect and extract tables on this page
            try:
                table_finder = page.find_tables()

                for table_number, table in enumerate(
                    table_finder.tables,
                    start=1
                ):

                    table_data = table.extract()

                    if table_data:
                        page_content["tables"].append({
                            "table_number": table_number,
                            "data": table_data
                        })

                        # Add table content to searchable page text
                        table_text = "\n".join(
                            " | ".join(
                                str(cell) if cell is not None else ""
                                for cell in row
                            )
                            for row in table_data
                        )

                        page_content["text"] += (
                            "\n\nTABLE DATA:\n" + table_text
                        )

            except Exception as error:
                print(
                    f"Table extraction warning on page "
                    f"{page_number}: {error}"
                )

            extracted_pages.append(page_content)

    finally:
        document.close()

    return extracted_pages
