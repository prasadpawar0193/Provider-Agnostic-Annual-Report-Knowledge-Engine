
import os
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    MIN_SIMILARITY_SCORE,
    MISSING_ANSWER
)

from services.pdf_extractor import extract_pdf_content
from services.chunker import chunk_pages
from services.retriever import DocumentRetriever
from services.llm_provider import generate_answer


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

document_store = {}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def build_prompt(question, results):
    """
    Prepare document context and strict answer instructions.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):
        page_number = result["metadata"]["page_number"]

        context_parts.append(
            f"""
[CHUNK {index}]
[PAGE NUMBER: {page_number}]
DOCUMENT TEXT:
{result["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a financial annual report analysis assistant.

Answer the user's question using ONLY the provided document evidence.

STRICT RULES:

1. Use only information available in the document context.
2. Never invent, estimate, or assume financial figures.
3. Copy financial values accurately from the document.
4. Preserve the original currency, number, unit, and percentage.
5. For Indian Rupees, use the actual symbol ₹, never the letter I.
6. Preserve units such as crore, lakh, million and billion.
7. Do not change the meaning or value of any financial figure.
8. Keep the answer short, clear and factual.
9. Do not combine unrelated financial figures.
10. If the answer is not available in the context, respond exactly:
{MISSING_ANSWER}
11. Do not mention a source page in your answer.
12. Do not include unrelated information.

IMPORTANT:
The page number is metadata associated with each chunk.
Use the document text as evidence, not the page number alone.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Return only the answer.
"""

    return prompt


def get_supporting_citations(results, answer):
    """
    Return source pages only when their retrieved text contains
    meaningful evidence for the generated answer.

    This is a conservative text-based check, not a semantic
    guarantee of citation correctness.
    """

    if not answer or answer.strip() == MISSING_ANSWER:
        return []

    answer_lower = answer.lower()

    citations = []
    seen_pages = set()

    # Extract important words from the answer.
    answer_words = {
        word.strip(".,:;()[]{}")
        for word in answer_lower.split()
        if len(word.strip(".,:;()[]{}")) > 2
    }

    # Ignore common words that do not establish evidence.
    stop_words = {
        "the", "was", "were", "for", "from", "with",
        "and", "that", "this", "has", "had", "have",
        "crore", "crores", "lakh", "lakhs", "million",
        "billion", "rupees", "rupee"
    }

    answer_words -= stop_words

    for result in results:

        page_number = result["metadata"]["page_number"]
        source_text = result["text"]

        source_lower = source_text.lower()

        # Check whether the answer's meaningful terms occur
        # in the source chunk.
        matching_words = answer_words.intersection(
            set(source_lower.split())
        )

        if matching_words and page_number not in seen_pages:

            citations.append({
                "page_number": page_number,
                "excerpt": source_text[:400]
            })

            seen_pages.add(page_number)

    return citations


@app.route("/", methods=["GET", "POST"])
def home():

    answer = None
    citations = []
    error = None
    document_id = None
    filename = None

    if request.method == "POST":

        # ---------------- PDF UPLOAD ----------------

        if "pdf_file" in request.files:

            file = request.files["pdf_file"]

            if not file or file.filename == "":
                error = "Please select a PDF file."

            elif not allowed_file(file.filename):
                error = "Only PDF files are allowed."

            else:

                safe_filename = secure_filename(file.filename)

                unique_filename = (
                    f"{uuid.uuid4().hex}_{safe_filename}"
                )

                pdf_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_filename
                )

                try:

                    file.save(pdf_path)

                    pages = extract_pdf_content(pdf_path)

                    if not pages or not any(
                        page["text"].strip() for page in pages
                    ):
                        raise ValueError(
                            "No readable text found in this PDF."
                        )

                    chunks = chunk_pages(
                        pages,
                        chunk_size=CHUNK_SIZE,
                        overlap=CHUNK_OVERLAP
                    )

                    if not chunks:
                        raise ValueError(
                            "Unable to create document chunks."
                        )

                    retriever = DocumentRetriever()
                    retriever.index_chunks(chunks)

                    document_id = uuid.uuid4().hex

                    document_store[document_id] = {
                        "filename": safe_filename,
                        "retriever": retriever,
                        "pdf_path": pdf_path
                    }

                    filename = safe_filename

                except Exception as e:

                    error = f"PDF processing failed: {str(e)}"

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

        # ---------------- QUESTION ANSWERING ----------------

        elif "question" in request.form:

            document_id = request.form.get(
                "document_id", ""
            )

            question = request.form.get(
                "question", ""
            ).strip()

            document = document_store.get(document_id)

            if not document:

                error = "Please upload your annual report again."

            elif not question:

                error = "Please enter a question."

            else:

                filename = document["filename"]

                try:

                    results = document["retriever"].search(
                        question,
                        top_k=TOP_K
                    )

                    if (
                        not results
                        or results[0]["score"] < MIN_SIMILARITY_SCORE
                    ):

                        answer = MISSING_ANSWER

                    else:

                        prompt = build_prompt(
                            question,
                            results
                        )

                        answer = generate_answer(prompt).strip()

                        if not answer:
                            answer = MISSING_ANSWER

                        # Remove unnecessary answer labels.
                        answer = answer.removeprefix("ANSWER:").strip()

                        # Generate conservative source references.
                        citations = get_supporting_citations(
                            results,
                            answer
                        )

                except Exception as e:

                    error = (
                        f"Unable to generate answer: {str(e)}"
                    )

    return render_template(
        "index.html",
        answer=answer,
        citations=citations,
        error=error,
        document_id=document_id,
        filename=filename
    )


@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error="File is too large. Maximum upload size is 50 MB."
    ), 413


if __name__ == "__main__":

    app.run(debug=True)
