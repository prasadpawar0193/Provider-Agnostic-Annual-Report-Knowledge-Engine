
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

# Store document data for the current application session
document_store = {}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def home():

    answer = None
    citations = []
    error = None
    document_id = None
    filename = None

    if request.method == "POST":

        # Handle PDF upload
        if "pdf_file" in request.files:

            file = request.files["pdf_file"]

            if not file or file.filename == "":
                error = "Please select a PDF file."

            elif not allowed_file(file.filename):
                error = "Only PDF files are allowed."

            else:
                safe_filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
                pdf_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_filename
                )

                try:
                    file.save(pdf_path)

                    # Extract PDF content
                    pages = extract_pdf_content(pdf_path)

                    if not pages or not any(
                        page["text"].strip() for page in pages
                    ):
                        raise ValueError(
                            "No readable text found in this PDF."
                        )

                    # Create document chunks
                    chunks = chunk_pages(
                        pages,
                        chunk_size=CHUNK_SIZE,
                        overlap=CHUNK_OVERLAP
                    )

                    if not chunks:
                        raise ValueError(
                            "Unable to create document chunks."
                        )

                    # Index document chunks
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

        # Handle question answering
        elif "question" in request.form:

            document_id = request.form.get("document_id", "")
            question = request.form.get("question", "").strip()

            document = document_store.get(document_id)

            if not document:
                error = "Please upload your annual report again."

            elif not question:
                error = "Please enter a question."

            else:
                filename = document["filename"]

                try:
                    # Retrieve relevant chunks
                    results = document["retriever"].search(
                        question,
                        top_k=TOP_K
                    )

                    # Check whether relevant information exists
                    if (
                        not results
                        or results[0]["score"] < MIN_SIMILARITY_SCORE
                    ):
                        answer = MISSING_ANSWER

                    else:
                        # Prepare document context
                        context_parts = []

                        for result in results:
                            context_parts.append(
                                f"Page {result['metadata']['page_number']}:\n"
                                f"{result['text']}"
                            )

                        context = "\n\n".join(context_parts)

                        prompt = f"""
You are an annual report analysis assistant.

Answer the question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent financial figures.
3. If the answer is not available in the context, respond exactly:
   {MISSING_ANSWER}
4. Include relevant source page numbers in your answer.
5. Keep the answer clear and factual.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                        answer = generate_answer(prompt)

                        if not answer.strip():
                            answer = MISSING_ANSWER

                        # Attach source citations
                        citations = [
                            {
                                "page_number": result["metadata"]["page_number"],
                                "excerpt": result["text"][:400]
                            }
                            for result in results
                        ]

                except Exception as e:
                    error = f"Unable to generate answer: {str(e)}"

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

