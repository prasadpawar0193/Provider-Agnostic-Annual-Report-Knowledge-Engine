
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentRetriever:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.chunks = []
        self.document_vectors = None

    def index_chunks(self, chunks):
        """
        Convert document chunks into TF-IDF vectors.
        """

        if not chunks:
            raise ValueError("No document chunks available for indexing")

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        self.document_vectors = self.vectorizer.fit_transform(texts)

    def search(self, question, top_k=5):
        """
        Find the most relevant document chunks.
        """

        if self.document_vectors is None:
            raise ValueError("Please index the document before searching")

        question_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector,
            self.document_vectors
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            # Ignore chunks with no meaningful similarity
            if scores[index] <= 0:
                continue

            result = {
                "text": self.chunks[index]["text"],
                "metadata": self.chunks[index]["metadata"],
                "score": float(scores[index])
            }

            results.append(result)

        return results
