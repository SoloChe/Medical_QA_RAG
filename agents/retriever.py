import os
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, docs_path="./data/rag/rag_data.txt"):
        self.docs_path = docs_path
        self.documents = self._load_documents()
        self.tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_documents(self):
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Document file not found: {self.docs_path}")
        with open(self.docs_path, "r") as f:
            documents = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(documents)} documents from {self.docs_path}")
        return documents

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = [(self.documents[i], scores[i]) for i in ranked_indices]
        return results

if __name__ == "__main__":
    retriever = BM25Retriever()
    query = "What are the symptoms of Alzheimer's disease?"
    results = retriever.retrieve(query, top_k=3)
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: {doc}")
