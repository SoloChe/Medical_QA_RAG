import os
import glob
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, docs_path="./data/KB_books", 
                 index_path="./data/KB_books/books_index.faiss", 
                 model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                 batch_size=32,
                 device='cpu'):
        self.docs_path = docs_path
        self.index_path = index_path
        self.batch_size = batch_size

        # Check for GPU availability
        self.device = device
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        # Load documents
        self.documents = self._load_documents()
        # Load or build index
        if not os.path.exists(index_path):
            print("No FAISS index found. Building a new one...")
            self._build_index()
        else:
            print(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(index_path)
            if self.device == "cuda":
                self._move_index_to_gpu()

    def _load_documents(self):
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Document directory not found: {self.docs_path}")
        
        jsonl_files = glob.glob(os.path.join(self.docs_path, "*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in {self.docs_path}")

        documents = []
        for file in jsonl_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        if "text" in doc:
                            documents.append({
                                              "text":doc["text"],
                                              "chunk_id":doc.get("chunk_id", "unknown"),
                                              "source":doc.get("source", "unknown")
                                              })
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line in {file}")

        print(f"Loaded {len(documents)} documents from {len(jsonl_files)} files.")
        return documents

    def _build_index(self):
        # Generate embeddings in batches
        print("Generating embeddings...")
        embeddings = []
        for i in range(0, len(self.documents), self.batch_size):
            batch = [doc["text"] for doc in self.documents[i:i+self.batch_size]]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        # Stack all embeddings
        embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        if self.device == "cuda":
            print("Using GPU for indexing...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(embeddings)

        # Save the index (on CPU)
        if self.device == "cuda":
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, self.index_path)
        else:
            faiss.write_index(self.index, self.index_path)

        print(f"FAISS index saved to {self.index_path}")

    def _move_index_to_gpu(self):
        print("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def retrieve(self, query, top_k=5):
        # Encode the query
        query_embedding = self.model.encode([query], show_progress_bar=False)

        # Perform the search
        D, I = self.index.search(query_embedding, top_k)
        
        # Return results
        results = [
            {
                "text": self.documents[i]["text"],
                "score": float(D[0][j]),
                "source": self.documents[i].get("source", "unknown"),
                "chunk_id": self.documents[i].get("chunk_id", f"chunk_{i}")
            }
            for j, i in enumerate(I[0])
        ]
        return results


if __name__ == "__main__":
    retriever = FAISSRetriever()
    query = "What are the symptoms of Alzheimer's disease?"
    results = retriever.retrieve(query, top_k=3)
    for result in results:
        score = result["score"]
        doc = result["text"]
        source = result["source"]
        chunk_id = result["chunk_id"]
        print(f"\nChunk_id: {chunk_id}, Source: {source}, Score: {score:.4f}\nDocument: {doc[:500]}...\n")

