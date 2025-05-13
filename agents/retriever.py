import os
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, docs_path="./data/rag/rag.json", 
                 index_path="./data/rag/pubmedqa_index.faiss", 
                 model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                 batch_size=32):
        self.docs_path = docs_path
        self.index_path = index_path
        self.batch_size = batch_size

        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Load documents
        self.documents = self._load_documents()
        
        # Load or build index
        if not os.path.exists(index_path):
            print("No FAISS index found. Building a new one...")
            self._build_index()
        else:
            print(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            if self.device == "cuda":
                self._move_index_to_gpu()

    def _load_documents(self):
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Document file not found: {self.docs_path}")
        
        documents = []
        with open(self.docs_path, "r") as f:
            for line in f:
                doc = json.loads(line.strip())
                documents.append(doc["text"])
        
        print(f"Loaded {len(documents)} documents from {self.docs_path}")
        return documents

    def _build_index(self):
        # Generate embeddings in batches
        print("Generating embeddings...")
        embeddings = []
        for i in range(0, len(self.documents), self.batch_size):
            batch = self.documents[i:i+self.batch_size]
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

        # Save the index
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
        results = [(self.documents[i], D[0][j]) for j, i in enumerate(I[0])]
        return results

if __name__ == "__main__":
    retriever = FAISSRetriever()
    query = "What are the symptoms of Alzheimer's disease?"
    results = retriever.retrieve(query, top_k=3)
    for doc, score in results:
        print(f"Score: {score:.4f} | Document: {doc}")
