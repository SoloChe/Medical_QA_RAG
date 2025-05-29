import os
import glob
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from template import *
import json


class Ranker:
    def __init__(self, device):
        self.device = device
        model_name = "gsarti/biobert-nli"  # or use a biomedical reranker
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self.model.eval()

    def rank(self, query, docs):
        pairs = [f"{query} [SEP] {doc['text']}" for doc in docs]
        inputs = self.tokenizer(
            pairs, padding=True, max_length=512, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            self.model.eval()
            scores = self.model(**inputs).logits
        return scores.cpu().tolist()


class Parser:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def get_prompt(self, question):
        question_parse_prompt = question_parse.render(question=question)
        messages = [
            {"role": "system", "content": question_parse_system},
            {"role": "user", "content": question_parse_prompt},
        ]
        return messages

    def string_to_list(self, string):
        sub_questions = json.loads(string)
        return sub_questions["sub_questions"]

    def question_parse(self, question):

        # Generate the prompt using the chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            self.get_prompt(question), tokenize=False, add_generation_prompt=True
        )

        # Tokenize the formatted prompt
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1000,  # Ensure it respects the model's max length
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode the generated output
        decoded_output = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return self.string_to_list(decoded_output)


class CustomizeSentenceTransformer(
    SentenceTransformer
):  # change the default pooling "MEAN" to "CLS"
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print(
            "No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(
                model_name_or_path
            )
        )
        token = kwargs.get("token", None)
        cache_folder = kwargs.get("cache_folder", None)
        revision = kwargs.get("revision", None)
        trust_remote_code = kwargs.get("trust_remote_code", False)
        if (
            "token" in kwargs
            or "cache_folder" in kwargs
            or "revision" in kwargs
            or "trust_remote_code" in kwargs
        ):
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
                tokenizer_args={
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        return [transformer_model, pooling_model]


class FAISSRetriever:
    def __init__(
        self,
        docs_path="./data/KB_books_v2",
        index_path="./data/KB_books_v2/books_index.faiss",
        model_name="ncbi/MedCPT-Query-Encoder",
        batch_size=256,
        use_ranker=False,
        device="cpu",
    ):

        self.docs_path = docs_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.device = device

        # self.model = SentenceTransformer(model_name, device=self.device)
        self.query_model = CustomizeSentenceTransformer(model_name, device=self.device)
        self.query_model.eval()
        self.documents = self._load_documents()

        self.ranker = Ranker(device) if use_ranker else None

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
                            documents.append(
                                {
                                    "text": doc["text"],
                                    "chunk_id": doc.get("chunk_id", "unknown"),
                                    "source": doc.get("source", "unknown"),
                                }
                            )
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line in {file}")

        print(f"Loaded {len(documents)} documents from {len(jsonl_files)} files.")
        return documents

    def _build_index(self):
        # Generate embeddings in batches
        self.doc_model = CustomizeSentenceTransformer(
            "ncbi/MedCPT-Article-Encoder", device=self.device
        )
        self.doc_model.eval()
        print("Generating embeddings...")
        embeddings = []
        for i in range(0, len(self.documents), self.batch_size):
            batch = [
                [doc["source"], doc["text"]]
                for doc in self.documents[i : i + self.batch_size]
            ]
            batch_embeddings = self.doc_model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)

        # Stack all embeddings
        embeddings = np.vstack(embeddings)

        # Create FAISS index
        assert (
            embeddings.shape[1] == 768
        ), "Expected embedding dimension is 768 for MedCPT-Article-Encoder."
        self.index = faiss.IndexFlatL2(768)  # L2 distance
        # self.index = faiss.IndexHNSWFlat(768, 32)
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
        query_embedding = self.query_model.encode([query], show_progress_bar=False)
        # Perform the search
        D, I = self.index.search(query_embedding, top_k)

        # Return results
        results = [
            {
                "text": self.documents[i]["text"],
                "score": float(D[0][j]),
                "source": self.documents[i].get("source", "unknown"),
                "chunk_id": self.documents[i].get("chunk_id", f"chunk_{i}"),
            }
            for j, i in enumerate(I[0])
        ]
        # Rank the results
        if self.ranker:
            ranked_scores = self.ranker.rank(query, results)
            for i, score in enumerate(ranked_scores):
                results[i]["ranker_score"] = score
            results = sorted(results, key=lambda x: x["ranker_score"][1], reverse=True)
        else:
            results = sorted(results, key=lambda x: x["score"], reverse=True)

        return self.format_context(results)

    def format_context(self, passages):
        # Select top-k passages
        top_passages = passages
        # Combine passages with metadata for citation
        context_passages = []
        for idx, p in enumerate(top_passages):
            source = p.get("source", "unknown")
            chunk_id = p.get("chunk_id", f"chunk_{idx}")
            text = p["text"].strip()
            context_passages.append(f"[{chunk_id} | {source}]: {text}")
            context = "\n\n".join(context_passages)
        return context


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query = "What are the symptoms of Alzheimer's disease?"
    parser = Parser(device=device)
    parsed_query = parser.question_parse(query)
    print(f"Parsed Query: {parsed_query}")

    retriever = FAISSRetriever(device=device)
    results = retriever.retrieve(query, top_k=3)
    print(results)

    # for result in results:
    #     score = result["score"]
    #     doc = result["text"]
    #     source = result["source"]
    #     chunk_id = result["chunk_id"]
    #     print(f"\nChunk_id: {chunk_id}, Source: {source}, Score: {score:.4f}\nDocument: {doc[:500]}...\n")
