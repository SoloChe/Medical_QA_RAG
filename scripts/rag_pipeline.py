import sys
import os
# Add the project root directory to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from agents.retriever import BM25Retriever
from agents.contextualizer import Contextualizer
from agents.generator import Generator
from agents.fact_checker import FactChecker
from agents.summarizer import Summarizer

class RAGPipeline:
    def __init__(self,
                 retriever_path="./data/rag/rag_data.txt",
                 context_model_name="mistralai/Mistral-7B-v0.1",
                 generator_model_name="mistralai/Mistral-7B-v0.1",
                 generator_model_dir="./saved_models/lora_finetuned/checkpoint-72000",
                 fact_checker_model="all-mpnet-base-v2",
                 summarizer_model="facebook/bart-large-cnn",
                 top_k=3,
                 device="cpu"):
        
        self.retriever = BM25Retriever(docs_path=retriever_path)
        self.contextualizer = Contextualizer(model_name=context_model_name, top_k=top_k)
        self.generator = Generator(base_model_name=generator_model_name, adapter_dir=generator_model_dir, device=device)
        self.fact_checker = FactChecker(model_name=fact_checker_model, device=device)
        self.summarizer = Summarizer(model_name=summarizer_model, device=device)

    def run(self, query, top_k=3, max_length=512, temperature=0.7, summarize=True):
        # Retrieve top-k passages
        retrieved_docs = [doc for doc, _ in self.retriever.retrieve(query, top_k=top_k)]
        print(f"[INFO] Retrieved {len(retrieved_docs)} documents")
        
        # Contextualize the input
        context_input = self.contextualizer.build_input(query, retrieved_docs)
        print(f"[INFO] Contextualized Input: {context_input[:200]}...")
        
        # Generate the response
        response = self.generator.generate(context_input, max_length=max_length, temperature=temperature)
        print(f"[INFO] Generated Response: {response}")
        
        # Fact check the response
        is_factually_correct, top_matches = self.fact_checker.check_fact(response, retrieved_docs)
        print(f"[INFO] Fact Check: {'PASS' if is_factually_correct else 'FAIL'}")
        if not is_factually_correct:
            print("[INFO] Top Supporting Passages:")
            for context, score in top_matches:
                print(f"Score: {score:.4f} | Context: {context}")
        
        # Summarize the response (optional)
        if summarize:
            response = self.summarizer.summarize(response)
            print(f"[INFO] Summarized Response: {response}")
        
        return response

if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    query = "What are the symptoms of Alzheimer's disease?"
    final_response = rag_pipeline.run(query)
    print("\nFinal Response:", final_response)
