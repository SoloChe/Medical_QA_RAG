from .retriever import FAISSRetriever
from .contextualizer import Contextualizer
from .generator import Generator
from .fact_checker import FactChecker
from .summarizer import Summarizer
import torch

class RAGPipeline:
    def __init__(self,
                 retriever_path="./data/KB_books",
                 retriever_index_path="./data/KB_books/books_index.faiss", 
                 context_model_name="mistralai/Mistral-7B-Instruct-v0.2",
                 generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
                 generator_model_dir=None,
                 fact_checker_model="all-mpnet-base-v2",
                 summarizer_model="facebook/bart-large-cnn",
                 top_k_con=5,
                 choice=False,
                 multi_choice=False,
                 device="cpu"):
        
        self.retriever = FAISSRetriever(docs_path=retriever_path, index_path=retriever_index_path, device=device)
        self.contextualizer = Contextualizer(model_name=context_model_name, top_k=top_k_con, choice=choice, multi_choice=multi_choice, device=device)
        self.generator = Generator(base_model_name=generator_model_name, adapter_dir=generator_model_dir, device=device)
        self.fact_checker = FactChecker(model_name=fact_checker_model, device=device)
        self.summarizer = Summarizer(model_name=summarizer_model, device=device)

    def run(self, query, top_k_ret=10, max_new_tokens_gen=1000, do_sample_gen=False, summarize=True, fact_check=True):
        # Retrieve top-k passages
        retrieved_docs = [doc for doc in self.retriever.retrieve(query, top_k=top_k_ret)]
        # print(f"\n[INFO] Retrieved {len(retrieved_docs)} documents")
        
        # Contextualize the input
        context_input = self.contextualizer.build_input(query, retrieved_docs)
        # print(f"\n[INFO] Contextualized Input:\n{context_input}...")
        
        # Generate the response
        response = self.generator.generate(context_input, 
                                           max_new_tokens=max_new_tokens_gen,
                                           do_sample=do_sample_gen)
        # print(f"\n[INFO] Generated Response:\n{response}")
        
        # Fact check the response
        if fact_check:
            is_factually_correct, top_matches = self.fact_checker.check_fact(response, retrieved_docs)
            print(f"\n[INFO] Fact Check: {'PASS' if is_factually_correct else 'FAIL'}")
            if not is_factually_correct:
                print("[INFO] Top Supporting Passages:")
                for context, score in top_matches:
                    print(f"Score: {score:.4f} | Context: {context}")
        
        # Summarize the response (optional)
        if summarize:
            response = self.summarizer.summarize(response)
            print(f"\n[INFO] Summarized Response: {response}")
        
        return response

if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    query = "What are the symptoms of Alzheimer's disease?"
    final_response = rag_pipeline.run(query)
    # print("\nFinal Response:", final_response)
