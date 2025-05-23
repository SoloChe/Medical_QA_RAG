from .retriever import FAISSRetriever
from .contextualizer import Contextualizer
from .generator import Generator
from .fact_checker import FactChecker
from .summarizer import Summarizer
from .corrector import Corrector
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGPipeline:
    def __init__(self,
                 retriever_path="./data/KB_books",
                 retriever_index_path="./data/KB_books/books_index.faiss", 
                 LLM_name="mistralai/Mistral-7B-Instruct-v0.2",
                 generator_model_dir=None,
                 use_corrector=True,
                 fact_checker_model=None, # "all-mpnet-base-v2",
                 summarizer_model=None, # "facebook/bart-large-cnn",
                 device="cpu"):
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        LLM = AutoModelForCausalLM.from_pretrained(LLM_name, trust_remote_code=True).to(device)
        LLM.eval()  
        
        self.retriever = FAISSRetriever(docs_path=retriever_path, index_path=retriever_index_path, device=device)
        self.contextualizer = Contextualizer(tokenizer)
        self.generator = Generator(LLM, tokenizer, adapter_dir=generator_model_dir)
        self.corrector = Corrector(LLM, tokenizer) if use_corrector else None
        
        # Optional components
        self.fact_checker = FactChecker(model_name=fact_checker_model, device=device) if fact_checker_model else None
        self.summarizer = Summarizer(model_name=summarizer_model, device=device) if summarizer_model else None
            
        

    def run(self, question, options=None, top_k_ret=5,
            max_new_tokens_gen=1000, do_sample_gen=False):
        # Retrieve top-k passages
        retrieved_docs = [doc for doc in self.retriever.retrieve(question, top_k=top_k_ret)]
        # print(f"\n[INFO] Retrieved {len(retrieved_docs)} documents")
        
        # Contextualize the input
        context_input = self.contextualizer.build_input(question, retrieved_docs, options, top_k=top_k_ret)
        # print(f"\n[INFO] Contextualized Input:\n{context_input}...")
        
        # Generate the response
        response = self.generator.generate(context_input, 
                                           max_new_tokens=max_new_tokens_gen,
                                           do_sample=do_sample_gen)
        # print(f"\n[INFO] Generated Response:\n{response}")
        
        if self.corrector:
            status, response, critique, revised_response = self.corrector.correct(question, retrieved_docs, options, response)
        
        # Fact check the response
        if self.fact_checker:
            is_factually_correct, top_matches = self.fact_checker.check_fact(response, retrieved_docs)
            print(f"\n[INFO] Fact Check: {'PASS' if is_factually_correct else 'FAIL'}")
            if not is_factually_correct:
                print("[INFO] Top Supporting Passages:")
                for context, score in top_matches:
                    print(f"Score: {score:.4f} | Context: {context}")
        
        # Summarize the response (optional)
        if self.summarizer:
            response = self.summarizer.summarize(response)
            print(f"\n[INFO] Summarized Response: {response}")
        
        if self.corrector:
            return status, response, critique, revised_response
        return response

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag_pipeline = RAGPipeline(device=device)
    question = "What are the symptoms of Alzheimer's disease?"
    options = "A: Memory loss, B: Confusion, C: Both A and B, D: None of the above"
    final_response = rag_pipeline.run(question, options, top_k_ret=5)
    print("\nFinal Response:", final_response)
    
    # import json
    # response = json.loads(final_response)
    # Final_Choice = response.get("answer_choice", "unknown")
    # print("\nFinal Choice:", Final_Choice)
