from agents.retriever import FAISSRetriever
from agents.contextualizer import Contextualizer
from agents.generator import Generator
from agents.corrector import Corrector
from prompts.template import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Union


class RAGPipeline:
    def __init__(
        self,
        retriever_path: str = "./data/KB_books_v2",
        retriever_index_path: str = "./data/KB_books_v2/books_index.faiss",  # v2: MedCPT-Query-Encoder
        LLM_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        generator_model_dir: str = None,
        use_corrector: bool = False,
        use_ranker: bool = False,
        device: str = "cpu",
    ) -> None:

        tokenizer = AutoTokenizer.from_pretrained(LLM_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        LLM = AutoModelForCausalLM.from_pretrained(LLM_name, trust_remote_code=True).to(
            device
        )
        LLM.eval()

        self.retriever = FAISSRetriever(
            docs_path=retriever_path,
            index_path=retriever_index_path,
            use_ranker=use_ranker,
            device=device,
        )
        self.contextualizer = Contextualizer(tokenizer)
        self.generator = Generator(LLM, tokenizer, adapter_dir=generator_model_dir)
        self.corrector = Corrector(LLM, tokenizer) if use_corrector else None


    def run(
        self,
        question: str,
        options: Optional[str] = None,
        top_k_ret: int = 5,
        max_new_tokens_gen: int = 1000,
        do_sample_gen: bool = False,
    ) -> Union[Tuple[bool, str, str, Optional[str]], Tuple[str, str]]:

        # Retrieve top-k passages
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k_ret)
        # print(f"\n[INFO] Retrieved {len(retrieved_docs)} documents")

        # Contextualize the input
        context_input = self.contextualizer.build_input(
            question, retrieved_docs, options
        )
        # print(f"\n[INFO] Contextualized Input:\n{context_input}...")

        # Generate the response
        response = self.generator.generate(
            context_input, max_new_tokens=max_new_tokens_gen, do_sample=do_sample_gen
        )
        # print(f"\n[INFO] Generated Response:\n{response}")

        if self.corrector:
            status, response, critique, revised_response = self.corrector.correct(
                question, retrieved_docs, options, response
            )

            return status, response, critique, revised_response, retrieved_docs

        return response, retrieved_docs


class NO_RAGPipeline:
    def __init__(
        self, LLM_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "cpu"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_name, trust_remote_code=True
        ).to(device)
        self.model.eval()
        self.device = device

    def run(
        self,
        question: str,
        options: str = None,
        top_k_ret: int = 0,  # placeholder for compatibility
        max_new_tokens_gen: int = 1000,
        do_sample_gen: bool = False,
    ) -> Tuple[str, None]:

        prompt_med = general_med.render(question=question, options=options)
        messages = [
            {"role": "system", "content": general_med_system},
            {"role": "user", "content": prompt_med},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer.encode(inputs, truncation=True, max_length=4000)
        inputs = self.tokenizer.decode(inputs, skip_special_tokens=True)
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_gen,
                do_sample=do_sample_gen,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), None


if __name__ == "__main__":
    # testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    question = "What are the symptoms of Alzheimer's disease?"
    options = "A: Memory loss, B: Confusion, C: Both A and B, D: None of the above"
    rag_pipeline = RAGPipeline(device=device)
    final_response_rag, context = rag_pipeline.run(question, options, top_k_ret=5)
    print("\nFinal Response from RAG Pipeline:\n", final_response_rag)
    print("\nContextualized Input:\n", context)

    # med_pipeline = NO_RAGPipeline(device=device)
    # final_response_med = med_pipeline.run(question, options)
    # print("\nFinal Response from Med Pipeline:\n", final_response_med)

    # import json
    # response = json.loads(final_response)
    # Final_Choice = response.get("answer_choice", "unknown")
    # print("\nFinal Choice:", Final_Choice)
