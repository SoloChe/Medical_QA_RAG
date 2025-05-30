from transformers import AutoTokenizer
from prompts.template import *
from typing import List, Dict, Optional


class Contextualizer:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 4000) -> None:
        self.max_length = max_length
        self.tokenizer = tokenizer

    def truncate_context(self, text: str) -> str:
        # Tokenize to check length
        text = self.tokenizer.apply_chat_template(
            text, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(
            text, truncation=True, max_length=self.max_length
        )
        # Decode back to text
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    def build_input(
        self, question: str, context: str, options: Optional[str] = None
    ) -> str:
        # Format the full_prompt
        full_prompt = self.prepare_prompt(
            question, context, options, free=(options == None)
        )
        # Truncate if too long
        return self.truncate_context(full_prompt)

    def prepare_prompt(
        self, question: str, context: str, options: str, free: Optional[bool] = False
    ) -> List[Dict[str, str]]:

        if not free:
            prompt_medrag = general_medrag.render(
                context=context, question=question, options=options
            )
            messages = [
                {"role": "system", "content": general_medrag_system},
                {"role": "user", "content": prompt_medrag},
            ]
        else:
            prompt_medrag_free = general_medrag_free.render(
                context=context, question=question
            )
            messages = [
                {"role": "system", "content": general_medrag_system_free},
                {"role": "user", "content": prompt_medrag_free},
            ]
        return messages


if __name__ == "__main__":
    # Sample usage
    question = "What are the symptoms of Alzheimer's disease?"
    options = "A: Memory loss, B: Confusion, C: Both A and B, D: None of the above"

    context = """
    [1970 | Psichiatry_DSM-5]: deficit early in the course might suggest Alzheimer's disease...

    [5538 | Neurology_Adams]: both the genetic aspects and therapeutic measures...

    [1973 | Psichiatry_DSM-5]: and progressive worsening of memory...
    """

    LLM_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(LLM_name, trust_remote_code=True)
    contextualizer = Contextualizer(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    full_input = contextualizer.build_input(question, context, options=options)
    print(full_input)
