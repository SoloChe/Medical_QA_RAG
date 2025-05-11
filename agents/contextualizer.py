import os
from transformers import AutoTokenizer

class Contextualizer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", max_length=512, top_k=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        self.top_k = top_k

    def format_context(self, query, passages):
        # Limit to top-k passages
        top_passages = passages[:self.top_k]
        # Combine the question and top-k passages
        context = f"QUESTION: {query}\nCONTEXT:\n"
        context += "\n".join(top_passages)
        return context

    def truncate_context(self, text):
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        # Decode back to text
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    def build_input(self, query, passages):
        # Format the context
        context = self.format_context(query, passages)
        # Truncate if too long
        return self.truncate_context(context)

if __name__ == "__main__":
    # Sample usage
    contextualizer = Contextualizer()
    query = "What are the symptoms of Alzheimer's disease?"
    passages = [
        "Alzheimer's disease is a progressive neurological disorder.",
        "Early symptoms include memory loss and confusion.",
        "Risk factors include age, family history, and genetics."
    ]
    full_input = contextualizer.build_input(query, passages)
    print(full_input)