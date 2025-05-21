from transformers import AutoTokenizer

class Contextualizer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", max_length=2000, top_k=3, choice=False, multi_choice=False, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device=device)
        self.max_length = max_length
        self.top_k = top_k
        self.choice = choice
        self.multi_choice = multi_choice

    def format_context(self, query, passages):
        # Select top-k passages
        top_passages = passages[:self.top_k]

        # Combine passages with metadata for citation
        context_passages = []
        for idx, p in enumerate(top_passages):
            source = p.get("source", "unknown")
            chunk_id = p.get("chunk_id", f"chunk_{idx}")
            text = p["text"].strip()
            context_passages.append(f"[{chunk_id} | {source}]: {text}")

        # Combine the question and context
        if not self.choice and not self.multi_choice:
            context = f"<s>[INST] Instruction: Answer the question based on the given contexts and explain your reasoning. \nQUESTION: {query}\nCONTEXT:\n"
            context += "\n\n".join(context_passages)
            context += "[/INST]"
        elif self.choice:
            context = f"<s>[INST] Instruction: Answer the question based on the given contexts. Output 'yes' or 'no' only. \nQUESTION: {query}\nCONTEXT:\n"
            context += "\n\n".join(context_passages)
            context += "[/INST]"
        elif self.multi_choice:
            context = f"<s>[INST] Instruction: Answer the question based on the given contexts and options. Output answer index only. \nQUESTION: {query}\nCONTEXT:\n"
            context += "\n\n".join(context_passages)
            context += "[/INST]"
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
        {
            "text": "Alzheimer's disease is a progressive neurological disorder.",
            "source": "MedGuide2023",
            "chunk_id": "chunk_001"
        },
        {
            "text": "Early symptoms include memory loss and confusion.",
            "source": "MedGuide2023",
            "chunk_id": "chunk_002"
        },
        {
            "text": "Risk factors include age, family history, and genetics.",
            "source": "CDC_Report_2022",
            "chunk_id": "chunk_045"
        }
    ]
    full_input = contextualizer.build_input(query, passages)
    print(full_input)