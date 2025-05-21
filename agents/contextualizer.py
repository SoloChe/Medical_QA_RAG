from transformers import AutoTokenizer
from liquid import Template

class Contextualizer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", max_length=3000, top_k=3, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device=device)
        self.max_length = max_length
    

    def format_context(self, passages, top_k):
        # Select top-k passages
        top_passages = passages[:top_k]

        # Combine passages with metadata for citation
        context_passages = []
        for idx, p in enumerate(top_passages):
            source = p.get("source", "unknown")
            chunk_id = p.get("chunk_id", f"chunk_{idx}")
            text = p["text"].strip()
            context_passages.append(f"[{chunk_id} | {source}]: {text}")
            context = "\n\n".join(context_passages)
        return context
    
    def truncate_context(self, text):
        # Tokenize to check length
        text = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        # Decode back to text
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text

    def build_input(self, question, context, options=None, top_k=3):
        # Format the full_prompt
        context = self.format_context(context, top_k)
        full_prompt = self.prepare_prompt(question, context, options, free=(options==None))
        # Truncate if too long
        return self.truncate_context(full_prompt)
    
    def prepare_prompt(self, question, context, options, free=False):
        # from https://github.com/SoloChe/MedRAG/blob/main/src/template.py
        general_medrag_system = '''You are a helpful medical expert, and your task is to answer a binary-choice or multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
        
        general_medrag = Template('''
        Here are the relevant documents:
        {{context}}

        Here is the question:
        {{question}}

        Here are the potential choices:
        {{options}}

        Please think step-by-step and generate your output in json:
        ''')
        
        general_medrag_system_free = '''You are a helpful medical expert, and your task is to answer medical question using the relevant documents. Please first think step-by-step and then answer the question. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer": Str{}}.'''
        
        general_medrag_free = Template('''
        Here are the relevant documents:
        {{context}}

        Here is the question:
        {{question}}

        Please think step-by-step and generate your output in json:
        ''')
        
        if not free:
            prompt_medrag = general_medrag.render(context=context, question=question, options=options)
            messages=[
                    {"role": "system", "content": general_medrag_system},
                    {"role": "user", "content": prompt_medrag}
            ]
        else:
            prompt_medrag = general_medrag_free.render(context=context, question=question)
            messages=[
                    {"role": "system", "content": general_medrag_system_free},
                    {"role": "user", "content": prompt_medrag_free}
            ]
        return messages
    

if __name__ == "__main__":
    # Sample usage
    contextualizer = Contextualizer()
    question = "What are the symptoms of Alzheimer's disease?"
    options = "A: Memory loss, B: Confusion, C: Both A and B, D: None of the above"
    context = [
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
    full_input = contextualizer.build_input(question, context, options=options)
    print(full_input)