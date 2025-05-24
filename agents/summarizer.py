from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Summarizer:
    def __init__(self, model, tokenizer, max_length=1000):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.device = model.device

    def summarize(self, text, max_length=None, min_length=30, do_sample=False, temperature=0.7):
        # Ensure max_length is set
        max_length = max_length or self.max_length
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        summary_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    summarizer = Summarizer()
    text = "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink (atrophy) and brain cells to die. Alzheimer's disease is the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently. Early symptoms include memory loss and confusion."
    summary = summarizer.summarize(text)
    print("Summary:", summary)
