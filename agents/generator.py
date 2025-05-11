import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

class Generator:
    def __init__(self, model_name="./saved_models/lora_finetuned", device="cuda", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load base model and fine-tuned LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base_model, model_name).to(device)
        
        self.max_length = max_length
        self.device = device

    def generate(self, prompt, max_length=None, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_length=max_length or self.max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Sample usage
    # TODO: need to check if the model is loaded correctly
    generator = Generator(model_name="./saved_models/lora_finetuned")
    prompt = "QUESTION: What are the symptoms of Alzheimer's disease?\nCONTEXT:\nAlzheimer's disease is a progressive neurological disorder.\nEarly symptoms include memory loss and confusion.\nRisk factors include age, family history, and genetics."
    response = generator.generate(prompt)
    print(response)
