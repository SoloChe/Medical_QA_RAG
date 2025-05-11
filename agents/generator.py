import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Generator:
    def __init__(self, base_model_name="mistralai/Mistral-7B-v0.1", adapter_dir="./saved_models/lora_finetuned", device="cpu"):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

        # Load the LoRA adapter weights
        self.model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)

        self.device = device

    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the generator
    generator = Generator(base_model_name="mistralai/Mistral-7B-v0.1", adapter_dir="./saved_models/lora_finetuned/checkpoint-69000")
    prompt = "QUESTION: What are the symptoms of Alzheimer's disease?\nCONTEXT:\nAlzheimer's disease is a progressive neurological disorder.\nEarly symptoms include memory loss and confusion."
    response = generator.generate(prompt)
    print(response)
