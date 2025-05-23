from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
class Generator:
    def __init__(self, base_model_name, adapter_dir, device="cpu"):
        
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, device=device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

        if adapter_dir:
            # Load the LoRA adapter weights
            self.model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
        else:
            # Load the base model without LoRA
            self.model = base_model.to(device)
        
        self.model.eval()
        self.device = device

    def generate(self, prompt, max_new_tokens=1000, do_sample=False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad(): # Disable gradient computation for inference
            
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    


if __name__ == "__main__":
    # Test the generator
    generator = Generator(base_model_name="mistralai/Mistral-7B-Instruct-v0.2", adapter_dir=None)
    prompt = "<s>[INST] Instruction: Answer the question based on the context.\nQUESTION: What are the symptoms of Alzheimer's disease?\nCONTEXT:\nAlzheimer's disease is a progressive neurological disorder.\nEarly symptoms include memory loss and confusion. [/INST]"
    response = generator.generate(prompt)
    print(response)
