import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from liquid import Template
from pydantic import BaseModel, ValidationError

class CritiqueSchema(BaseModel):
    comments: str
    missing_context: str
    status: bool

class Corrector:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device=device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
    def _generate(self, prompt, max_new_tokens=1000):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    
    def _get_prompt(self, task: str, question: str, context: str, answer: str = "", critique: str = "") -> str:
        
        templates = {
            "generate": None,
            "critique": None,
            "revise": None
            }
        
        if task not in templates:
            raise ValueError("Unknown task type.")
        
        return templates[task].render({
            "question": question,
            "context": context,
            "answer": answer,
            "critique": critique
        })

    def _correct(self, question, context, answer):
        critique_prompt = self._get_prompt("critique", question, context, answer)
        critique = self._generate(critique_prompt)

        if any(x in critique.lower() for x in ["no issues", "correct", "nothing to fix", "no further changes"]):
            return answer, critique, 0.0, True

        revise_prompt = self._get_prompt("revise", question, context, answer, critique)
        revised = self._generate(revise_prompt)
        score = self._edit_distance(answer, revised)
        return revised, critique, score, revised.strip() == answer.strip()

    def multi_round_correct(self, question, context, max_rounds=2):
        history = []
        initial_prompt = self._get_prompt("generate", question, context)
        answer = self._generate(initial_prompt)

        for round_num in range(1, max_rounds + 1):
            revised, critique, score, stop = self._correct(question, context, answer)
            history.append({
                "Round": round_num,
                "Answer": answer,
                "Critique": critique,
                "Revised Answer": revised,
                "Edit Distance Score": round(score, 3)
            })
            if stop:
                break
            answer = revised

        return history
