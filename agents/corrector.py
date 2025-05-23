import torch
from .template import *
import re

def extract_status(critique):
    match = re.search(r'"status"\s*:\s*(true|false|True|False)', critique)
    if match:
        answer = match.group(1)
        return answer
    else:
        return "unknown"
class Corrector:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = self.model.device
        
    def _generate(self, prompt, max_new_tokens=2000, do_sample=False):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad(): 
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def _get_prompt(self, task, question, context, options, answer, critique=None):
        if task == "critique":
            prompt_critique = general_critique.render(context=context, question=question, options=options, LLM_answer=answer)
            messages=[
                    {"role": "system", "content": general_critique_system},
                    {"role": "user", "content": prompt_critique}
            ]
        elif task == "revise":
            prompt_revise = general_revise.render(context=context, question=question, options=options, LLM_answer=answer, LLM_critique=critique)
            messages=[
                    {"role": "system", "content": general_revise_system},
                    {"role": "user", "content": prompt_revise}
            ]
        else:
            raise ValueError("Unknown task type.")
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def correct(self, question, context, options, answer):
        critique_prompt = self._get_prompt("critique", question, context, options, answer)
        critique = self._generate(critique_prompt)
        status = extract_status(critique)
        
        if status:
            return status, answer, critique, None

        revise_prompt = self._get_prompt("revise", question, context, options, answer, critique)
        revised = self._generate(revise_prompt)
        return status, answer, critique, revised

    # def multi_round_correct(self, question, context, max_rounds=2):
    #     history = []
    #     initial_prompt = self._get_prompt("generate", question, context)
    #     answer = self._generate(initial_prompt)

    #     for round_num in range(1, max_rounds + 1):
    #         revised, critique, score, stop = self._correct(question, context, answer)
    #         history.append({
    #             "Round": round_num,
    #             "Answer": answer,
    #             "Critique": critique,
    #             "Revised Answer": revised,
    #             "Edit Distance Score": round(score, 3)
    #         })
    #         if stop:
    #             break
    #         answer = revised

    #     return history
