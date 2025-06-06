import torch
from prompts.template import *
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Union


class Corrector:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = self.model.device

    def _get_prompt(
        self,
        task: str,
        question: str,
        context: str,
        options: str,
        answer: str,
        critique: Optional[str] = None,
    ) -> str:
        if task == "critique":
            prompt_critique = general_critique.render(
                context=context, question=question, options=options, LLM_answer=answer
            )
            messages = [
                {"role": "system", "content": general_critique_system},
                {"role": "user", "content": prompt_critique},
            ]
        elif task == "revise":
            prompt_revise = general_revise.render(
                context=context,
                question=question,
                options=options,
                LLM_answer=answer,
                LLM_critique=critique,
            )
            messages = [
                {"role": "system", "content": general_revise_system},
                {"role": "user", "content": prompt_revise},
            ]
        else:
            raise ValueError("Unknown task type.")

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def correct(
        self, question: str, context: str, options: str, answer: str
    ) -> Tuple[bool, str, str, Optional[str]]:
        critique_prompt = self._get_prompt(
            "critique", question, context, options, answer
        )
        critique = self._generate(critique_prompt)
        status = self._extract_status(critique)
        # TODO: handle the case
        missing_context = self._extract_missing_context(critique)

        if isinstance(status, bool) and status:
            return status, answer, critique, None
        else:
            revise_prompt = self._get_prompt(
                "revise", question, context, options, answer, critique
            )
            revised = self._generate(revise_prompt)
            return status, answer, critique, revised

    # TODO: Implement multi-round correction
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

    @staticmethod
    def _extract_status(critique: str) -> Union[bool, str]:
        match = re.search(r'"status"\s*:\s*(true|false|True|False)', critique)
        if match:
            value = match.group(1)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                value = value.lower()
                if value in ("true", "1"):
                    return True
                elif value in ("false", "0"):
                    return False
        else:
            return "unknown"

    @staticmethod
    def _extract_missing_context(critique: str) -> Optional[str]:
        match = re.search(r'"missing_documents"\s*:\s*"([^"]+)"', critique)
        missing_context = match.group(1) if match else "None"
        return None if missing_context == "None" else missing_context

    def _generate(
        self, prompt: str, max_new_tokens: int = 2000, do_sample: bool = False
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
