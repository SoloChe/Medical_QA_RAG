from liquid import Template
# from https://github.com/SoloChe/MedRAG/blob/main/src/template.py

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a binary-choice or multi-choice medical question using the relevant documents. Please first think step-by-step and then choose only one answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer. If you are not sure, select the most possible option.'''
general_medrag = Template(
            '''
            Here are the relevant documents (most relevant first):
            {{context}}

            Here is the question:
            {{question}}

            Here are the potential choices:
            {{options}}

            Please think step-by-step and generate your output in json:
            '''
)

general_med_system = '''You are a helpful medical expert, and your task is to answer a binary-choice or multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer. If you are not sure, select the most possible option.'''
general_med = Template(
            '''
            Here is the question:
            {{question}}

            Here are the potential choices:
            {{options}}

            Please think step-by-step and generate your output in json:
            '''
)


general_medrag_system_free = '''You are a helpful medical expert, and your task is to answer medical question using the relevant documents. Please first think step-by-step and then answer the question. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer": Str{}}.'''
general_medrag_free = Template(
            '''
            Here are the relevant documents (most relevant first):
            {{context}}

            Here is the question:
            {{question}}

            Please think step-by-step and generate your output in json:
            '''
)


question_parse_system = '''You are a helpful medical expert. Your task is to decompose the given medical question into sub-questions that can help clarify or solve the problem. Then, list sub-questions in importance order (most important first). Think step-by-step before answering. Format your output as a valid JSON object using double quotes and no trailing commas:
{
  "sub_questions": [<list of questions>]
}
'''
question_parse = Template(
            '''
            Here is the question:
            {{question}}

            Please think step-by-step and generate your output in json:
            '''
)

general_critique_system = '''You are a senior medical expert and your task is to critique the answer of a given binary-choice or multi-choice medical question and the corresponding step-by-step thinking underlying it based on the given context. Please point out any factual errors, missing information, or weaknesses. Organize your output in a json formatted as Dict{"critique": Str{}, "missing_context": Str{}, "status": bool{True/False}}. If the answer is correct, please output "True" else "False" in "status". If more context is needed, please output the queries for retriever else output "None" in "missing_context".'''
general_critique = Template(
            '''
            Here are the relevant documents (most relevant first):
            {{context}}

            Here is the question:
            {{question}}
            
            Here are the potential choices:
            {{options}}
            
            Here are the answer and step_by_step_thinking in json format:
            {{LLM_answer}}
            
            Please think step-by-step and generate your output in json:
            '''
)

general_revise_system = '''You are a senior medical expert and your task is to revise the answer of a given binary-choice or multi-choice medical question based on the given retrieved context and comments from another medical expert to make it medically accurate and comprehensive. Please first think step-by-step and then choose only one answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer. If you are not sure, If you are not sure, select the most possible option.'''
general_revise = Template(
            '''
            Here are the relevant documents (most relevant first):
            {{context}}

            Here is the question:
            {{question}}
            
            Here are the potential choices:
            {{options}}
            
            Here is the answer and step_by_step_thinking in json format:
            {{LLM_answer}}
            
            Here are the comments from another medical expert in json format:
            {{LLM_critique}}
            
            Please think step-by-step and generate your output in json:
            '''
)
