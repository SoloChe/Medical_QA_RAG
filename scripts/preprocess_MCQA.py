import os
import json
import argparse
import zipfile
import random
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Diverse instruction templates
PROMPT_TEMPLATES = [
    "Instruction: Answer the following question.\nQuestion: {question}\nOptions: {options}\nAnswer:",
    "Choose the best answer.\nQ: {question}\nOptions: {options}\nA:",
    "Select the correct answer: {question} Options: {options} A:",
    "Given the question, select the most appropriate option.\nQ: {question}\nOptions: {options}\nA:",
    "You're a helpful assistant. Please answer this medical question.\nQ: {question}\nChoices: {options}\nA:",
    "Solve this medical multiple-choice question:\n{question}\nChoices: {options}\nAnswer:",
    "You are taking a medical exam.\nQ: {question}\nOptions: {options}\nSelect the correct answer:",
    "MCQ Test:\nQuestion: {question}\nOptions: {options}\nCorrect Answer:",
    "User asks: {question}\nOptions: {options}\nYour answer:",
    "Patient question: {question}\nAvailable answers: {options}\nChoose:",
    "AI Quiz Helper:\nPrompt: {question}\nOptions: {options}\nPick one:",
    "Q: {question}\nChoices: {options}\nA:"
]

# Optional contextual role-based prefixes
INSTRUCTION_PREFIXES = [
    "You are an AI medical assistant.",
    "You are tutoring a medical student.",
    "This is a USMLE preparation question.",
    "You are solving a diagnostic MCQ.",
    "Medical expert task:"
]

# Keyword-based synonym replacements
KEYWORDS = ["question", "answer", "choose", "correct"]
REPLACEMENTS = {
    "question": ["problem", "query"],
    "answer": ["response", "reply"],
    "choose": ["select", "pick"],
    "correct": ["right", "accurate"]
}

#TODO: Implement the function to add explanations
def adding_explanations(text):
    pass

def apply_synonym_replacement(template):
    replaced_text = template
    placeholders = ["{question}", "{options}"]
    for word in KEYWORDS:
        if word in template.lower():
            synonyms = REPLACEMENTS.get(word, [])
            if synonyms:
                for variant in [word, word.capitalize()]:
                    if variant in replaced_text:
                        synonym = random.choice(synonyms)
                        replaced_text = replaced_text.replace(variant, synonym)
    for ph in placeholders:
        if ph not in replaced_text:
            return template  # Fallback if corrupted
    return replaced_text


def unzip_data(data_zip_path, extract_dir):
    print(f"Extracting {data_zip_path}...")
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def preprocess_medmcqa(data_dir, output_dir, val_ratio=0.1):
    print("Processing MedMCQA...")
    dataset_path = os.path.join(data_dir, "raw")
    dataset = load_dataset("json", data_files=os.path.join(dataset_path, "*.json"), split="train")
    dataset_list = [sample for sample in dataset]
    train_data, val_data = train_test_split(dataset_list, test_size=val_ratio, random_state=42)

    def save_data(data, output_file):
        count = 0
        with open(output_file, "w") as f:
            for sample in tqdm(data, desc=f"Writing {output_file}"):
                question = sample.get("question", "").strip()
                choices = [sample.get("opa", ""), sample.get("opb", ""), sample.get("opc", ""), sample.get("opd", "")]
                answer_key = sample.get("cop", None)

                if answer_key is None or not isinstance(answer_key, int) or not (1 <= answer_key <= 4):
                    continue
                if any(c.strip() == "" for c in choices):
                    continue

                answer_index = answer_key - 1
                answer = choices[answer_index].strip()

                indexed_choices = list(enumerate(choices))
                random.shuffle(indexed_choices)
                choices = [c for _, c in indexed_choices]
                new_answer_index = [i for i, (_, c) in enumerate(indexed_choices) if c.strip() == answer][0]
                answer = choices[new_answer_index].strip()

                # Template with synonyms and prefix
                template = apply_synonym_replacement(random.choice(PROMPT_TEMPLATES))
                prefix = random.choice(INSTRUCTION_PREFIXES)
                prompt_text = f"{prefix}\n{template.format(question=question, options=', '.join(choices)).strip()}"

                instruction = f"<s>[INST] {prompt_text} [/INST]"
                response = f" {answer} </s>"
                text = f"{instruction}{response}"
                
                # Optional: Add explanations
                explanation = adding_explanations(f'{prompt_text} {answer}')
            
                f.write(json.dumps({"instruction": instruction, "response": response, "text": text}) + "\n")
                count += 1
        print(f"Saved {count} samples to {output_file}")

    os.makedirs(output_dir, exist_ok=True)
    save_data(train_data, os.path.join(output_dir, "mistral_train.jsonl"))
    save_data(val_data, os.path.join(output_dir, "mistral_val.jsonl"))

    print(f"Total training samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_zip", type=str, default="./data/raw/data.zip", help="Path to the MedMCQA data zip file")
    parser.add_argument("--data_dir", type=str, default="./data/raw/MCQA", help="Directory to extract raw data to")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Directory to save processed data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    unzip_data(args.data_zip, args.data_dir)
    preprocess_medmcqa(args.data_dir, args.output_dir, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
