import os
import json
import argparse
import zipfile
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def unzip_data(data_zip_path, extract_dir):
    print(f"Extracting {data_zip_path}...")
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def preprocess_medmcqa(data_dir, output_dir, val_ratio=0.1):
    print("Processing MedMCQA...")
    # Load the MedMCQA dataset from the extracted directory
    dataset_path = os.path.join(data_dir, "raw")
    dataset = load_dataset("json", data_files=os.path.join(dataset_path, "*.json"), split="train")
    dataset_list = [sample for sample in dataset]
    
    # Split the dataset into train and validation sets
    train_data, val_data = train_test_split(dataset_list, test_size=val_ratio, random_state=42)
    
    def save_data(data, output_file):
        count = 0
        with open(output_file, "w") as f:
            for sample in tqdm(data, desc=f"Writing {output_file}"):
                question = sample.get("question", "").strip()
                choices = [sample.get("opa", ""), sample.get("opb", ""), sample.get("opc", ""), sample.get("opd", "")]
                answer_key = sample.get("cop", None)
                
                # Skip samples with missing or invalid answer keys
                if answer_key is None or not isinstance(answer_key, int) or not (1 <= answer_key <= 4):
                    print(f"Invalid answer key for question: {question} Choices: {choices} Answer key: {answer_key}")
                    continue
                
                # Convert 1-based to 0-based index
                answer_index = answer_key - 1
                answer = choices[answer_index].strip()
                
                # Skip if any choice is empty
                if any(c.strip() == "" for c in choices):
                    print(f"Empty choice detected for question: {question} Choices: {choices}")
                    continue
                
                instruction = f"Q: {question} Options: {', '.join(choices)}"
                response = f"A: {answer}"
                f.write(json.dumps({"instruction": instruction, "response": response}) + "\n")
                count += 1
        print(f"Saved {count} samples to {output_file}")
    
    # Save training and validation data
    save_data(train_data, os.path.join(output_dir, "medmcqa_train.jsonl"))
    save_data(val_data, os.path.join(output_dir, "medmcqa_val.jsonl"))
    
    print(f"Total training samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_zip", type=str, default="./data/raw/data.zip", help="Path to the MedMCQA data zip file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to extract raw data to")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Directory to save processed data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Unzip the data first
    unzip_data(args.data_zip, args.data_dir)

    # Process the MedMCQA data with train/val split
    preprocess_medmcqa(args.data_dir, args.output_dir, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
