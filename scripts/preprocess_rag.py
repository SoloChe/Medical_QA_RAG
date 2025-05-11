import json
import argparse

def extract_medical_docs(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                instruction = data.get("instruction", "").strip()
                response = data.get("response", "").strip()
                
                # Save both the instruction and response as separate context pieces
                if instruction:
                    f_out.write(instruction + "\n")
                if response:
                    f_out.write(response + "\n")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    print(f"Extracted contexts saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/processed/medmcqa_train.jsonl", help="Path to the MedMCQA data file")
    parser.add_argument("--output_dir", type=str, default="./data/rag/rag_data.txt", help="Directory to save processed data")
    args = parser.parse_args()
    
    extract_medical_docs(args.input_dir, args.output_dir)
