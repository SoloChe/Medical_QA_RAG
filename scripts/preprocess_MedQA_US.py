import json
import os
import argparse


def clean_text(text):
    return text.replace("\n", " ").replace("\t", " ").strip()

def process_pubmedqa(input_file, output_file):
    # Process the nested JSON structure
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        
        for line in infile:
            sample = json.loads(line.strip()) 
          

      
            # Extract question, long answer, and contexts
            question = sample.get("question").strip()
            options = sample.get("options")
            options = f"A: {options.get('A')}, B: {options.get('B')}, C: {options.get('C')}, D: {options.get('D')}, E: {options.get('E')}"
            answer_idx = sample.get("answer_idx")
            
            question_options = question + "\nOptions:\n" + options
            # print(question_options)
           
            entry = {
                "query": question_options,
                "question": question,
                "options": options,
                "answer_idx": answer_idx,
            }
            outfile.write(json.dumps(entry) + "\n")


    print(f"Processed chunks saved to {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubMedQA data")
    parser.add_argument("--input_dir", type=str, default="./data/raw/MedQA/data_clean/questions/US/test.jsonl", help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="./data/processed/MedQA.jsonl", help="Path to save the processed JSONL file")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    # Run the processing function
    process_pubmedqa(args.input_dir, args.output_dir)

   