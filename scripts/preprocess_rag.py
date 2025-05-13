import json
import os
import argparse

def clean_text(text):
    return text.replace("\n", " ").replace("\t", " ").strip()

def process_pubmedqa(input_file, output_file):
    examples = []

    # Process the nested JSON structure
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        data = json.load(infile)

        for pmid, sample in data.items():
            # Extract question, long answer, and contexts
            question = sample.get("QUESTION", "").strip()
            long_answer = sample.get("LONG_ANSWER", "").strip()
            contexts = sample.get("CONTEXTS", [])

            # Process long answer
            if long_answer:
                entry = {
                    "id": pmid,
                    "text": long_answer,
                    "source": "PubMedQA"
                }
                outfile.write(json.dumps(entry) + "\n")
                if len(examples) < 2:
                    examples.append(entry)

            # Process contexts
            for context in contexts:
                context = context.strip()
                if 50 < len(context) < 300:  # Chunk length limits
                    entry = {
                        "id": pmid,
                        "text": context,
                        "source": "PubMedQA"
                    }
                    outfile.write(json.dumps(entry) + "\n")
                    if len(examples) < 2:
                        examples.append(entry)

    print(f"Processed chunks saved to {output_file}")
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubMedQA data")
    parser.add_argument("--input_dir", type=str, default="./data/rag/ori_pqal.json", help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="./data/rag/rag.json", help="Path to save the processed JSONL file")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Run the processing function
    examples = process_pubmedqa(args.input_dir, args.output_dir)

    # Print some examples for verification
    print("Sample Processed Entries:")
    for example in examples:
        print(json.dumps(example, indent=2))
