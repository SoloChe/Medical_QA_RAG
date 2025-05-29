import json
import os
import argparse

# TODO: prompt design for instruct model


def clean_text(text):
    return text.replace("\n", " ").replace("\t", " ").strip()


def process_pubmedqa(input_file, output_file):
    # Process the nested JSON structure
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        data = json.load(infile)

        for pmid, sample in data.items():
            # Extract question, long answer, and contexts
            question = sample.get("QUESTION", "").strip()
            contexts = sample.get("CONTEXTS", [])
            long_answer = sample.get("LONG_ANSWER", "").strip()
            short_answer = sample.get("final_decision")

            # Process long answer
            # join all contexts into a single string
            contexts = "\n".join(contexts)
            contexts = contexts.strip()
            entry = {
                "id": pmid,
                "question": question,
                "contexts": contexts,
                "long_answer": long_answer,
                "short_answer": short_answer,
            }
            outfile.write(json.dumps(entry) + "\n")

    print(f"Processed chunks saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubMedQA data")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw/PubMedQA/ori_pqal.json",
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed/PubMedQA.jsonl",
        help="Path to save the processed JSONL file",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Run the processing function
    process_pubmedqa(args.input_dir, args.output_dir)
