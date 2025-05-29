# Filename: split_books_combined.py

import os
import glob
import json
import argparse
import spacy
from tqdm import tqdm


def semantic_chunk(text, max_words=200, overlap=30):
    """
    Split text into semantically coherent chunks with overlap using spaCy.
    """
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in doc.sents:
        words = sent.text.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] + words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    book_paths = glob.glob(os.path.join(args.input_dir, "*.txt"))

    for book_path in tqdm(book_paths, desc="Processing books"):
        book_name = os.path.basename(book_path).replace(".txt", "")
        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = text.replace("\n", " ").strip()
        chunks = semantic_chunk(text, args.chunk_size, args.overlap)

        output_path = os.path.join(args.output_dir, f"{book_name}_chunks.jsonl")
        with open(output_path, "w", encoding="utf-8") as out_file:
            for i, chunk in enumerate(chunks):
                record = {"source": book_name, "chunk_id": i, "text": chunk}
                out_file.write(json.dumps(record) + "\n")

    print(f"\nCombined file saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine and split books into semantic chunks for RAG."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw/MedQA/data_clean/textbooks/en",
        help="Directory containing .txt books",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/KB_books",
        help="Directory to save combined JSONL",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=200, help="Max words per chunk"
    )
    parser.add_argument(
        "--overlap", type=int, default=30, help="Overlap words between chunks"
    )
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1_000_000_000
    main(args)
