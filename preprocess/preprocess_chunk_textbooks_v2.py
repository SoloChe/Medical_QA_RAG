import os
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from https://github.com/SoloChe/MedRAG/blob/main/src/data/textbooks.py


def ends_with_ending_punctuation(s):
    ending_punctuation = (".", "?", "!")
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


if __name__ == "__main__":

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fdir = "./data/raw/MedQA/data_clean/textbooks/en"
    fnames = sorted(os.listdir(fdir))

    out_dir = "./data/KB_books_v2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(fdir, fname)
        texts = text_splitter.split_text(open(fpath).read().strip())
        saved_text = [
            json.dumps(
                {
                    "chunk_id": "_".join([fname.replace(".txt", ""), str(i)]),
                    "source": fname.strip(".txt"),
                    "text": re.sub("\s+", " ", texts[i]),
                }
            )
            for i in range(len(texts))
        ]
        with open(
            "./data/KB_books_v2/{:s}".format(fname.replace(".txt", ".jsonl")), "w"
        ) as f:
            f.write("\n".join(saved_text))
