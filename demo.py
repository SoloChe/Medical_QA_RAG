import torch
import argparse
from agents.rag_pipeline import RAGPipeline


def get_response(query, top_k_ret=3, max_new_tokens_gen=1000, do_sample_gen=False):
    pipeline = RAGPipeline(top_k_con=5)
    response = pipeline.run(query, 
                            top_k_ret=top_k_ret, 
                            max_new_tokens_gen=max_new_tokens_gen, 
                            do_sample_gen=do_sample_gen)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Query to ask the model')
    args = parser.parse_args()
    
    response = get_response(args.query)
    print(f"Response: {response}")