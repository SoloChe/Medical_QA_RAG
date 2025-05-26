import os
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, ".."))
import torch
import argparse
from agents.rag_pipeline import RAGPipeline, NO_RAGPipeline
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import time
import logging 
import re
from utils import str_to_bool


def extract_prediction(llm_output):
    match = re.search(r'"answer_choice"\s*:\s*"([^"]+)"', llm_output)
    if match:
        answer = match.group(1)
        return answer[0]
    else:
        return "unknown"
    
def load_pipeline(RAG=True, use_corrector=False, use_ranker=False, device="cpu"):
    if RAG:
        pipeline = RAGPipeline(use_ranker=use_ranker, 
                               use_corrector=use_corrector, 
                               device=device)
    else:
        pipeline = NO_RAGPipeline(device=device)
    return pipeline

def get_response(pipeline, question, options, top_k_ret=5, max_new_tokens_gen=2000, do_sample_gen=False):
    response = pipeline.run(question,
                            options,
                            top_k_ret=top_k_ret, # this will not work for NO_RAGPipeline 
                            max_new_tokens_gen=max_new_tokens_gen, 
                            do_sample_gen=do_sample_gen)
    return response


def main(args):
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = os.path.join(args.log_dir, f"eval_{args.eval_name}_{cur_time}.log")
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    
    logger.info(f"Loading RAG pipeline: {args.rag}")
    rag = load_pipeline(RAG=args.rag, 
                        use_ranker=args.use_ranker, 
                        use_corrector=args.use_corrector,
                        device=device)
    eval_dataset = load_dataset("json", data_files=args.eval_file, split='train')
    
    logger.info("Running evaluation...")
    predictions = []
    references = []
    count_unknown = 0
    for count, sample in enumerate(eval_dataset):
       
        question = sample.get("question")
        options = sample.get("options")
        answer = sample.get("answer_idx") 
        
        logger.info(f"Question: {question}")
        logger.info(f"Options: {options}")

        if args.rag and args.use_corrector:
            status, pred_raw, critique, revised_pred_raw = get_response(rag, question, options, top_k_ret=args.top_k_ret)
            if revised_pred_raw:
                logger.info(f"Pass: {status}")
                logger.info(f"Initial Prediction: {pred_raw}")
                logger.info(f"Critique: {critique}")
                logger.info(f"Revised Prediction: {revised_pred_raw}")
                pred_raw = revised_pred_raw
            else:
                logger.info(f"Pass: {status}")
                logger.info(f"Initial Prediction: {pred_raw}")
                logger.info(f"Critique: {critique}")
        else:
            pred_raw = get_response(rag, question, options,  top_k_ret=args.top_k_ret)
            logger.info(f"Prediction: {pred_raw}")
        
        
        
        pred_label = extract_prediction(pred_raw)
        if pred_label == "U": count_unknown += 1
        
        logger.info(f"Answer: {answer}, Prediction: {pred_label}")
        logger.info("=" * 50)
    
        predictions.append(pred_label)
        references.append(answer)
        
        if (count+1) % 10 == 0:
            logger.info(f"+" * 100)
            logger.info(f"Evaluated {len(predictions)} samples.")
            logger.info(f"Count Unknown: {count_unknown}")
            acc = accuracy_score(references, predictions)
            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"+" * 100)
    
    Final_acc = accuracy_score(references, predictions)
    logger.info(f"Number of samples: {len(predictions)}")
    logger.info(f"Final Accuracy: {Final_acc:.4f}")
    logger.info(f"Count Unknown: {count_unknown}")
    
    logger.info(f"Evaluation completed successfully with args {args}.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, default='MedQA_v2', help='Name of the evaluation dataset')
    parser.add_argument('--eval_file', type=str, default='./data/processed/MedQA.jsonl', help='Path to the eval file')
    parser.add_argument('--log_dir', type=str, default='./logs_eval', help='Directory to save logs')
    parser.add_argument('--rag', type=str_to_bool, default=True, help='Use RAG pipeline if True, else use NO_RAG pipeline')
    parser.add_argument('--top_k_ret', type=int, default=10, help='Top K passages to retrieve in RAG pipeline')
    parser.add_argument('--use_corrector', type=str_to_bool, default=True, help='Use corrector in RAG pipeline')
    parser.add_argument('--use_ranker', type=str_to_bool, default=True, help='Use ranker in RAG pipeline')
    args = parser.parse_args()
    
    main(args)
    
  