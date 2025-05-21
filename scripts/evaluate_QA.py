import os
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, ".."))
import torch
import argparse
from agents.rag_pipeline import RAGPipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from datasets import load_dataset
import time
import logging 

def normalize_prediction_yesno(pred):
    pred = pred.lower()
    if "yes" in pred:
        return "yes"
    elif "no" in pred:
        return "no"
    else:
        return "unknown"
    
def normalize_prediction_abcde(pred):
    pred = pred.lower()
    if "a" in pred:
        return "A"
    elif "b" in pred:
        return "B"
    elif "c" in pred:
        return "C"
    elif "d" in pred:
        return "D"
    elif "e" in pred:
        return "E"
    else:
        return "unknown"

def load_pipeline(RAG=True, choice=False, multi_choice=False, device="cpu"):
    if RAG:
        pipeline = RAGPipeline(top_k_con=5, choice=choice, multi_choice=multi_choice, device=device)
    else:
        # TODO: Implement the pipeline without RAG
        pass
    return pipeline

def get_response(pipeline, query, top_k_ret=5, max_new_tokens_gen=1000, do_sample_gen=False):
    # from top_k_ret embeddings select the most relevant top_k_con
    response = pipeline.run(query, 
                            top_k_ret=top_k_ret, 
                            max_new_tokens_gen=max_new_tokens_gen, 
                            do_sample_gen=do_sample_gen,
                            summarize=False, 
                            fact_check=False)
    return response

def evaluate(y_true, y_pred):
    """Calculate accuracy, F1, recall (sensitivity), and specificity."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="yes", average="binary")
    recall = recall_score(y_true, y_pred, pos_label="yes", average="binary")  # Sensitivity

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["no", "yes"]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        specificity = 0.0

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "specificity": specificity
    }

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
    
    # Load the RAG pipeline
    if args.eval_name == 'PubMedQA':
        choice = True
        multi_choice = False
        normalize_prediction = normalize_prediction_yesno
    elif args.eval_name == 'MedMCQA' or args.eval_name == 'MedQA':
        choice = False
        multi_choice = True
        normalize_prediction = normalize_prediction_abcde
    
    rag = load_pipeline(RAG=True, choice=choice, multi_choice=multi_choice, device=device)
    eval_dataset = load_dataset("json", data_files=args.eval_file, split='train')
    
    logger.info("Running evaluation...")
    predictions = []
    references = []
    for sample in eval_dataset:
       
        question = sample.get("query") 
        # answer = sample.get("short_answer") # For PubMedQA
        answer = sample.get("answer_idx")  # For and MedQA

        pred_raw = get_response(rag, question)
        pred_label = normalize_prediction(pred_raw)
        predictions.append(pred_label)
        references.append(answer.strip().lower())

        logger.info(f"Question: {question}")
        logger.info(f"Generated: {pred_raw}")
        logger.info(f"Answer: {answer}, Prediction: {pred_label}")
        logger.info("=" * 50)
        
    logger.info(f"Evaluated {len(predictions)} samples.")
    metrics = evaluate(references, predictions)

    logger.info("Evaluation Metrics:")
    for key, val in metrics.items():
        logger.info(f"{key}: {val:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, default='MedQA', help='Name of the evaluation dataset')
    parser.add_argument('--eval_file', type=str, default='./data/processed/MedQA.jsonl', help='Path to the eval file')
    parser.add_argument('--log_dir', type=str, default='./logs_eval', help='Directory to save logs')
    args = parser.parse_args()
    
    main(args)
    
  