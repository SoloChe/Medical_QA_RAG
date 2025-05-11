from sentence_transformers import SentenceTransformer, util
import torch

class FactChecker:
    def __init__(self, model_name="all-mpnet-base-v2", threshold=0.7, device="cpu"):
        self.model = SentenceTransformer(model_name).to(device)
        self.threshold = threshold
        self.device = device

    def check_fact(self, response, context_list, top_k=3):
        # Encode the response and context passages
        response_emb = self.model.encode(response, convert_to_tensor=True, device=self.device)
        context_embs = self.model.encode(context_list, convert_to_tensor=True, device=self.device)
        
        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(response_emb, context_embs).squeeze()
        top_indices = torch.topk(similarities, top_k).indices
        top_scores = similarities[top_indices].cpu().numpy()
        top_contexts = [context_list[i] for i in top_indices.cpu().numpy()]
        
        # Check if any context has a high enough similarity
        is_factually_correct = any(score > self.threshold for score in top_scores)
        
        return is_factually_correct, list(zip(top_contexts, top_scores))

if __name__ == "__main__":
    fact_checker = FactChecker()
    response = "Alzheimer's disease typically causes memory loss and confusion."
    context_list = [
        "Alzheimer's disease is a progressive neurological disorder.",
        "Early symptoms include memory loss and confusion.",
        "Risk factors include age, family history, and genetics."
    ]
    is_correct, top_matches = fact_checker.check_fact(response, context_list)
    print(f"Factually Correct: {is_correct}")
    print("Top Matches:")
    for context, score in top_matches:
        print(f"Score: {score:.4f} | Context: {context}")
