# Medical QA RAG System (updating)
This repository contains a medical question-answering system that utilizes a Retrieval-Augmented Generation (RAG) approach. The system is designed to answer medical questions by retrieving relevant documents from a knowledge base consists of textbooks used in ([MedQA](https://github.com/jind11/MedQA)) and generating responses from multi-agent pipeline. Finally, the system can be deployed on AWS using Docker and SageMaker. 

The current knowledge base is for testing only. The final knowledge base will be a collection of medical documents, including PubMed articles, clinical guidelines, and other relevant resources. The system is designed to be extensible and can be easily adapted to different domains or datasets.

[05/08/2025 update]: Finished the training scripts of LoRa and RAG pipeline.

[05/09/2025 update]: Debugged the training scripts and the RAG pipeline. The system is now fully functional.

[05/10/2025 update]: Finished the training scripts of LoRa+PPO. Finished the preparation of the deployment scripts except `predictor.py`. 

[05/12/2025 update]: Preprocessed PubMedQA for knowledge base. Changed the retriever from BM25 to FAISS+BioBERT. 

[05/13/2025 update]: Changed model to `mistralai/Mistral-7B-Instruct-v0.2` for generator and re-preprocessed data (adding tags) for fine-tuning.

[05/17/2025 update]: Preprocessed textbooks for knowledge base using semantic split with overlap.

[05/19/2025 update]: Finished the initial version of Medical QA RAG system. run `demo.py` for demo.

[05/19/2025 update]: Finished the evaluation pipeline for RAG system.

[Next]: Working on retrieval part and will add more agents, e.g., reasoning and tool use. Exploring the reinforcement learning (PPO) for alignment.

## Agent Pipeline
1. Retriever Model: FAISS+BioBERT
2. Contextualizer Model: mistralai/Mistral-7B-Instruct-v0.2
3. Generator Model: mistralai/Mistral-7B-Instruct-v0.2 (optional fine-tuning)
4. Fact-Checker Model: all-mpnet-base-v2 (optional)
5. Summarizer Model: facebook/bart-large-cnn (optional)

## Fine-tuning
The generator model is fine-tuned using LoRa and PPO for alignment. The configuration files for both methods are provided in the `configs` directory. The fine-tuning process is performed using the Hugging Face Trainer API (LoRa) and customized training loops (LoRa+PPO) on HPC (Slurm job schedular) with Nvidia-A100@80GB. The training scripts are located in the `scripts` directory. The fine-tuned models are saved in the `saved_models` directory (not uploaded).

## Deployment
The system can be deployed on AWS using Docker and SageMaker. The deployment scripts and configuration files are located in the `deployment` directory. The Dockerfile is used to create a container image for the application, which can be deployed on AWS SageMaker. 

More details will be provided in the near future.



