# Medical QA RAG System (updating)
This repository contains a medical question-answering system that utilizes a Retrieval-Augmented Generation (RAG) approach. The system is designed to answer medical questions by retrieving relevant documents from a knowledge base ([PubMedQA](https://pubmedqa.github.io/)) and generating responses from multi-agent pipeline. Finally, the system can be deployed on AWS using Docker and SageMaker. 

The current knowledge base ([PubMedQA](https://pubmedqa.github.io/)) is for testing only. The final knowledge base will be a collection of medical documents, including PubMed articles, clinical guidelines, and other relevant resources. The system is designed to be extensible and can be easily adapted to different domains or datasets.

[05/08/2025 update]: Finished the training scripts of LoRa and RAG pipeline.

[05/09/2025 update]: Debugged the training scripts and the RAG pipeline. The system is now fully functional.

[05/10/2025 update]: Finished the training scripts of LoRa+PPO. Finished the preparation of the deployment scripts except `predictor.py`. 

[05/12/2025 update]: Preprocessed PubMedQA for knowledge base. Changed the retriever from BM25 to FAISS+BioBERT. 

[05/13/2025 update]: Changed model to `mistralai/Mistral-7B-Instruct-v0.2` for generator and re-preprocessed data (adding tags) for fine-tuning.



[Next]: Exploring the reinforcement learning (PPO) for alignment. Trying to eliminate the repeated generation.

## Agent Pipeline
1. Retriever Model: FAISS+BioBERT
2. Contextualizer Model: mistralai/Mistral-7B-v0.1
3. Fine-tuned Generator Model: mistralai/Mistral-7B-Instruct-v0.2 on [MedMCQA](https://medmcqa.github.io/) dataset
4. Fact-Checker Model: all-mpnet-base-v2 (optional)
5. Summarizer Model: facebook/bart-large-cnn (optional)

## Fine-tuning
The generator model is fine-tuned using LoRa and PPO for alignment. The configuration files for both methods are provided in the `configs` directory. The fine-tuning process is performed using the Hugging Face Trainer API (LoRa) and customized training loops (LoRa+PPO) on HPC (Slurm job schedular) with Nvidia-A100@80GB. The training scripts are located in the `scripts` directory. The fine-tuned models are saved in the `saved_models` directory (not uploaded).

## Deployment
The system can be deployed on AWS using Docker and SageMaker. The deployment scripts and configuration files are located in the `deployment` directory. The Dockerfile is used to create a container image for the application, which can be deployed on AWS SageMaker. 

More details will be provided in the near future.

## System Structure
The system is built using the following components:

```
medical_qa_rag/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # Raw downloaded datasets (MedMCQA, PubMedQA, etc.)
│   ├── processed/               # Preprocessed and tokenized data
│   ├── medical_docs.txt         # Knowledge base for RAG
│   └── medical_qa.jsonl         # Fine-tuning data (instruction-response pairs)
├── saved_models/
│   ├── lora_finetuned/          # Fine-tuned LLaMA model
│   └── lora_ppo_finetuned/      # LoRA fine-tuned model
├── scripts/
│   ├── preprocess_data.py       # Data preprocessing scripts
│   ├── train_lora.py            # Fine-tuning script for LoRA
│   ├── train_lora_ppo.py        # Fine-tuning script for LoRA with PPO
│   ├── rag_pipeline.py          # RAG pipeline script
│   ├── preprocess.py            # Preprocessing data for fine-tuning
│   ├── preprocess_rag.py        # Preprocessing data for RAG
│   ├── utils.py                 # Utility functions 
├── agents/
│   ├── retriever.py             # Dense retrieval agent (DPR, BM25)
│   ├── contextualizer.py        # Context conditioning agent
│   ├── generator.py             # Response generation agent
│   ├── fact_checker.py          # Fact-checking agent (optional)
│   └── summarizer.py            # Summarization agent (optional)
├── configs/
│   ├── train_lora_hf.sh         # Model configuration file
│   └── train_lora_ppo_hf.sh     # Training configuration file
├── deployment/
│   ├── Dockerfile               # Dockerfiles for AWS deployment 
│   └── pipeline/                # SageMaker pipeline scripts and configuration
└── notebooks/
    └── demo.ipynb               # Jupyter notebook for demo
```
