Medical QA RAG System (updating)
========================
This repository contains a medical question-answering system that utilizes a Retrieval-Augmented Generation (RAG) approach. The system is designed to answer medical questions by retrieving relevant documents from a knowledge base ([MedMCQA](https://medmcqa.github.io/)) and generating responses from multi-agent pipeline. Finally, the system can be deployed on AWS using Docker and SageMaker.

More details will be provided in the future.

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
├── agents/
│   ├── retriever.py             # Dense retrieval agent (DPR, BM25)
│   ├── contextualizer.py        # Context conditioning agent
│   ├── generator.py             # Response generation agent
│   ├── fact_checker.py          # Fact-checking agent (optional)
│   └── summarizer.py            # Summarization agent (optional)
├── app/
│   ├── server.py                # Flask API for live inference
│   ├── client.py                # Client script for testing the API
│   └── utils.py                 # Common utilities
├── configs/
│   ├── model_config.yaml        # Model configuration file
│   └── training_config.yaml     # Training configuration file
├── deployment/
│   ├── docker/                  # Dockerfiles for AWS deployment
│   └── sagemaker/               # SageMaker scripts and configuration
└── notebooks/
    └── analysis.ipynb           # Jupyter notebook for exploratory analysis
```
