# Medical QA RAG System (updating)

This repository contains a medical question-answering system that utilizes a Retrieval-Augmented Generation (RAG) approach. The system is designed to answer medical questions by retrieving relevant documents from a knowledge base ([MedMCQA](https://medmcqa.github.io/)) and generating responses from multi-agent pipeline. Finally, the system can be deployed on AWS using Docker and SageMaker.

## Agent Pipeline
Fine-tuned Generator Model: mistralai/Mistral-7B-v0.1
Retriever Model: BM25
Contextualizer Model: mistralai/Mistral-7B-v0.1
Fact-Checker Model: all-mpnet-base-v2 (optional)
Summarizer Model: facebook/bart-large-cnn (optional)

More details will be provided in the future.

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
│   ├── requirements.txt         # Docker requirements
│   └── pipeline/                # SageMaker pipeline scripts and configuration
└── notebooks/
    └── demo.ipynb               # Jupyter notebook for demo
```
