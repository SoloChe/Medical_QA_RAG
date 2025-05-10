#!/bin/bash
#SBATCH --job-name='train_lora_hf'
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1               
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -p general                
#SBATCH -q debug
            
#SBATCH -t 00-00:10:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


torchrun --nproc_per_node=1 scripts/train_lora_hf.py \
    --train_file ./data/processed/medmcqa_train.jsonl \
    --val_file ./data/processed/medmcqa_val.jsonl \
    --model_name mistralai/Mistral-7B-v0.1 \
    --output_dir ./saved_models/lora_finetuned \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --logging_steps 50 \
    --save_steps 500 \
    --wandb_project Medical_QA_LoRA_Training


