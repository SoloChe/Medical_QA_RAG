#!/bin/bash
#SBATCH --job-name='train_lora_hf'
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1               
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-12:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate medicalQARAG

# slurm setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

torchrun --nproc_per_node=1 \
         --nnodes=1\
         --rdzv_backend=c10d\
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\
        scripts/train_lora_hf.py \
        --train_file ./data/processed/medmcqa_train.jsonl \
        --val_file ./data/processed/medmcqa_val.jsonl \
        --model_name mistralai/Mistral-7B-v0.1 \
        --output_dir ./saved_models/lora_finetuned \
        --batch_size 3 \
        --learning_rate 1e-5 \
        --num_train_epochs 2 \
        --logging_steps 50 \
        --save_steps 1000 \
        --wandb_project MedicalQA_LoRA_Fine_Tuning \


