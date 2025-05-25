#!/bin/bash
#SBATCH --job-name='eval'
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1               
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -p general                
#SBATCH -q public
            
#SBATCH -t 00-10:00:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out

module purge
module load mamba/latest
source activate medicalQARAG


~/.conda/envs/medicalQARAG/bin/python scripts/evaluate_QA.py \
        --rag True \
        --top_k_ret 20 \
        --use_corrector False\
        --use_ranker False \


