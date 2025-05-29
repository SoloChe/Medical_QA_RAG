import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import PPOTrainer, PPOConfig
import json
import logging
from datetime import datetime
import psutil
import GPUtil
from utils import compute_metrics, SaveQAExamplesCallback, save_qa_examples


def log_system_info(logger):
    logger.info("System Information:")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    logger.info(
        f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB"
    )
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(
            f"GPU {gpu.id}: {gpu.name}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB, Utilization: {gpu.load * 100:.2f}%"
        )


def preprocess_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["instruction"],
        text_target=examples["response"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def train_ppo(args):
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Log start time
    start_time = datetime.now()
    logger.info("Starting PPO Fine-Tuning...")
    logger.info(f"Start Time: {start_time}")
    log_system_info(logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project, name="PPO_Fine_Tuning", config=vars(args))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model with the LoRA adapter
    model = PeftModel.from_pretrained(args.model_name, args.adapter_dir).to(device)
    logger.info(f"Loaded LoRA fine-tuned model from {args.adapter_dir}")

    # Load the training dataset
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    # Load a small portion of the validation set for QA examples
    val_dataset = load_dataset("json", data_files=args.val_file, split="train").select(
        range(100)
    )

    # Set up PPO Trainer
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
    )
    ppo_trainer = PPOTrainer(model, tokenizer, config=ppo_config)
    logger.info(
        f"PPO Trainer initialized with batch_size={args.batch_size}, ppo_epochs={args.ppo_epochs}"
    )

    # Training Loop
    for step in range(args.max_steps):
        if step % args.save_steps == 0 and step > 0:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"[INFO] Checkpoint saved at step {step}")
            # Save QA examples
            output_file = os.path.join(
                args.output_dir, f"qa_examples_step_{step}.jsonl"
            )
            save_qa_examples(
                model,
                tokenizer,
                val_dataset,
                output_file,
                num_examples=5,
                logger=logger,
                wandb_run=wandb.run,
            )
            logger.info(f"[INFO] Saved QA examples at step {step}")

        if step % args.logging_steps == 0:
            logger.info(f"[INFO] Logging at step {step}")
            wandb.log({"current_step": step})

        batch = train_dataset.shuffle().select(range(ppo_config.batch_size))
        inputs = batch["instruction"]
        responses = []
        rewards = []

        for input_text in inputs:
            prompt = f"QUESTION: {input_text}\nCONTEXT:"
            output = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(model.device),
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(generated_text)

            # Placeholder reward function (replace with a real reward model)
            reward = len(generated_text.split()) / len(input_text.split())
            rewards.append(reward)

        # Run PPO update
        ppo_trainer.step(inputs, responses, rewards)
        wandb.log({"reward_mean": sum(rewards) / len(rewards)})
        logger.info(
            f"[INFO] Step {step+1}/{args.max_steps} - Avg Reward: {sum(rewards) / len(rewards):.4f}"
        )

    # Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"PPO fine-tuned model saved to {args.output_dir}")

    # Log end time
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Training completed in {duration}")
    log_system_info()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/processed/medmcqa_train.jsonl",
        help="Path to the training file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="./data/processed/medmcqa_val.jsonl",
        help="Path to the validation file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./saved_models/lora_finetuned",
        help="Directory to the LoRA fine-tuned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/lora_ppo_finetuned",
        help="Directory to save the final model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=3, help="Number of PPO epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum training steps"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Medical_QA_LoRA_PPO",
        help="Wandb project name",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Checkpoint save steps"
    )
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    args = parser.parse_args()

    # Start PPO training
    train_ppo(args)


if __name__ == "__main__":
    main()
