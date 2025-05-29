import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import json
import logging
from datetime import datetime
import psutil
import GPUtil
from utils import compute_metrics, SaveQAExamplesCallback
import time


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


# Preprocessing function
def preprocess_function(example, tokenizer, max_length=512):
    text = example["text"]
    # Tokenize
    tokenized = tokenizer(
        text, truncation=True, padding="max_length", max_length=max_length
    )
    # Labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_lora(args):
    # Create output directory with timestamp
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.output_dir, exist_ok=True)

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
    logger.info("Starting LoRA Fine-Tuning...")
    logger.info(f"Start Time: {start_time}")

    log_system_info(logger)

    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project, name="LoRA_Fine_Tuning", config=vars(args))

    # Save the config to a local file for reference
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad token to eos_token: {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added [PAD] token and resized model embeddings")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    logger.info(f"Loaded model and tokenizer from {args.model_name}")

    # Prepare the LoRA configuration
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.15,
        bias="lora_only",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info(
        f"LoRA configuration applied with r={lora_config.r}, alpha={lora_config.lora_alpha}"
    )

    # Load the training and validation datasets
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = load_dataset("json", data_files=args.val_file, split="validation")
    # Preprocess the validation dataset')
    val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), batched=True
    )
    # select only the first 100 examples for validation
    val_dataset = val_dataset.select(range(100))

    logger.info(f"Loaded and preprocessed training data from {args.train_file}")
    logger.info(f"Loaded and preprocessed validation data from {args.val_file}")

    # Training arguments
    training_args = TrainingArguments(
        run_name="MedicalQA_LoRA_Fine_Tuning",
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        label_names=["labels"],
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=2,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=["wandb"],
        fp16=True,
        save_total_limit=10,
        # gradient_checkpointing=True,  # Memory optimization
        optim="adamw_torch",
        weight_decay=0.01,
        ddp_find_unused_parameters=False,  # For multi-GPU
        remove_unused_columns=False,  # important
        deepspeed=None,  # Use DeepSpeed if needed
    )
    logger.info("Training arguments configured")

    callback = SaveQAExamplesCallback(
        logger=logger,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        num_examples=5,
        wandb_run=wandb.run,
        save_steps=args.save_steps,
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the LoRA fine-tuned model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"LoRA fine-tuned model saved to {args.output_dir}")

    # Log end time
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Training completed in {duration}")

    # Log final system info
    log_system_info(logger)


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
        "--output_dir",
        type=str,
        default="./models/lora_finetuned",
        help="Directory to save the LoRA fine-tuned model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Checkpoint save steps"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Medical_QA_LoRA",
        help="Wandb project name",
    )
    args = parser.parse_args()

    # Stage 1: LoRA Fine-Tuning
    train_lora(args)


if __name__ == "__main__":
    main()
