import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import json
import os
import torch
from transformers import TrainerCallback


def compute_metrics(pred):
    # Extract labels and predictions
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)

    # Mask out padding tokens
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    # Basic Classification Metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class SaveQAExamplesCallback(TrainerCallback):
    def __init__(
        self,
        logger,
        val_dataset,
        tokenizer,
        output_dir,
        num_examples=5,
        save_steps=500,
        wandb_run=None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_examples = num_examples
        self.wandb_run = wandb_run
        self.logger = logger
        self.save_steps = save_steps

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.save_steps == 0:
            step = state.global_step
            output_file = os.path.join(
                self.output_dir, f"sample_qa_examples_step_{step}.json"
            )
            self.logger.info(
                f"Saving {self.num_examples} QA examples at step {step}..."
            )
            save_qa_examples(
                model,
                self.tokenizer,
                self.val_dataset,
                output_file,
                num_examples=self.num_examples,
                wandb_run=self.wandb_run,
                logger=self.logger,
            )


def save_qa_examples(
    model,
    tokenizer,
    val_dataset,
    output_file,
    num_examples=2,
    logger=None,
    wandb_run=None,
):

    logger.info(f"Saving {num_examples} QA examples to {output_file}")
    model.eval()
    examples = []
    table_data = []  # Corrected

    for sample in val_dataset.select(range(num_examples)):
        # Prepare the input and generate the response
        instruction_text = sample["instruction"]
        true_response_text = sample["response"][1:-5] # remove the fist space and last </s> token
        
        instruction_token = tokenizer(
            instruction_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt").to(model.device)

        model_output = model.generate(
            **instruction_token,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1
        )
        generated_response = tokenizer.decode(model_output[0], skip_special_tokens=True)

        # Add to local JSON
        examples.append(
            {
                "instruction": instruction_text,
                "true_response": true_response_text,
                "generated_response": generated_response,
            }
        )

        # Add to WandB table if provided
        if wandb_run is not None:
            table_data.append(
                [
                    examples[-1]["instruction"],
                    examples[-1]["true_response"],
                    examples[-1]["generated_response"],
                ]
            )

    # Log to WandB if available
    if wandb_run is not None:
        columns = ["Instruction", "True Response", "Generated Response"]
        table = wandb.Table(columns=columns, data=table_data)
        wandb_run.log({"Sample QA Examples": table})

    # Save to local JSON file
    with open(output_file, "w") as f:
        json.dump(examples, f, indent=4)
    logger.info(f"Saved QA examples to {output_file}")
    model.train()  # Set the model back to training mode
