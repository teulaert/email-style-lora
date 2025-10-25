#!/usr/bin/env python3
"""
Unified training script for fine-tuning LLMs on personal email style
Supports: Llama 3.1 8B, Mistral 7B
Supports: Single GPU or Multi-GPU (DDP) training
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model


# Model configurations
MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
}


def setup_distributed():
    """Initialize distributed training if using multiple GPUs"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_data(file_path):
    """Load JSONL dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess_function(example, tokenizer, max_length):
    """Convert messages to model input format"""
    messages = example.get("messages", [])

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    result = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()

    return result


def main():
    parser = argparse.ArgumentParser(description="Train LLM on personal email style")

    # Model selection
    parser.add_argument("--model", type=str, choices=["llama", "mistral"], required=True,
                        help="Model to train: llama or mistral")

    # GPU configuration
    parser.add_argument("--gpus", type=int, default=1, choices=[1, 2],
                        help="Number of GPUs to use (1 or 2)")

    # Data paths
    parser.add_argument("--train-data", type=str, default="training_data/train.jsonl",
                        help="Path to training data")
    parser.add_argument("--val-data", type=str, default="training_data/validation.jsonl",
                        help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for model (default: models/{model}-email-lora)")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")

    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Other options
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Run evaluation every N steps")

    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    # Check if using torchrun for multi-GPU
    if args.gpus > 1 and world_size == 1:
        if is_main_process:
            print("\n⚠️  ERROR: You specified --gpus 2 but launched with python instead of torchrun")
            print("\nTo train with 2 GPUs, use:")
            print(f"  torchrun --nproc_per_node=2 train.py --model {args.model} --gpus 2")
            print("\nFor single GPU training, use:")
            print(f"  python train.py --model {args.model} --gpus 1\n")
        return

    # Set output directory based on model type
    if args.output_dir is None:
        if args.model == "llama":
            args.output_dir = "models/llama-3.1-8b-email-lora"
        else:  # mistral
            args.output_dir = "models/mistral-7b-email-lora"

    # Print configuration
    if is_main_process:
        print("="*80)
        print("PERSONAL EMAIL STYLE FINE-TUNING")
        print("="*80)
        model_config = MODELS[args.model]
        print(f"\n📦 Model: {model_config['name']}")
        print(f"🖥️  GPUs: {world_size}")
        print(f"📊 Batch size: {args.batch_size} (per device)")
        print(f"📈 Gradient accumulation: {args.grad_accum}")
        print(f"🎯 Effective batch size: {args.batch_size * args.grad_accum * world_size}")
        print(f"📚 Epochs: {args.epochs}")
        print(f"💾 Output: {args.output_dir}")
        print("="*80)

    # Load tokenizer
    if is_main_process:
        print("\n1. Loading tokenizer...")
    model_config = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    if is_main_process:
        print("\n2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Configure LoRA
    if is_main_process:
        print("\n3. Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=model_config["lora_targets"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()

    # Move model to GPU and wrap with DDP if multi-GPU
    model = model.to(f'cuda:{local_rank}')
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load datasets
    if is_main_process:
        print(f"\n4. Loading training data from {args.train_data}...")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize datasets
    if is_main_process:
        print("\n5. Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        remove_columns=val_dataset.column_names
    )

    # Training arguments
    if is_main_process:
        print("\n6. Configuring training parameters...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        load_best_model_at_end=False,
        save_total_limit=3,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Initialize trainer
    if is_main_process:
        print("\n7. Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    if is_main_process:
        print("\n8. Starting training...")
        print("="*80)
        print(f"📧 Training examples: {len(train_dataset)}")
        print(f"✅ Validation examples: {len(val_dataset)}")
        print(f"⏱️  Estimated time: ~{len(train_dataset) // (args.batch_size * args.grad_accum * world_size) * args.epochs * 14 / 3600:.1f} hours")
        print("="*80)

    # Check for existing checkpoints
    resume_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            if is_main_process:
                print(f"\n🔄 Resuming from checkpoint: {checkpoints[-1]}\n")

    # Train
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    if is_main_process:
        print("\n9. Saving final model...")
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"📦 Model saved to: {args.output_dir}/final")
        print(f"\n🧪 Test your model:")
        print(f"  python test_model.py --model {args.model}")
        print(f"\n📊 Compare with base model:")
        print(f"  python compare_models.py --model {args.model}")
        print("="*80)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
