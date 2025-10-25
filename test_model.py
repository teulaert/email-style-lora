#!/usr/bin/env python3
"""
Test your fine-tuned email style model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Model base names
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def load_model(model_type, adapter_path):
    """Load the base model and LoRA adapter"""
    print(f"Loading base model: {MODELS[model_type]}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_type],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model.eval()
    return model, tokenizer


def generate_email(model, tokenizer, prompt, max_length=300, temperature=0.7):
    """Generate an email response"""
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response.lower():
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Test your fine-tuned model")
    parser.add_argument("--model", type=str, choices=["llama", "mistral"], required=True,
                        help="Model type: llama or mistral")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (default: models/{model}-email-lora/final)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Generation temperature (0.1-1.0, default: 0.7)")

    args = parser.parse_args()

    # Set default adapter path based on model type
    if args.adapter is None:
        if args.model == "llama":
            args.adapter = "models/llama-3.1-8b-email-lora/final"
        else:  # mistral
            args.adapter = "models/mistral-7b-email-lora/final"

    print("="*80)
    print("PERSONAL EMAIL STYLE MODEL - INTERACTIVE TEST")
    print("="*80)
    print(f"\n📦 Model: {MODELS[args.model]}")
    print(f"🎯 Adapter: {args.adapter}")
    print(f"🌡️  Temperature: {args.temp}")
    print("\n" + "="*80)

    # Load model
    model, tokenizer = load_model(args.model, args.adapter)

    print("\n✅ Model loaded successfully!")
    print("\n" + "="*80)
    print("EXAMPLE PROMPTS")
    print("="*80)

    # Example prompts
    examples = [
        "Write a reply thanking someone for their email and saying you'll get back to them soon.",
        "Write a short email asking a colleague if they're available for a quick call this week.",
        "Write a reply apologizing for a late response and providing an update.",
    ]

    for i, prompt in enumerate(examples, 1):
        print(f"\n📧 Example {i}: {prompt}")
        print("-"*80)
        response = generate_email(model, tokenizer, prompt, max_length=200, temperature=args.temp)
        print(response)
        print("-"*80)

    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your prompts below. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("Your prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not prompt:
                continue

            print("\n⏳ Generating...\n")
            response = generate_email(model, tokenizer, prompt, max_length=400, temperature=args.temp)
            print("="*80)
            print(response)
            print("="*80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
