#!/usr/bin/env python3
"""
Compare base model vs your fine-tuned model side-by-side
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


def load_model(model_type, adapter_path=None):
    """Load a single model (base or fine-tuned)"""
    print(f"Loading model: {MODELS[model_type]}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_type],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_type])

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_tokens=300):
    """Generate a response from the model"""
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
            max_new_tokens=max_tokens,
            temperature=0.7,
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
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--model", type=str, choices=["llama", "mistral"], required=True,
                        help="Model type: llama or mistral")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (default: models/{model}-email-lora/final)")

    args = parser.parse_args()

    # Set default adapter path based on model type
    if args.adapter is None:
        if args.model == "llama":
            args.adapter = "models/llama-3.1-8b-email-lora/final"
        else:  # mistral
            args.adapter = "models/mistral-7b-email-lora/final"

    print("="*80)
    print("COMPARING BASE MODEL vs YOUR FINE-TUNED MODEL")
    print("="*80)
    print(f"\n📦 Base Model: {MODELS[args.model]}")
    print(f"✨ Fine-tuned: {args.adapter}")
    print("\nThis shows how your fine-tuned model differs from the base model.")
    print("Look for personal touches, typical phrases, or style elements you use.\n")

    # Try to load both models at once (faster)
    sequential_mode = False
    base_model, base_tokenizer, finetuned_model, finetuned_tokenizer = None, None, None, None

    try:
        print("⏳ Loading both models (this is faster)...")
        base_model, base_tokenizer = load_model(args.model)
        finetuned_model, finetuned_tokenizer = load_model(args.model, args.adapter)
        print("✅ Both models loaded successfully!\n")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print("\n⚠️  Not enough memory to load both models at once.")
        print("   Falling back to sequential mode (slower but uses less memory)...\n")
        sequential_mode = True
        # Clean up any partially loaded models
        if base_model:
            del base_model
            del base_tokenizer
        if finetuned_model:
            del finetuned_model
            del finetuned_tokenizer
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("READY TO COMPARE!")
    print("="*80)

    # Test prompts - these should reveal personal writing style
    test_prompts = [
        "Write a brief email reply thanking someone for their message and saying you'll get back to them soon.",
        "Write a short email asking a colleague if they have time for a quick call this week.",
        "Write a reply to someone asking when you can deliver a project, saying it will be ready next week.",
        "Write a quick email apologizing for a late response.",
        "Write an email confirming you received someone's documents and will review them.",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"PROMPT {i}/{len(test_prompts)}:")
        print(f"{'='*80}")
        print(f"{prompt}\n")

        if sequential_mode:
            # Load models one at a time
            print("⏳ Generating BASE MODEL response...")
            base_model, base_tokenizer = load_model(args.model)
            base_response = generate_response(base_model, base_tokenizer, prompt, max_tokens=200)
            del base_model
            del base_tokenizer
            torch.cuda.empty_cache()

            print("⏳ Generating YOUR FINE-TUNED MODEL response...")
            finetuned_model, finetuned_tokenizer = load_model(args.model, args.adapter)
            finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt, max_tokens=200)
            del finetuned_model
            del finetuned_tokenizer
            torch.cuda.empty_cache()
        else:
            # Both models already loaded - just generate
            print("⏳ Generating BASE MODEL response...")
            base_response = generate_response(base_model, base_tokenizer, prompt, max_tokens=200)

            print("⏳ Generating YOUR FINE-TUNED MODEL response...")
            finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt, max_tokens=200)

        print("\n" + "─"*80)
        print("📧 BASE MODEL (Generic):")
        print("─"*80)
        print(base_response)

        print("\n" + "─"*80)
        print("✨ YOUR MODEL (Fine-tuned on your emails):")
        print("─"*80)
        print(finetuned_response)

        print("\n" + "─"*80)
        print("💭 What to look for:")
        print("   • Specific phrases or words you commonly use")
        print("   • Your typical greeting/closing style")
        print("   • Formality level (casual vs formal)")
        print("   • Sentence structure and length")
        print("   • Use of contractions, punctuation")
        print("   • Language mixing (if bilingual)")
        print("─"*80)

        input("\n⏸️  Press Enter to continue...")

    # Interactive mode
    print("\n\n" + "="*80)
    print("INTERACTIVE COMPARISON MODE")
    print("="*80)
    print("Enter your own prompts to compare the models.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("Your prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not prompt:
                continue

            if sequential_mode:
                print("\n⏳ Generating BASE MODEL response...")
                base_model, base_tokenizer = load_model(args.model)
                base_response = generate_response(base_model, base_tokenizer, prompt, max_tokens=250)
                del base_model
                del base_tokenizer
                torch.cuda.empty_cache()

                print("⏳ Generating YOUR FINE-TUNED MODEL response...")
                finetuned_model, finetuned_tokenizer = load_model(args.model, args.adapter)
                finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt, max_tokens=250)
                del finetuned_model
                del finetuned_tokenizer
                torch.cuda.empty_cache()
            else:
                print("\n⏳ Generating responses...")
                base_response = generate_response(base_model, base_tokenizer, prompt, max_tokens=250)
                finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt, max_tokens=250)

            print("─"*80)
            print("📧 BASE MODEL:")
            print("─"*80)
            print(base_response)

            print("\n" + "─"*80)
            print("✨ YOUR MODEL:")
            print("─"*80)
            print(finetuned_response)
            print("─"*80 + "\n")

        except KeyboardInterrupt:
            print("\n")
            break

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
