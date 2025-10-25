# Personal Email Style Fine-Tuning

Fine-tune LLMs (Llama 3.1 or Mistral 7B) on your personal email style using LoRA. Train the model to write emails that sound like you!

## Features

- ✅ **Multiple Models**: Support for Llama 3.1 8B and Mistral 7B
- ✅ **Flexible Training**: Single GPU or Multi-GPU (DDP) training
- ✅ **Easy to Use**: Simple command-line interface
- ✅ **Memory Efficient**: Uses LoRA for parameter-efficient fine-tuning
- ✅ **Resume Training**: Automatic checkpoint resuming
- ✅ **Quality Tools**: Compare your model against the base model

## Prerequisites

- Python 3.9+
- NVIDIA GPU with 16GB+ VRAM (24GB+ recommended for comfortable training)
- CUDA 11.8+ or 12.x
- 20+ GB free disk space

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/teulaert/email-style-lora
cd email-style-lora

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

**⚠️ Important:** Export your **Sent Items** (emails YOU wrote), not your inbox!

Export your sent emails from Gmail (Google Takeout) or your email client in `.mbox` format.

**For Gmail:**
- Go to [Google Takeout](https://takeout.google.com/)
- Select only "Mail"
- Click "All Mail data included" and deselect everything except **"Sent"**
- Download the `.mbox` file

```bash
# Process your mbox files (sent items only!)
python email_processor.py --input path/to/Sent.mbox --output training_data

# Split into train/validation sets
python split_data.py --input training_data/processed.jsonl --train-ratio 0.9
```

This will create:
- `training_data/train.jsonl` - Training set (90%)
- `training_data/validation.jsonl` - Validation set (10%)

### 3. Train Your Model

#### Single GPU Training

```bash
python train.py --model mistral --gpus 1 --epochs 3
```

#### Multi-GPU Training (Recommended)

```bash
torchrun --nproc_per_node=2 train.py --model mistral --gpus 2 --epochs 3
```

**Available options:**
- `--model`: Choose `llama` or `mistral`
- `--gpus`: Number of GPUs (1 or 2)
- `--batch-size`: Per-device batch size (default: 2)
- `--epochs`: Number of training epochs (default: 3)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--output-dir`: Output directory (default: models/{model}-email-lora)

### 4. Test Your Model

```bash
# Interactive testing
python test_model.py --model mistral

# With custom temperature
python test_model.py --model mistral --temp 0.8
```

### 5. Compare with Base Model

See how your fine-tuned model differs from the base model:

```bash
python compare_models.py --model mistral
```

### 6. Analyze Your Writing Style

Understand what patterns make your writing unique:

```bash
python analyze_style.py
```

## Project Structure

```
mail-training/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── train.py                 # Unified training script
├── test_model.py            # Test trained models
├── compare_models.py        # Compare base vs fine-tuned
├── analyze_style.py         # Analyze writing style
│
├── email_processor.py       # Process mbox files
├── split_data.py            # Split train/val data
│
├── training_data/           # (gitignored - your personal data)
│   ├── train.jsonl
│   └── validation.jsonl
│
└── models/                  # (gitignored - trained models)
    └── {model}-email-lora/
        ├── checkpoint-100/
        ├── checkpoint-200/
        └── final/          # Use this for inference
```

## Training Details

### Memory Usage

- **Single GPU (Mistral 7B)**: ~18-20 GB VRAM
- **Single GPU (Llama 3.1 8B)**: ~20-22 GB VRAM
- **Dual GPU**: ~16-18 GB VRAM per GPU (with DDP)

### Training Time

On 2x RTX 5090 with ~3,000 emails:
- **Single GPU**: 2-3 hours
- **Dual GPU (DDP)**: 1-1.5 hours

### Hyperparameters

Default settings work well for most cases:
- LoRA rank: 32
- LoRA alpha: 64
- Batch size: 2 per GPU
- Gradient accumulation: 8 steps
- Learning rate: 2e-4
- Epochs: 3

## Troubleshooting

### Out of Memory Error

**Solution**: Reduce batch size:
```bash
python train.py --model mistral --gpus 1 --batch-size 1
```

### "403 Client Error" for Llama

**Solution**: Request access to Llama 3.1 on HuggingFace:
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Request access (usually approved within hours)
3. Login: `huggingface-cli login`

### Training is Too Slow

**Solution**: Use multi-GPU training:
```bash
torchrun --nproc_per_node=2 train.py --model mistral --gpus 2
```

### Model Output is Generic

This is normal! Your model captures subtle patterns. Use `compare_models.py` and `analyze_style.py` to see the differences. Look for:
- Your typical greetings (e.g., "Hoi" vs "Hi")
- Your closing style (e.g., "Dank!" vs "Best regards")
- Sentence length and structure
- Formality level
- Language mixing (if bilingual)

## Advanced Usage

### Custom Data Format

Your training data should be in JSONL format with this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that writes emails in a personal, authentic style."
    },
    {
      "role": "user",
      "content": "Write an email with subject: Meeting follow-up"
    },
    {
      "role": "assistant",
      "content": "Hi,\n\nThanks for the meeting today..."
    }
  ]
}
```

### Export Merged Model

To merge the LoRA adapter with the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(model, "models/mistral-email-lora/final")
model = model.merge_and_unload()

model.save_pretrained("models/mistral-email-merged")
tokenizer = AutoTokenizer.from_pretrained("models/mistral-email-lora/final")
tokenizer.save_pretrained("models/mistral-email-merged")
```

## Privacy & Security

⚠️ **Important**: Your training data contains personal emails and is automatically excluded from git (`.gitignore`). Never commit:
- `training_data/` directory
- `models/` directory
- Any log files containing email content

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers/)
- Uses [PEFT](https://github.com/huggingface/peft) for LoRA training
- Supports [Llama 3.1](https://huggingface.co/meta-llama) and [Mistral](https://huggingface.co/mistralai)

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review existing issues on GitHub
3. Open a new issue with details about your setup

---

**Happy Training!** 🚀
