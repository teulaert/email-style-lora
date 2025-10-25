#!/usr/bin/env python3
"""
Split processed email data into train and validation sets
"""

import json
import random
from pathlib import Path
from collections import Counter


def split_dataset(input_file: str, output_dir: str = "training_data",
                  train_ratio: float = 0.9, seed: int = 42):
    """
    Split JSONL dataset into train and validation sets

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory for output files
        train_ratio: Ratio of data to use for training (default 0.9 = 90%)
        seed: Random seed for reproducibility
    """

    # Set random seed for reproducibility
    random.seed(seed)

    # Load all data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Total examples: {len(data)}")

    # Shuffle data
    random.shuffle(data)

    # Calculate split point
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"\nSplit:")
    print(f"  Training: {len(train_data)} examples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} examples ({len(val_data)/len(data)*100:.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save train set
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nSaved training set to: {train_file}")

    # Save validation set
    val_file = output_path / "validation.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved validation set to: {val_file}")

    # Print statistics for each set
    def print_stats(dataset, name):
        languages = Counter(item['metadata']['language'] for item in dataset)
        word_counts = [item['metadata']['word_count'] for item in dataset]
        total_words = sum(word_counts)
        avg_words = total_words / len(dataset)

        print(f"\n{name} Statistics:")
        print(f"  Examples: {len(dataset)}")
        print(f"  Total words: {total_words:,}")
        print(f"  Average words: {avg_words:.0f}")
        print(f"  Language distribution:")
        for lang, count in languages.most_common():
            pct = count / len(dataset) * 100
            print(f"    {lang}: {count} ({pct:.1f}%)")

    print_stats(train_data, "Training Set")
    print_stats(val_data, "Validation Set")

    print("\n" + "="*60)
    print("Dataset split complete! Ready for training.")
    print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split email dataset for training')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('--output', default='training_data', help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Training set ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    split_dataset(args.input, args.output, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
