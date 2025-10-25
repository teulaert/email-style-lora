#!/usr/bin/env python3
"""
Analyze your email training data to identify your unique writing patterns
"""

import json
from collections import Counter
import re


def load_training_data():
    """Load your training emails"""
    emails = []
    with open("training_data/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Extract assistant messages
            if "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "assistant":
                        emails.append(msg.get("content", ""))
    return emails


def analyze_patterns(emails):
    """Analyze writing patterns in your emails"""

    print("="*80)
    print("YOUR EMAIL WRITING STYLE ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(emails)} training emails...\n")

    # Emails are already the assistant responses
    email_bodies = [email.strip() for email in emails if email.strip()]

    if not email_bodies:
        print("Could not find email responses in training data.")
        return

    print(f"Found {len(email_bodies)} email responses\n")

    # 1. Common opening phrases
    print("─"*80)
    print("📬 COMMON OPENING PHRASES:")
    print("─"*80)
    openings = []
    for body in email_bodies:
        lines = body.strip().split('\n')
        if lines:
            first_line = lines[0].strip()[:100]  # First 100 chars
            if first_line:
                openings.append(first_line)

    opening_counter = Counter(openings)
    for opening, count in opening_counter.most_common(10):
        if count > 1:
            print(f"  • {opening[:80]}{'...' if len(opening) > 80 else ''} ({count}x)")

    # 2. Common closing phrases
    print("\n" + "─"*80)
    print("👋 COMMON CLOSING PHRASES:")
    print("─"*80)
    closings = []
    for body in email_bodies:
        lines = [l.strip() for l in body.strip().split('\n') if l.strip()]
        if lines:
            last_line = lines[-1][:100]
            if last_line:
                closings.append(last_line)

    closing_counter = Counter(closings)
    for closing, count in closing_counter.most_common(10):
        if count > 1:
            print(f"  • {closing[:80]}{'...' if len(closing) > 80 else ''} ({count}x)")

    # 3. Common words/phrases
    print("\n" + "─"*80)
    print("💬 FREQUENT WORDS YOU USE:")
    print("─"*80)
    all_text = " ".join(email_bodies).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)  # Words 3+ letters

    # Filter out very common words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                  'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
                  'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
                  'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'het', 'een',
                  'van', 'dat', 'die', 'voor', 'met', 'niet'}

    words = [w for w in words if w not in stop_words]
    word_counter = Counter(words)

    print("Top 20 words:")
    for word, count in word_counter.most_common(20):
        print(f"  • {word}: {count}x")

    # 4. Language detection
    print("\n" + "─"*80)
    print("🌍 LANGUAGE USAGE:")
    print("─"*80)
    dutch_indicators = ['graag', 'bedankt', 'groet', 'mvg', 'heb', 'kunnen', 'zou', 'jij', 'ik']
    english_indicators = ['thanks', 'please', 'regards', 'best', 'could', 'would', 'hello', 'regards']

    dutch_count = sum(1 for body in email_bodies if any(word in body.lower() for word in dutch_indicators))
    english_count = sum(1 for body in email_bodies if any(word in body.lower() for word in english_indicators))

    total = len(email_bodies)
    print(f"  • Dutch emails: {dutch_count} ({dutch_count/total*100:.1f}%)")
    print(f"  • English emails: {english_count} ({english_count/total*100:.1f}%)")

    # 5. Email length stats
    print("\n" + "─"*80)
    print("📏 EMAIL LENGTH PATTERNS:")
    print("─"*80)
    lengths = [len(body.split()) for body in email_bodies]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    print(f"  • Average length: {avg_length:.0f} words")
    print(f"  • Shortest: {min_length} words")
    print(f"  • Longest: {max_length} words")

    # 6. Formality indicators
    print("\n" + "─"*80)
    print("🎩 FORMALITY LEVEL:")
    print("─"*80)

    contractions = ["don't", "won't", "can't", "wouldn't", "couldn't", "i'm", "i'll", "it's", "that's"]
    contraction_count = sum(body.lower().count(c) for body in email_bodies for c in contractions)

    formal_phrases = ["sincerely", "regards", "respectfully", "pleased to", "thank you for your"]
    formal_count = sum(body.lower().count(p) for body in email_bodies for p in formal_phrases)

    casual_words = ["hey", "yeah", "cool", "awesome", "gonna", "wanna"]
    casual_count = sum(body.lower().count(w) for body in email_bodies for w in casual_words)

    print(f"  • Contractions used: {contraction_count}x")
    print(f"  • Formal phrases: {formal_count}x")
    print(f"  • Casual words: {casual_count}x")

    if contraction_count > formal_count * 2:
        print("  → Style: Tends toward CASUAL/INFORMAL")
    elif formal_count > contraction_count * 2:
        print("  → Style: Tends toward FORMAL")
    else:
        print("  → Style: BALANCED between formal and casual")

    # 7. Punctuation patterns
    print("\n" + "─"*80)
    print("✏️  PUNCTUATION HABITS:")
    print("─"*80)

    exclamations = sum(body.count('!') for body in email_bodies)
    questions = sum(body.count('?') for body in email_bodies)

    print(f"  • Exclamation marks: {exclamations}x")
    print(f"  • Question marks: {questions}x")
    print(f"  • Average per email: {exclamations/len(email_bodies):.1f} !, {questions/len(email_bodies):.1f} ?")

    # 8. Sample emails
    print("\n" + "─"*80)
    print("📧 SAMPLE EMAILS FROM YOUR TRAINING DATA:")
    print("─"*80)
    print("\n(Showing 3 random examples to remind you of your style)\n")

    import random
    samples = random.sample(email_bodies[:100], min(3, len(email_bodies)))

    for i, sample in enumerate(samples, 1):
        print(f"\nExample {i}:")
        print("─"*60)
        print(sample[:400] + ("..." if len(sample) > 400 else ""))
        print("─"*60)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\n💡 When comparing models, look for these patterns in your fine-tuned responses.")


def main():
    try:
        emails = load_training_data()
        analyze_patterns(emails)
    except FileNotFoundError:
        print("Error: Could not find training_data/train.jsonl")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
