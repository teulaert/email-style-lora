#!/usr/bin/env python3
"""
Email Processor for LLM Fine-tuning
Extracts and processes sent emails from mbox files for personal style training
"""

import mailbox
import email
import re
import json
import html2text
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from langdetect import detect, LangDetectException


class EmailProcessor:
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False

        # Statistics
        self.stats = {
            "total_emails": 0,
            "sent_emails": 0,
            "processed_emails": 0,
            "skipped_empty": 0,
            "skipped_short": 0,
            "languages": Counter(),
            "avg_length": 0,
            "total_words": 0
        }

    def clean_text(self, text: str) -> str:
        """Clean email text by removing signatures, quoted replies, etc."""
        if not text:
            return ""

        # Common signature patterns (Dutch and English)
        signature_patterns = [
            r'Met vriendelijke groet',
            r'Kind regards',
            r'Best regards',
            r'Regards',
            r'Mvg',
            r'Groet',
            r'Groeten',
            r'Cheers',
            r'Thanks',
            r'Bedankt',
            r'Hartelijke groet',
            r'Vriendelijke groet',
        ]

        # Compile regex to find signature start
        # This will match lines that start with any signature pattern
        # even if they have additional text after (like "/ Kind regards")
        signature_regex = re.compile(
            r'^\s*(' + '|'.join(signature_patterns) + r')[,\s/].*$',
            re.IGNORECASE | re.MULTILINE
        )

        # Find signature and cut there
        match = signature_regex.search(text)
        if match:
            text = text[:match.start()]

        # Remove quoted replies (lines starting with > or |)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip quoted lines
            if stripped.startswith('>') or stripped.startswith('|'):
                continue

            # Stop at common signature delimiters
            if stripped in ['--', '___', '---'] or stripped.startswith('-- '):
                break

            # Stop at "On ... wrote:" patterns (forwarded/reply headers)
            # More flexible patterns to catch variations
            if re.match(r'^On .+wrote:?$', stripped, re.IGNORECASE):
                break
            if re.match(r'^Op .+schreef.+:?$', stripped, re.IGNORECASE):  # Dutch
                break
            # Catch patterns with dates like "On 10 Feb 2023, at 11:49"
            if re.match(r'^On \d+.*at \d+:', stripped, re.IGNORECASE):
                break

            # Skip lines with just special characters (likely artifacts)
            if re.match(r'^[\W_]+$', stripped) and len(stripped) < 5:
                continue

            # Skip very long lines (likely HTML artifacts)
            if len(line) > 500:
                continue

            # Skip lines that are likely phone numbers by themselves
            if re.match(r'^(Tel|Telefoon|Phone|Mobile|Mob)[\s:]+.*[\d\s\+\(\)\-]+$', stripped, re.IGNORECASE):
                continue

            cleaned_lines.append(line)

        # Join lines and clean up extra whitespace
        cleaned = '\n'.join(cleaned_lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 newlines
        cleaned = re.sub(r' {2,}', ' ', cleaned)  # Remove extra spaces

        return cleaned.strip()

    def extract_body(self, msg) -> Optional[str]:
        """Extract email body, preferring plain text over HTML"""
        body = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break  # Prefer plain text
                    except:
                        continue

                elif content_type == "text/html" and body is None:
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body = self.html_converter.handle(html_content)
                    except:
                        continue
        else:
            # Not multipart
            content_type = msg.get_content_type()
            try:
                if content_type == "text/plain":
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == "text/html":
                    html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    body = self.html_converter.handle(html_content)
            except:
                return None

        return body

    def detect_language(self, text: str) -> str:
        """Detect if email is in Dutch or English"""
        try:
            # Take a sample of text (first 500 chars for efficiency)
            sample = text[:500] if len(text) > 500 else text
            lang = detect(sample)

            # Map to main languages, default to 'en' for unknown
            if lang in ['nl', 'en']:
                return lang
            else:
                return 'unknown'
        except LangDetectException:
            return 'unknown'

    def process_email(self, msg) -> Optional[Dict]:
        """Process a single email message"""
        try:
            # Extract metadata
            subject = msg.get('Subject', '').strip()
            from_addr = msg.get('From', '').strip()
            to_addr = msg.get('To', '').strip()

            # Extract and clean body
            body = self.extract_body(msg)
            if not body:
                self.stats['skipped_empty'] += 1
                return None

            body = self.clean_text(body)

            # Skip if too short (likely automated/system emails)
            word_count = len(body.split())
            if word_count < 10:
                self.stats['skipped_short'] += 1
                return None

            # Detect language
            language = self.detect_language(body)
            self.stats['languages'][language] += 1

            # Update statistics
            self.stats['total_words'] += word_count

            # Create training format
            # System prompt can be customized based on language
            system_prompt = "You are an AI assistant that writes emails in a personal, authentic style."

            user_prompt = f"Write an email with subject: {subject}"
            if to_addr:
                user_prompt += f"\nTo: {to_addr}"

            training_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": body}
                ],
                "metadata": {
                    "language": language,
                    "word_count": word_count,
                    "subject": subject
                }
            }

            return training_example

        except Exception as e:
            print(f"Error processing email: {e}")
            return None

    def process_mbox_file(self, mbox_path: str, max_emails: Optional[int] = None):
        """Process an mbox file and extract sent emails"""
        print(f"\nProcessing: {mbox_path}")
        mbox_path = Path(mbox_path)

        if not mbox_path.exists():
            print(f"Error: {mbox_path} does not exist")
            return

        # Handle both .mbox files and directories
        if mbox_path.is_dir():
            # Gmail Takeout format: .mbox directories contain a file named "mbox"
            mbox_file_in_dir = mbox_path / "mbox"
            if mbox_file_in_dir.exists():
                # Process the mbox file directly
                mbox_path = mbox_file_in_dir
            else:
                # Check for mbox files in directory
                mbox_files = list(mbox_path.glob("*.mbox"))
                if not mbox_files:
                    print(f"No mbox files found in {mbox_path}")
                    return

                for mbox_file in mbox_files:
                    self.process_mbox_file(str(mbox_file), max_emails)
                return

        try:
            mbox = mailbox.mbox(str(mbox_path))
            processed = []

            for idx, msg in enumerate(mbox):
                if max_emails and idx >= max_emails:
                    break

                self.stats['total_emails'] += 1

                # For Gmail takeout, sent emails are often in specific folders
                # or you may need to check if you're the sender
                # Adjust this logic based on your mbox structure
                processed_email = self.process_email(msg)

                if processed_email:
                    processed.append(processed_email)
                    self.stats['processed_emails'] += 1

                # Progress indicator
                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1} emails...")

            # Save to JSONL
            if processed:
                output_file = self.output_dir / f"{mbox_path.stem}_processed.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in processed:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                print(f"Saved {len(processed)} processed emails to {output_file}")

        except Exception as e:
            print(f"Error reading mbox file {mbox_path}: {e}")

    def process_directory(self, source_dir: str, max_emails: Optional[int] = None):
        """Process all mbox files in a directory recursively"""
        source_path = Path(source_dir)

        # Find all .mbox files and directories
        mbox_files = list(source_path.rglob("*.mbox"))
        mbox_dirs = [d for d in source_path.rglob("*") if d.is_dir() and d.suffix == ".mbox"]

        all_mboxes = mbox_files + mbox_dirs

        if not all_mboxes:
            print(f"No mbox files found in {source_dir}")
            return

        print(f"Found {len(all_mboxes)} mbox files/folders")

        for mbox_path in all_mboxes:
            self.process_mbox_file(str(mbox_path), max_emails)

    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total emails scanned: {self.stats['total_emails']}")
        print(f"Successfully processed: {self.stats['processed_emails']}")
        print(f"Skipped (empty): {self.stats['skipped_empty']}")
        print(f"Skipped (too short): {self.stats['skipped_short']}")
        print(f"\nLanguage distribution:")
        for lang, count in self.stats['languages'].most_common():
            percentage = (count / self.stats['processed_emails'] * 100) if self.stats['processed_emails'] > 0 else 0
            print(f"  {lang}: {count} ({percentage:.1f}%)")

        if self.stats['processed_emails'] > 0:
            avg_words = self.stats['total_words'] / self.stats['processed_emails']
            print(f"\nAverage words per email: {avg_words:.0f}")
            print(f"Total words: {self.stats['total_words']:,}")

        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process emails for LLM fine-tuning')
    parser.add_argument('source', help='Source directory or mbox file')
    parser.add_argument('--output', default='processed_data', help='Output directory')
    parser.add_argument('--max-emails', type=int, help='Maximum emails to process (for testing)')
    parser.add_argument('--sample', action='store_true', help='Process only first 100 emails (for testing)')

    args = parser.parse_args()

    processor = EmailProcessor(output_dir=args.output)

    max_emails = args.max_emails
    if args.sample:
        max_emails = 100

    source_path = Path(args.source)
    # If it's a .mbox directory, treat it as an mbox file location
    if source_path.is_dir() and source_path.suffix == ".mbox":
        processor.process_mbox_file(args.source, max_emails)
    elif source_path.is_dir():
        processor.process_directory(args.source, max_emails)
    else:
        processor.process_mbox_file(args.source, max_emails)

    processor.print_statistics()


if __name__ == "__main__":
    main()
