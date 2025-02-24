import os
import csv
from collections import Counter

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def extract_vocab_from_csv(input_csv):
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        urls = [row[0] for row in reader]
        
        # Gather all characters from the URLs
        all_chars = ''.join(urls)
        
        # Create a Counter object to count character frequencies
        char_count = Counter(all_chars)
        
        # Sort characters by frequency and create vocab
        vocab = sorted(char_count.keys())

        # Add special tokens to the vocab
        vocab = SPECIAL_TOKENS + vocab
    
    return vocab

def process_all_csv_files():
    all_vocab = set()
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv"):
                input_csv = os.path.join(root, file)
                vocab = extract_vocab_from_csv(input_csv)
                all_vocab.update(vocab)
    
    # 去除空字符串
    all_vocab.discard('')
    
    # Write the combined vocabulary to a single file
    with open("vocab.txt", "w", encoding="utf-8") as vocab_file:
        for char in sorted(all_vocab):
            vocab_file.write(char + '\n')
    
    print("Vocabulary extraction completed")


# 输出一个ascii码内可打印字符的simply_vacab.txt
def generate_simple_vocab():
    with open("simple_vocab.txt", "w", encoding="utf-8") as vocab_file:
        for i in range(33, 127):
            vocab_file.write(chr(i) + '\n')
        # 特殊token
        for token in SPECIAL_TOKENS:
            vocab_file.write(token + '\n')
    
    print("Simple vocabulary generation completed")

if __name__ == '__main__':
    process_all_csv_files()
    generate_simple_vocab()
