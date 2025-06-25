#!/usr/bin/env python3
"""Khmer Character Analysis Script"""
import os
import sys
import json
from collections import Counter

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "modules"))

try:
    from khtext.khchar import *
    print(" Successfully imported khchar")
except ImportError as e:
    print(f" Error: {e}")
    sys.exit(1)

def analyze_corpus():
    corpus_file = "data/khmer_clean_text.txt"
    if not os.path.exists(corpus_file):
        print(f" File not found: {corpus_file}")
        return
    
    print(" Reading text corpus...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        f.seek(0, 2)
        total_size = f.tell()
        f.seek(0)
        text_sample = f.read(1000000)  # 1MB sample
    
    print(f" Corpus size: {total_size:,} bytes")
    print(f" Analyzing {len(text_sample):,} characters...")
    
    # Find Khmer characters
    text_chars = set()
    char_freq = Counter()
    
    for char in text_sample:
        if 0x1780 <= ord(char) <= 0x17FF:
            char_code = ord(char)
            text_chars.add(char_code)
            char_freq[char_code] += 1
    
    defined_chars = set(ALL_CHARS)
    in_both = text_chars & defined_chars
    missing = text_chars - defined_chars
    
    coverage = len(in_both) / len(text_chars) * 100 if text_chars else 0
    
    print(f"\n RESULTS:")
    print(f"  Unique Khmer chars in text: {len(text_chars)}")
    print(f"  Characters in definitions: {len(defined_chars)}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Missing from definitions: {len(missing)}")
    
    if missing:
        print(f"\n MISSING CHARACTERS:")
        missing_sorted = sorted(missing, key=lambda x: char_freq[x], reverse=True)
        for char_code in missing_sorted:
            char = chr(char_code)
            freq = char_freq[char_code]
            print(f"  U+{char_code:04X} {char} (freq: {freq:,})")
    
    # Save results
    results = {
        "total_file_size": total_size,
        "sample_size": len(text_sample),
        "text_chars_count": len(text_chars),
        "defined_chars_count": len(defined_chars),
        "coverage": coverage,
        "missing_count": len(missing),
        "missing_chars": sorted(list(missing)),
        "top_frequencies": [(char, freq) for char, freq in char_freq.most_common(30)]
    }
    
    with open("khmer_text_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Results saved to: khmer_text_analysis_results.json")
    
    return results

if __name__ == "__main__":
    analyze_corpus()

