#!/usr/bin/env python3

#python code/comparison.py data/icdev models/ic_hmm_icdev.output models/ic_hmm_raw_icdev.output

import argparse
from typing import List, Tuple

def parse_obs_tag_tokens(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse whitespace-separated tokens that are either:
      - 'obs/tag' (e.g., '2/H')  -> returns obs=['2',...], tags=['H',...]
      - 'tag'     (e.g., 'H')    -> returns obs=['? ',...], tags=['H',...]
    """
    obs, tags = [], []
    for tok in text.split():
        if '/' in tok:
            o, t = tok.split('/', 1)
            obs.append(o.strip())
            tags.append(t.strip())
        else:
            obs.append('?')
            tags.append(tok.strip())
    return obs, tags

def read_seq(path: str) -> Tuple[List[str], List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return parse_obs_tag_tokens(content)

def accuracy(gold: List[str], pred: List[str]) -> float:
    if len(gold) != len(pred):
        raise ValueError(f"Length mismatch: gold={len(gold)} vs pred={len(pred)}")
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)

def main():
    ap = argparse.ArgumentParser(description="Compare model outputs against a gold tag sequence.")
    ap.add_argument("gold_file", help="Path to gold file (obs/tag tokens or just tags)")
    ap.add_argument("file1",     help="Path to first prediction file (obs/tag tokens or just tags)")
    ap.add_argument("file2",     help="Path to second prediction file (obs/tag tokens or just tags)")
    ap.add_argument("--labels", nargs="*", help="Optional tag label set for printing (unused but kept for extensibility)")
    args = ap.parse_args()

    gold_obs, gold_tags = read_seq(args.gold_file)
    _, pred1_tags = read_seq(args.file1)
    _, pred2_tags = read_seq(args.file2)

    n = len(gold_tags)
    if not (len(pred1_tags) == len(pred2_tags) == n):
        raise ValueError(f"All sequences must have the same length. "
                         f"gold={len(gold_tags)}, file1={len(pred1_tags)}, file2={len(pred2_tags)}")

    acc1 = accuracy(gold_tags, pred1_tags)
    acc2 = accuracy(gold_tags, pred2_tags)

    print(f"Total tokens: {n}")
    print(f"Accuracy file1: {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"Accuracy file2: {acc2:.4f}  ({acc2*100:.2f}%)")
    if acc1 == acc2:
        print("Both files have the same accuracy.")
    elif acc1 > acc2:
        print("file1 is closer to gold.")
    else:
        print("file2 is closer to gold.")

    # Indices where file1 is wrong and file2 is correct
    improved_indices = [
        i for i, (g, p1, p2) in enumerate(zip(gold_tags, pred1_tags, pred2_tags))
        if p1 != g and p2 == g
    ]

    print("\nTokens incorrect in file1 but correct in file2:")
    if not improved_indices:
        print("(none)")
        return

    # Pretty print header
    print("idx\tobs\tgold\tfile1\tfile2")
    for i in improved_indices:
        obs = gold_obs[i]
        g = gold_tags[i]
        p1 = pred1_tags[i]
        p2 = pred2_tags[i]
        print(f"{i}\t{obs}\t{g}\t{p1}\t{p2}")

if __name__ == "__main__":
    main()
