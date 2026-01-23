#!/usr/bin/env python3
"""Cut a .npy token file to a target number of tokens."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Cut a .npy token file to target size")
    parser.add_argument("--input", help="Input .npy file path")
    parser.add_argument("--output", help="Output .npy file path")
    parser.add_argument("--tokens", type=float, required=True, help="Target number of tokens in billions")
    args = parser.parse_args()

    target_tokens = int(args.tokens * 1e9)

    data = np.memmap(args.input, dtype=np.uint32, mode="r")
    print(f"Input: {len(data):,} tokens ({len(data)/1e9:.4f}B)")

    if len(data) < target_tokens:
        print(f"Warning: Input ({len(data)/1e9:.4f}B) < target ({args.tokens}B), using all tokens")
        cut = np.array(data)
    else:
        cut = np.array(data[:target_tokens])

    print(f"Output: {len(cut):,} tokens ({len(cut)/1e9:.4f}B)")

    cut_label_mask = np.ones_like(cut, dtype=np.uint8)

    with open(args.output, "wb") as f:
        cut.tofile(f)

    with open(args.output.replace("train_ids", "label_mask"), "wb") as f:
        cut_label_mask.tofile(f)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
