#!/usr/bin/env python3
"""Utilities for working with .npy token files."""

import argparse
import numpy as np
from pathlib import Path


def cut_npy(input_path, output_path, target_tokens_billions):
    """Cut a .npy token file to a target number of tokens."""
    target_tokens = int(target_tokens_billions * 1e9)

    data = np.memmap(input_path, dtype=np.uint32, mode="r")
    print(f"Input: {len(data):,} tokens ({len(data)/1e9:.4f}B)")

    if len(data) < target_tokens:
        print(f"Warning: Input ({len(data)/1e9:.4f}B) < target ({target_tokens_billions}B), using all tokens")
        cut = np.array(data)
    else:
        cut = np.array(data[:target_tokens])

    print(f"Output: {len(cut):,} tokens ({len(cut)/1e9:.4f}B)")

    cut_label_mask = np.ones_like(cut, dtype=np.uint8)

    with open(output_path, "wb") as f:
        cut.tofile(f)

    with open(output_path.replace("train_ids", "label_mask"), "wb") as f:
        cut_label_mask.tofile(f)

    print(f"Saved to {output_path}")


def concatenate_npy(input_paths, output_path):
    """Concatenate multiple .npy files into a single file."""
    print(f"Concatenating {len(input_paths)} files...")
    all_tokens = []
    seq_len = 2048

    for i, path in enumerate(input_paths, 1):
        print(f"[{i}/{len(input_paths)}] {path}")
        tokens = np.memmap(path, dtype=np.uint32, mode='r')
        num_tokens = len(tokens)

        # Drop misaligned tokens from this file
        num_complete_instances = num_tokens // seq_len
        aligned_length = num_complete_instances * seq_len
        dropped = num_tokens - aligned_length

        if dropped > 0:
            print(f"  {num_tokens:,} tokens - ✗ (dropping {dropped} tokens)")
            tokens_aligned = tokens[:aligned_length].copy()
        else:
            print(f"  {num_tokens:,} tokens - ✓")
            tokens_aligned = tokens[:].copy()

        all_tokens.append(tokens_aligned)

    concatenated = np.concatenate(all_tokens).astype(np.uint32)
    num_complete_instances = len(concatenated) // seq_len
    print(f"\nFinal aligned tokens: {len(concatenated):,} ({num_complete_instances:,} complete instances)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        concatenated.tofile(f)

    print(f"Saved to: {output_path}")
    print(f"Size: {Path(output_path).stat().st_size / (1024**3):.2f} GB")


# python npy_utils.py cut --input input.npy --output output.npy --tokens 1.5
# python npy_utils.py concat --input file1.npy file2.npy file3.npy --output combined.npy
def main():
    parser = argparse.ArgumentParser(description="Utilities for working with .npy token files")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Cut subcommand
    cut_parser = subparsers.add_parser("cut", help="Cut a .npy token file to target size")
    cut_parser.add_argument("--input", required=True, help="Input .npy file path")
    cut_parser.add_argument("--output", required=True, help="Output .npy file path")
    cut_parser.add_argument("--tokens", type=float, required=True, help="Target number of tokens in billions")

    # Concatenate subcommand
    concat_parser = subparsers.add_parser("concat", help="Concatenate .npy files")
    concat_parser.add_argument("--input", nargs="+", required=True, help="Input .npy files")
    concat_parser.add_argument("--output", required=True, help="Output path")

    args = parser.parse_args()

    if args.command == "cut":
        cut_npy(args.input, args.output, args.tokens)
    elif args.command == "concat":
        concatenate_npy(args.input, args.output)


if __name__ == "__main__":
    main()
