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
        tokens = np.memmap(path, dtype=np.uint32, mode="r")
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


def concat_paired_and_subsample(input_paths, label_mask_paths, output_dir, num_instances, seq_len=2048, seed=42):
    """Concatenate input_ids/label_mask files and subsample complete instances.

    If --label-mask is omitted, this creates an all-ones mask for every input shard.
    """
    use_provided_masks = label_mask_paths is not None
    if use_provided_masks and len(input_paths) != len(label_mask_paths):
        raise ValueError("--input and --label-mask must have the same number of files")

    all_input_chunks = []
    all_mask_chunks = []
    total_source_instances = 0

    print(f"Concatenating {len(input_paths)} files...")
    for i, input_path in enumerate(input_paths, 1):
        print(f"[{i}/{len(input_paths)}]")
        print(f"  input_ids : {input_path}")
        mask_path = label_mask_paths[i - 1] if use_provided_masks else None
        if use_provided_masks:
            print(f"  label_mask: {mask_path}")
        else:
            print("  label_mask: <generated ones_like>")

        input_ids = np.memmap(input_path, dtype=np.uint32, mode="r")
        if use_provided_masks:
            label_mask = np.memmap(mask_path, dtype=np.bool_, mode="r")
            if len(input_ids) != len(label_mask):
                raise ValueError(
                    f"Length mismatch for pair {i}: input_ids has {len(input_ids):,} tokens, "
                    f"label_mask has {len(label_mask):,} tokens"
                )

        num_tokens = len(input_ids)
        num_complete_instances = num_tokens // seq_len
        aligned_length = num_complete_instances * seq_len
        dropped = num_tokens - aligned_length
        total_source_instances += num_complete_instances

        if dropped > 0:
            print(f"  tokens={num_tokens:,} -> aligned={aligned_length:,} (dropping {dropped:,} tokens)")
        else:
            print(f"  tokens={num_tokens:,} -> aligned={aligned_length:,}")

        all_input_chunks.append(np.array(input_ids[:aligned_length], dtype=np.uint32))
        if use_provided_masks:
            all_mask_chunks.append(np.array(label_mask[:aligned_length], dtype=np.bool_))
        else:
            all_mask_chunks.append(np.ones(aligned_length, dtype=np.bool_))

    if not all_input_chunks:
        raise ValueError("No input files provided")

    concatenated_input = np.concatenate(all_input_chunks)
    concatenated_mask = np.concatenate(all_mask_chunks)
    assert len(concatenated_input) == len(concatenated_mask)

    total_instances = len(concatenated_input) // seq_len
    print(f"\nTotal aligned instances before subsample: {total_instances:,}")
    if total_instances == 0:
        raise ValueError("No complete instances found after alignment")
    if num_instances > total_instances:
        raise ValueError(f"Requested {num_instances:,} instances, but only {total_instances:,} are available")

    input_2d = concatenated_input.reshape(total_instances, seq_len)
    mask_2d = concatenated_mask.reshape(total_instances, seq_len)

    rng = np.random.default_rng(seed=seed)
    selected = np.sort(rng.choice(total_instances, size=num_instances, replace=False))
    sampled_input = input_2d[selected].reshape(-1)
    sampled_mask = mask_2d[selected].reshape(-1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_out = out_dir / "input_ids.npy"
    mask_out = out_dir / "label_mask.npy"

    with open(input_out, "wb") as f:
        sampled_input.astype(np.uint32).tofile(f)
    with open(mask_out, "wb") as f:
        sampled_mask.astype(np.bool_).tofile(f)

    print("\nSaved subsampled paired dataset:")
    print(f"  output_dir      : {out_dir}")
    print(f"  input_ids.npy   : {input_out} ({len(sampled_input):,} tokens)")
    print(f"  label_mask.npy  : {mask_out} ({len(sampled_mask):,} tokens)")
    print(f"  seq_len         : {seq_len}")
    print(f"  num_instances   : {num_instances}")
    print(f"  seed            : {seed}")


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

    # Concatenate paired and subsample subcommand
    paired_parser = subparsers.add_parser("concat-paired-subsample", help="Concat paired input_ids and subsample")
    paired_parser.add_argument("--input", nargs="+", required=True, help="Input input_ids.npy files")
    paired_parser.add_argument("--label-mask", nargs="+", help="label_mask. If omitted, creates all-ones masks.")
    paired_parser.add_argument("--output-dir", help="Output directory for combined input_ids.npy/label_mask.npy")
    paired_parser.add_argument("--num-instances", type=int, default=512, help="Number of instances to subsample")
    paired_parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    paired_parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")

    args = parser.parse_args()

    if args.command == "cut":
        cut_npy(args.input, args.output, args.tokens)
    elif args.command == "concat":
        concatenate_npy(args.input, args.output)
    elif args.command == "concat-paired-subsample":
        concat_paired_and_subsample(
            input_paths=args.input,
            label_mask_paths=args.label_mask,
            output_dir=args.output_dir,
            num_instances=args.num_instances,
            seq_len=args.seq_len,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
