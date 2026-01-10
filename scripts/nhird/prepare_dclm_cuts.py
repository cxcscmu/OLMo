#!/usr/bin/env python3
"""Prepare DCLM data cuts for 8phases_selective training.

This script:
1. Reads all phase*.yaml files under configs/nhird/8phases_selective/
2. Extracts NHIRD token count and DCLM file paths from each
3. Concatenates the DCLM files per phase
4. Cuts to exact token count specified in the comment
5. Saves to ${LOCAL_ROOT}/data/preprocessed/dclm/train_ids_olmo_<size>.npy
"""

import os
import re
from pathlib import Path

import numpy as np
import yaml

# Get LOCAL_ROOT from environment
LOCAL_ROOT = os.environ.get("LOCAL_ROOT")
if LOCAL_ROOT is None:
    raise ValueError("LOCAL_ROOT environment variable is not set. Please source .env")

# Directory containing phase configs
config_dir = Path("configs/nhird/8phases_selective")
output_dir = Path(LOCAL_ROOT) / "data/preprocessed/dclm"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Preparing DCLM Data Cuts for 8-Phase Selective Training")
print("=" * 80)
print(f"Config directory: {config_dir}")
print(f"Output directory: {output_dir}")
print("=" * 80)

# Find all phase yaml files
phase_files = sorted(config_dir.glob("phase*.yaml"))
print(f"\nFound {len(phase_files)} phase files:")
for pf in phase_files:
    print(f"  - {pf.name}")

# Process each phase
for phase_file in phase_files:
    phase_name = phase_file.stem  # e.g., "phase3"
    print(f"\n{'=' * 80}")
    print(f"Processing {phase_name}")
    print(f"{'=' * 80}")

    # Read the yaml file
    with open(phase_file, "r") as f:
        content = f.read()
        config = yaml.safe_load(content)

    # Find the data paths section in the raw content
    # We need to extract the comment with token count
    paths_section = content.split("paths:")[1].split("\n")

    nhird_token_count = None
    dclm_paths = []

    for line in paths_section:
        # Look for NHIRD comment with token count
        if "#SOURCE: NHIRD" in line:
            match = re.search(r"(\d+\.?\d*)BT", line)
            if match:
                nhird_token_count = float(match.group(1))
                print(f"Target token count: {nhird_token_count}B tokens")

        # Look for DCLM paths
        if "/preprocessed/dclm" in line:
            # Extract path, replacing ${LOCAL_ROOT} with actual value
            path_str = line.strip()[2:].strip()  # Remove "- " prefix
            path_str = path_str.replace("${LOCAL_ROOT}", LOCAL_ROOT)
            dclm_paths.append(path_str)

    if nhird_token_count is None:
        print(f"Warning: Could not find token count for {phase_name}, skipping...")
        continue

    print(f"\nDCLM source files:")
    for i, path in enumerate(dclm_paths, 1):
        print(f"  {i}. {Path(path).name}")

    # Concatenate the DCLM files
    print(f"\nConcatenating {len(dclm_paths)} DCLM files...")
    all_tokens = []

    for i, path in enumerate(dclm_paths, 1):
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            exit(0)

        data = np.memmap(path, dtype=np.uint32, mode="r")
        print(f"  File {i}: {len(data):,} tokens ({len(data) / 1e9:.4f}B)")
        all_tokens.append(np.array(data))

    # Concatenate all tokens
    concatenated = np.concatenate(all_tokens)
    print(f"\nTotal concatenated: {len(concatenated):,} tokens ({len(concatenated) / 1e9:.4f}B)")

    # Cut to target size
    target_tokens = int(nhird_token_count * 1e9)
    print(f"Target size: {target_tokens:,} tokens ({target_tokens / 1e9:.4f}B)")

    if len(concatenated) < target_tokens:
        print(f"Warning: Concatenated data ({len(concatenated):,}) is smaller than target ({target_tokens:,})")
        print(f"Using all available tokens instead")
        cut_tokens = concatenated
    else:
        cut_tokens = concatenated[:target_tokens]
        print(f"Cut to: {len(cut_tokens):,} tokens ({len(cut_tokens) / 1e9:.4f}B)")

    # Save to output file
    output_filename = f"train_ids_olmo_{nhird_token_count}B.npy"
    output_path = output_dir / output_filename
    print(f"\nSaving to: {output_path}")

    with open(output_path, "wb") as f:
        cut_tokens.tofile(f)

    # Verify
    verify_data = np.memmap(output_path, dtype=np.uint32, mode="r")
    print(f"Verification: {len(verify_data):,} tokens written")

    print(f"✓ {phase_name} complete!")

print(f"\n{'=' * 80}")
print("All phases processed successfully!")
print(f"{'=' * 80}")
print(f"\nOutput files saved to: {output_dir}")
print("\nGenerated files:")
for npy_file in sorted(output_dir.glob("train_ids_olmo_*.npy")):
    size = np.memmap(npy_file, dtype=np.uint32, mode="r").shape[0]
    print(f"  - {npy_file.name}: {size:,} tokens ({size / 1e9:.4f}B)")
print("=" * 80)
