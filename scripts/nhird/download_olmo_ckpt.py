import argparse
import csv
import os
from pathlib import Path
import requests
from tqdm import tqdm
from urllib.parse import urljoin


def download_file(url, save_path, chunk_size=8192):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=save_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def try_get_directory_listing(url):
    common_files = ["config.yaml", "train.pt", "model.safetensors", "optim.safetensors", "model.pt", "optim.pt"]
    found_files = []
    for pattern in common_files:
        try:
            test_url = urljoin(url.rstrip("/") + "/", pattern)
            response = requests.head(test_url)
            response.raise_for_status()
            found_files.append(pattern)
        except requests.exceptions.HTTPError as e:
            if response.status_code != 404:
                raise
        except requests.exceptions.RequestException:
            raise
    if len(found_files) <= 0:
        raise ValueError(f"No checkpoint files found at {url}")

    return found_files


def download_checkpoint(url, save_dir):
    base_path = Path(save_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {base_path}")
    available_files = try_get_directory_listing(url)

    if not available_files:
        raise ValueError("Matching files not found in directory")

    failed_files = []
    for file in available_files:
        file_url = urljoin(url.rstrip("/") + "/", file)
        file_path = base_path / file
        try:
            print(f"\nDownloading: {file}")
            download_file(file_url, file_path)
        except requests.exceptions.Timeout:
            print(f"Timeout error for {file}, retryin...")
            try:
                download_file(file_url, file_path)
            except requests.exceptions.RequestException as e:
                failed_files.append(file)
                print(f"Failed to download {file}: {e}")
        except requests.exceptions.RequestException as e:
            failed_files.append(file)
            print(f"Failed to download {file}: {e}")
    if failed_files:
        print(f"\nFAILED to download these files: {failed_files}")


def main():
    """
    Example usage:
    python scripts/nhird/download_ckpt.py download configs/official-0425/OLMo-2-0425-1B.csv --step 1907359
    """
    parser = argparse.ArgumentParser(description="Download OLMo checkpoints")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    download_parser = subparsers.add_parser("download", help="Download checkpoints from CSV file")
    download_parser.add_argument("csv_file", type=str, help="Path to the CSV file containing checkpoint URLs")
    download_parser.add_argument("--step", type=str, required=True, help="Specific step number to download")
    download_parser.add_argument(
        "--save-dir",
        type=str,
        default="/tmp/olmo/pretrained_ckpt/OLMo-2-0425-1B_stage1",
        help="Base directory to save downloaded checkpoints",
    )
    list_parser = subparsers.add_parser("list", help="List available checkpoint steps")
    list_parser.add_argument("csv_file", type=str, help="Path to the CSV file containing checkpoint URLs")
    args = parser.parse_args()

    print(f"Reading CSV file: {args.csv_file}")

    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        urls = [(row["Step"], row["Checkpoint Directory"]) for row in reader]

    if args.command == "list":
        print("Available steps:")
        for step, _ in urls:
            print(f"Step {step}")
        return
    elif args.step:
        urls = [(step, url) for step, url in urls if step == args.step]
        if not urls:
            print(f"Error: Step {args.step} not found in the CSV file.")
            print("Use list argument to see available checkpoint steps.")
            return

    print(f"Saving checkpoints to: {args.save_dir}")
    for step, url in urls:
        print(f"\nStep {step}:")
        print(f"URL: {url}")
        save_path = os.path.join(args.save_dir, f"step{step}")
        download_checkpoint(url, save_path)


if __name__ == "__main__":
    main()
