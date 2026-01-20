"""
Script for preparing the HellaSwag data for fine-tuning an OLMo model.
"""

import logging
import re
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    dataset = ds.load_dataset("hellaswag", split="train")
    # dataset = dataset.select(np.random.RandomState(42).choice(len(dataset), size=2250, replace=False))
    print("HellaSwag train dataset size:", len(dataset))

    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=[
            "ind",
            "activity_label",
            "ctx_a",
            "ctx_b",
            "ctx",
            "endings",
            "source_id",
            "split",
            "split_type",
            "label",
        ],
        num_proc=opts.num_proc,
    )

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len
        total_tokens += len(ex["input_ids"])
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"),
        dtype=np.uint32,
        mode="w+",
        shape=(total_tokens,),
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"),
        dtype=np.bool_,
        mode="w+",
        shape=(total_tokens,),
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def preprocess_text(text):
    """Preprocess text following HellaSwag conventions."""
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    # Build context: activity_label + ctx_a + ctx_b (capitalized)
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    query = preprocess_text(example["activity_label"] + ": " + ctx)

    # Get the correct ending
    gold_idx = int(example["label"])
    answer_text = preprocess_text(example["endings"][gold_idx])

    # Tokenize query part (masked - not trained on)
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    query_mask = [False] * len(query_tokens)

    # Tokenize answer part (not masked - trained on)
    # Include EOS token at the end
    answer_tokens = tokenizer.encode(" " + answer_text + tokenizer.eos_token, add_special_tokens=False)
    answer_mask = [True] * len(answer_tokens)

    # Combine
    input_ids = query_tokens + answer_tokens
    label_mask = query_mask + answer_mask

    # Truncate if needed
    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    # Pad if needed
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare HellaSwag dataset")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default=Path(__file__).parent / "tokenizers" / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


# Example usage:
# python scripts/prepare_hellaswag_data.py /path/to/output/hellaswag \
#  -t olmo_data/tokenizers/allenai_dolma2.json \
#  -s 4096 -j 8 --eos 100257 --pad 100277
if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
