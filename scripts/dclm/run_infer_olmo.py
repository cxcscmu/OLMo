from vllm import LLM, TokensPrompt, SamplingParams
from pathlib import Path
from tqdm import tqdm
import datasets
import json
import yaml
import sys
import re
import os

distill_prompt = """Your task is to read and paraphrase the provided text following these instructions:
- Delete clearly irrelevant content:
  - Website headers, navigation bars, or menu items (e.g., "Home | About | Contact")
  - Unrelated HTTP links (e.g., ads, trackers, developer tools)
  - Generic footers (e.g., contact info, privacy policies, unsubscribe links)
  - Empty lines or decorative elements (e.g., "---")
- Preserve all content that is relevant and meaningful:
  - Informative or independently useful
  - Related to the topic, even tangentially
  - Provides context, background, or supporting value
  - Includes technical terms, key concepts, factual details, reasoning, and examples
- Handle mixed-relevance sentences carefully:
  - Remove only the irrelevant fragment if the rest remains coherent
  - Delete the whole sentence if the remainder loses meaning
- Do not alter meaningful content unnecessarily:
  - Only delete or modify when content is clearly meaningless or off-topic
  - Preserve the original structure, logic, and depth of the text
- Do not add explanations, notes, assumptions, or claims not found in the original text
Here is the text:
{TEXT}
Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions.
Start your response immediately with "Here is a paraphrased version:" and then provide the paraphrased text."""

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=2048,
)


def make_conversation(
    example,
    prompt_column: str = "text",
    tokenizer=None,
    max_prompt_tokens=4096,
):
    prompt = []
    prompt.append(
        {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions.",
        }
    )

    text_content = example[prompt_column]

    # If tokenizer is provided, truncate based on token count
    if tokenizer is not None:
        # First, get the base prompt tokens (without the text)
        base_prompt = distill_prompt.replace("{TEXT}", "")
        base_tokens = tokenizer.encode(base_prompt)
        sys_tokens = tokenizer.encode("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions.")

        # Calculate available tokens for the text content
        available_tokens = max_prompt_tokens - len(base_tokens) - len(sys_tokens) - 50  # 50 token safety margin

        # Tokenize and truncate the text content
        text_tokens = tokenizer.encode(text_content)
        if len(text_tokens) > available_tokens:
            text_content = (
                tokenizer.decode(text_tokens[: available_tokens // 2])
                + "... "
                + tokenizer.decode(text_tokens[-(available_tokens // 2) :])
            )

    prompt.append(
        {
            "role": "user",
            "content": distill_prompt.format(TEXT=text_content),
        }
    )
    return {"prompt": prompt}


def transform(llm, conversations):
    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
    contents = []
    for output in outputs:
        content = output.outputs[0].text
        # Find the first occurrence that matches the pattern in the string
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")
    return contents


def vllm_inference(llm, dataset):
    batch_size = 8192
    transformed_texts = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference..."):
        batch = dataset[i : i + batch_size]["prompt"]
        transformed_batch = transform(llm, batch)
        transformed_texts.extend(transformed_batch)
    return transformed_texts


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def paths_from_config(config_path: Path, local_root: str | None) -> list[Path]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    data = config.get("data", {})
    paths = data.get("paths", [])
    out = []
    for path_str in paths:
        path_str = str(path_str)
        if local_root is not None:
            path_str = path_str.replace("${LOCAL_ROOT}", local_root)
        path_str = path_str.replace("dolma2-tokenizer", "text")
        path_str = path_str.replace(".npy", ".jsonl")
        out.append(Path(path_str))
    return out


def main():
    rank = int(sys.argv[1])
    print(f"Received argument: {rank}")
    llm = LLM(model="/bos/tmp7/cx_group/zichunyu/healthcare/olmo/out/synthetic_data_generator_OLMo2-1B_grpo/checkpoint-300")
    tokenizer = llm.get_tokenizer()
    config_path = Path(sys.argv[2])
    local_root = os.environ.get("LOCAL_ROOT")
    read_paths = paths_from_config(config_path, local_root)
    world_size = 6
    read_paths = [p for i, p in enumerate(read_paths) if i % world_size == rank]
    for read_file_path in tqdm(read_paths, desc="Processing files..."):
        write_file_path = Path(str(read_file_path).replace("text", "OLMo2-1B-grpo-300"))
        if write_file_path.exists():
            print(f"File already exists, skipping: {write_file_path}")
            continue
        print(f"Processing file: {read_file_path}")
        data_list = []
        tids = []
        tid = 0
        for item in load_jsonl(read_file_path):
            text = item["text"]
            for j in range(0, len(text), 7000):
                data_list.append({"text": text[j : j + 7000]})
                tids.append(tid)
            tid += 1
        dataset = datasets.Dataset.from_list(data_list).map(lambda x: make_conversation(x, tokenizer=tokenizer))
        transformed_texts = vllm_inference(llm, dataset)
        write_file_path.parent.mkdir(parents=True, exist_ok=True)
        with write_file_path.open("w") as f:
            print(f"Writing to file: {write_file_path}")
            prev_tid = 0
            texts = []
            cnt = 0
            for tid, text in zip(tids, transformed_texts):
                if tid == prev_tid:
                    texts.append(text)
                else:
                    cur_text = " ".join(texts)
                    if cur_text == "":
                        cnt += 1
                    f.write(json.dumps({"text": cur_text}) + "\n")
                    prev_tid = tid
                    texts = [text]
            cur_text = " ".join(texts)
            if cur_text == "":
                cnt += 1
            f.write(json.dumps({"text": cur_text}) + "\n")
            print(f"Total {len(transformed_texts)} items, {cnt} invalid items found.")


if __name__ == "__main__":
    main()
