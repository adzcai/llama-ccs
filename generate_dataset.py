import fire
from typing import Optional
from datasets import load_dataset
import pandas as pd
from tqdm import trange
import os

from llama import Tokenizer


def format_boolq(passage, question):
    """Prepares BoolQ prompts for running CCS."""
    return [
        f"{passage}\n\nAfter reading this passage, I have a question: {question}?\n\nTrue or False: {answer}"
        for answer in ("True", "False")
    ]


def load_prompt_dataset(
    tokenizer_path: str,
    num_prompts: Optional[int] = None,
    offset_idx: Optional[int] = 0,
    max_seq_len: int = 512,
    split: str = "train",
):
    """
    Loads the first num_prompts prompts from the BoolQ training dataset.
    For each one, we create two prompts, one with the answer set to "True" and one with the answer set to "False".
    All even-indexed prompts will have the answer set to "True" and all odd-indexed prompts will have the answer set to "False".
    """
    dataset = load_dataset("boolq", split=split)

    tokenizer_path = os.path.expanduser(tokenizer_path)
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    num_pruned = []
    idx_ary = (
        trange(offset_idx, offset_idx + num_prompts)
        if num_prompts is not None
        else trange(offset_idx, len(dataset))
    )

    plus_prompts, minus_prompts, labels, indices = [], [], [], []

    for i in idx_ary:
        row = dataset[i]
        prompt_plus, prompt_minus = format_boolq(row["passage"], row["question"])
        tokens = tokenizer.encode(prompt_plus, bos=True, eos=False)

        if len(tokens) + 4 > max_seq_len:  # add a small buffer
            num_pruned.append(i)
            continue

        plus_prompts.append(prompt_plus)
        minus_prompts.append(prompt_minus)
        labels.append(row["answer"])
        indices.append(i)

    print(
        f"Pruned {len(num_pruned)} prompts due to length (absolute indices {', '.join(str(i) for i in num_pruned)})"
    )

    return pd.DataFrame(
        {
            "idx": indices,
            "plus_prompt": plus_prompts,
            "minus_prompt": minus_prompts,
            "label": labels,
        },
    )


def main(
    output_file: str,
    tokenizer_path: str,
    offset_idx: int = 0,
    num_prompts: Optional[int] = None,
    max_seq_len: int = 512,
):
    df = load_prompt_dataset(tokenizer_path, num_prompts, offset_idx, max_seq_len)
    df.to_csv(output_file, index=False)
    print(f"Saved prompts to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
