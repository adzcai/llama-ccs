# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
import pandas as pd
from tqdm import trange

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def run_csv(
    generator: LLaMA,
    csv_path: str,
    save_activations_path: str,
    max_batch_size: int,
    temperature: float,
    top_p: float,
):
    prompt_df = pd.read_csv(csv_path, index_col=0)

    # create save directories
    save_dir = Path(save_activations_path)
    plus_dir = save_dir / "plus"
    minus_dir = save_dir / "minus"
    plus_dir.mkdir(parents=True, exist_ok=True)
    minus_dir.mkdir(parents=True, exist_ok=True)

    bsz = max_batch_size // 2
    for i in trange(0, len(prompt_df), bsz):
        idx = prompt_df.index[i : i + bsz]
        plus_prompts, minus_prompts = [
            prompt_df[col].loc[idx].tolist() for col in ["plus_prompt", "minus_prompt"]
        ]

        _, activations = generator.generate(
            plus_prompts + minus_prompts,
            max_gen_len=1,
            temperature=temperature,
            top_p=top_p,
            return_activations=True,
        )

        plus_activations, minus_activations = np.split(activations, 2, axis=0)
        for j, (plus, minus) in enumerate(zip(plus_activations, minus_activations)):
            np.save(plus_dir / f"{idx[j]}", plus)
            np.save(minus_dir / f"{idx[j]}", minus)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    save_activations_path: Optional[str] = None,
    prompt_csv: Optional[str] = None,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    if prompt_csv is not None:
        run_csv(
            generator,
            prompt_csv,
            save_activations_path,
            max_batch_size,
            temperature,
            top_p,
        )
        return

    prompts = [
        # the first question from BoolQ
        "Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\nAfter reading this passage, I have a question: do iran and afghanistan speak the same language?\n\nTrue or False: True",
        "Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\nAfter reading this passage, I have a question: do iran and afghanistan speak the same language?\n\nTrue or False: False",
    ]
    results = generator.generate(
        prompts,
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        return_activations=save_activations_path is not None,
    )

    if save_activations_path is not None:
        results, activations = results
        np.save(save_activations_path, activations)
        print(f"Saved activations to {save_activations_path}")

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
