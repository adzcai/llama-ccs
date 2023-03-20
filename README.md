# LLaMA-CCS

A lightly modified version of [facebookresearch/llama](https://github.com/facebookresearch/llama) that allows for saving the intermediate activations in order to run CCS.

Some experiments for [CS 229br Foundations of Deep Learning](https://boazbk.github.io/mltheoryseminar/) as taught in Spring 2023 at Harvard University by Boaz Barak (and teaching fellows Gustaf Ahdritz and Gal Kaplun).

## Setup

Make sure you have CUDA available.

```
pip install -r requirements.txt
```

To generate the BoolQ prompts, run

```
python generate_dataset.py ./data/boolq/prompts.csv --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

To evaluate a LLaMA model on the saved dataset, set the variables accordingly and run

```
torchrun --nproc_per_node $MP example.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model --save_activations_path ./data/boolq --prompt_csv ./data/prompts.csv
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## Related resources

[Discovering Latent Knowledge, in Language Models Without Supervision](https://arxiv.org/abs/2212.03827): The original paper by Collin Burns and Haotian Ye et al that proposes "Contrast-Consistent Search" (CCS).
- [collin-burns/discovering_latent_knowledge](https://github.com/collin-burns/discovering_latent_knowledge): The corresponding repository.
  - This is claimed to be quite buggy. See [Bugs of the Initial Release of CCS](https://docs.google.com/document/d/16Q8ZJFloA-x2lR65hs80rbbjX70TteCSMhuDQGcC75Q/edit?usp=sharing) by Fabien Roger.
- [How "Discovering Latent Knowledge in Language Models Without Supervision" Fits Into a Broader Alignment Scheme](https://www.lesswrong.com/posts/L4anhrxjv8j2yRKKp/how-discovering-latent-knowledge-in-language-models-without)

[What Discovering Latent Knowledge Did and Did Not Find](https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4): A writeup by Fabien Roger on takeaways from the original paper.

- [safer-ai/Exhaustive-CCS](https://github.com/safer-ai/Exhaustive-CCS): The corresponding repository. Similar to Collin Burns's but with fewer bugs.
- [Several experiments with CCS.](https://docs.google.com/document/d/1LCjjnUPN51gHl_rmCWEmmtbY-Wu1dixzOif14e-7i-U/edit)

[EleutherAI/elk](https://github.com/EleutherAI/elk): Contains many further innovations on top of CCS.
