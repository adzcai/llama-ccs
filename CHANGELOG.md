# Contributions and changelog

Initial code for evaluating LLaMA on [BoolQ](https://huggingface.co/datasets/boolq) written by Alex and Oam.
- Given the option to either shoehorn LLaMA to be compatible with HuggingFace's AutoModel in order to be used with the other repos listed in [the README](./README.md), we decided to implement it ourselves instead.

Code significantly reworked by Alex 2023-03-19.
- I separated out the prompt generation step and also enabled passing keyword arguments to enable the additional behaviour. (I initially wanted to do the CCS work in another repo to leave LLaMA relatively clean, but the LLaMA repo is a lot easier to hack on.)
- Also, the dataset generator now actually checks the number of tokens using the LLaMA tokenizer, instead of using a "4 characters per token" proxy that was causing it to fail on some prompts.
- The dataset generator now saves to a CSV file (runnable as a script).
- In `example.py`, one is now able to pass a CSV dataset and it will collect the activations on all prompts in that CSV. (This is a bit simpler than the indexing and offsetting we had earlier; all of that is now handled during the dataset generation stage.)
- Activations on all of BoolQ are now stored in `data/boolq`.
