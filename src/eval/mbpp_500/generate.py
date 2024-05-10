# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import json
import os

from argparse import ArgumentParser
from typing import List
from llmchain.utils.prompter import Prompter
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llmchain.utils.prompter import Prompter


def prepare_prompt_mbpp(prompter: Prompter) -> List[str]:
    """
    Prepares prompts for MBPP (https://huggingface.co/datasets/mbpp) using a given prompter.

    Args:
        prompter (Prompter): A Prompter to wrap instructions.
    Returns:
        List[str]: A list of prepared prompts.
    """
    ds = load_dataset("mbpp", split="test")
    instructions = ds["text"]
    test_cases = ds["test_list"]

    if len(instructions) != len(test_cases):
        raise ValueError("The length of instructions and test cases must be equal")

    prompts = []

    for instruction, test_case in zip(instructions, test_cases):
        prompt = prompter.generate_prompt(
            instruction
            + "\n"
            + "the function should pass the following test code:\n"
            + "\n".join(test_case)
        )
        prompts.append(prompt)

    return prompts


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_config: GenerationConfig,
    prompter: Prompter,
) -> List[str]:
    """
    Generates text using a given model and tokenizer.

    Args:
        model (AutoModelForCausalLM): The model to generate text.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        prompt (str): The prompt to generate text from.
        generation_config (GenerationConfig): The configuration for text generation.
        prompter:Prompter

    Returns:
        List[str]: A list of generated texts.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    s = model.generate(**inputs, generation_config=generation_config)
    outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
    return [
        prompter.get_response(output).replace(tokenizer.eos_token, "")
        for output in outputs
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="", help="please enter model path here")
    parser.add_argument(
        "--save_path", default="./result", help="please enter save path here"
    )
    parser.add_argument(
        "--temperature", default="", type=float, help="generation_config"
    )
    parser.add_argument("--n_samples", default="", type=int, help="generation_config")
    parser.add_argument(
        "--max_new_tokens", default="", type=int, help="generation_config"
    )
    parser.add_argument("--batch_size", default="", type=int, help="generation_config")
    parser.add_argument("--top_p", default="", type=float, help="generation_config")

    args = parser.parse_args()
    generations = []
    prompter = Prompter()

    # prepare prompt
    prompts = prepare_prompt_mbpp(prompter)

    if not args.model_path:
        raise ValueError("Please provide a model path")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to("cuda")
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.batch_size,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
    )

    for prompt in tqdm(prompts):
        samples_n = []
        if args.n_samples % args.batch_size != 0:
            raise ValueError("n_samples must be divisible by batch_size")

        for _ in range(int(args.n_samples / args.batch_size)):
            samples_n += generate(model, tokenizer, prompt, generation_config, prompter)

        generations.append(samples_n)

    if args.save_path:
        with open(args.save_path, "w") as f1:
            json.dump(generations, f1, indent=4)


if __name__ == "__main__":
    main()
