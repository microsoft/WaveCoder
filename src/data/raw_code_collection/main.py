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

import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from utils.kcenter_greedy import kCenterGreedy


def get_code_embedding(
    data: List[str], model_name: str, batch_size: int
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of code snippets using a sentence transformer model.

    Parameters:
    data (List[str]): A list of code snippets to embed.
    model_name (str): The name of the sentence transformer model to use.
    batch_size (int): The size of the batch for embedding generation.

    Returns:
    List[Dict[str, Any]]: A list of dictionaries with 'text' and 'embedding' keys.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=True)
    res = [{"text": t, "embedding": e.tolist()} for t, e in zip(data, embeddings)]

    return res


def coreset(embeddings: np.ndarray, num: int, seed: int) -> np.ndarray:
    """
    Select a coreset from a set of embeddings using the k-Center Greedy algorithm.

    Parameters:
    embeddings (np.ndarray): An array of embeddings.
    num (int): The number of elements to select for the coreset.

    Returns:
    np.ndarray: An array containing the coreset elements.
    """
    kcg = kCenterGreedy(X=embeddings, y=None, seed=seed)
    batch = kcg.select_batch_(model=None, already_selected=[], N=num)
    return embeddings[batch]


# Set up the argument parser


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed arguments as an object.
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings for code and create a coreset."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file containing code snippets in JSON format.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-roberta-large-v1",
        help="Pretrained model name for sentence transformer.",
    )
    parser.add_argument(
        "--coreset_size",
        type=int,
        required=True,
        help="Number of elements to include in the coreset.",
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main function to load data, generate embeddings, create a coreset, and save the results.
    """
    args = parse_args()

    # Load the dataset
    with open(args.data_path, "r") as f:
        data = f.readlines()

    data = [json.loads(d) for d in data]

    # Ensure the coreset size does not exceed the data size
    if args.coreset_size > len(data):
        raise ValueError("coreset_size exceeds the number of data entries")

    # Get code embeddings
    embeddings = get_code_embedding(data, args.model_name, args.batch_size)

    # Create a coreset from the dataset
    coreset_data = coreset(
        np.array([example["embedding"] for example in embeddings]),
        args.coreset_size,
        args.seed,
    )

    # Optionally, save the coreset to a file
    with open(args.save_path, "w") as f:
        json.dump(
            [
                {"text": embeddings[idx]["text"]}
                for idx, embedding in enumerate(coreset_data)
            ],
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
