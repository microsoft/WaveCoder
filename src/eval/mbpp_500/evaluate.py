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

import re
import json
import os

import subprocess
import argparse

from typing import List
from tqdm import tqdm
from evaluate import load


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def read_data(path: str) -> List:
    """Read data from a JSON file.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        List: The parsed data from the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def generate_py_file(reference_path: str, gen_code_path: str, save_path: str):
    """Generate Python files from the generated code.

    Args:
        reference_path (str): The file path to the reference JSON file.
        gen_code_path (str): The file path to the generated code JSON file.
        save_path (str): The directory path to save the generated Python files.
    """
    references = read_data(reference_path)
    generated_code = read_data(gen_code_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise ValueError("The save path already exists. Please provide a new path.")

    if len(references) != len(generated_code):
        raise ValueError(
            "The length of references list must be equal to the length of generated code list"
        )

    for i, reference in enumerate(references):
        case_path = os.path.join(save_path, f"case_{i}")
        if not os.path.exists(case_path):
            os.makedirs(case_path)

        code_candidate = generated_code[i]
        for j, code in enumerate(code_candidate):
            file_path = os.path.join(case_path, f"gen_{j}.py")
            with open(file_path, "w") as file:
                file.write(code.replace("\t", "    "))  # Replace tabs with spaces
                file.write("\n")
                file.write(reference)


def run_generated_py_file(reference_path: str, gen_code_path: str, scripts_folder: str):
    """Run the generated Python files and log the analysis.

    Args:
        reference_path (str): The file path to the reference JSON file.
        gen_code_path (str): The file path to the generated code JSON file.
        scripts_folder (str): The directory path where Python scripts are saved.
    """
    generate_py_file(reference_path, gen_code_path, scripts_folder)

    file_dirs = os.listdir(scripts_folder)
    for id, file_dir in enumerate(
        tqdm(sorted(file_dirs, key=lambda x: int("".join(re.findall(r"\d+", x)))))
    ):
        python_files = [
            f for f in os.listdir(f"{scripts_folder}{file_dir}") if f.endswith(".py")
        ]
        with open(f"{scripts_folder}/solution_information.txt", "a") as solution:
            solution.write(f"*****************Problem {id}*****************\n")
            for sid, file in enumerate(python_files):
                file_path = f"{scripts_folder}{file_dir}/{file}"
                with open(file_path, "r") as code_file:
                    code = code_file.read()
                try:
                    subprocess.run(
                        ["python", file_path],
                        check=True,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        timeout=10,
                    )
                    status = "passed"
                    error_type = "None"
                except subprocess.CalledProcessError as e:
                    status = "failed"
                    error_type = e.stderr
                except subprocess.TimeoutExpired as e1:
                    print(f"Timeout for problem {sid}")
                finally:
                    solution_information = f"Solution {sid}:\n\nStatus:{status}\nError:{error_type}\n\nCode:\n{code}\n\n"
                    solution.write(solution_information)


def pass_k_evaluation(reference_path: str, gen_code_path: str) -> List:
    """Evaluate the accuracy of the generated code.

    Args:
        reference_path (str): The file path to the reference JSON file.
        gen_code_path (str): The file path to the generated code JSON file.

    Returns:
        List: The evaluation results.
    """
    code_metric = load("code_eval")
    references = read_data(reference_path)
    generated_code = read_data(gen_code_path)
    results, _ = code_metric.compute(references=references, predictions=generated_code)
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze and run generated code.")

    parser.add_argument(
        "--reference_path", type=str, help="Path to the reference JSON file."
    )
    parser.add_argument(
        "--gen_code_path", type=str, help="Path to the generated code JSON file."
    )
    parser.add_argument(
        "--analyze_generation",
        action="store_true",
        help="If true, generate analysis for each problem.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./code_run/",
        help="Folder to save and run Python scripts.",
    )

    args = parser.parse_args()
    if args.analyze_generation:
        run_generated_py_file(args.reference_path, args.gen_code_path, args.save_path)

    results = pass_k_evaluation(args.reference_path, args.gen_code_path)
    print(results)


if __name__ == "__main__":
    main()
