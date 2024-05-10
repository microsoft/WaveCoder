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

import openai
import os
import ast
import random
import json
import re
import time
import ast
import http.client
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from sampler import GoodCase, BadCase
from utils import make_request


def load_source_data(path: str) -> List[Dict]:
    """
    Load source code data from a JSON file.

    Parameters:
    - path (str): The file path to the source code data file.

    Returns:
    - List[Dict]: A list of dictionaries containing source code data.
    """
    with open(path, "r") as f_s:
        ds = json.load(f_s)
    return [{"id": id, "text": d["text"]["code"]} for id, d in enumerate(ds)]


def load_prompt(path: str) -> str:
    """
    Load a prompt text from a file.

    Parameters:
    - path (str): The file path to the prompt text file.

    Returns:
    - str: The loaded prompt text.
    """
    with open(path, "r") as f1:
        prompt = f1.readlines()
    return "\n".join(prompt)


def analysis_filter(text: str) -> (str, str):
    """
    Filter out the analysis part from a text.

    Parameters:
    - text (str): The input text that may contain an analysis section.

    Returns:
    - (str, str): A tuple containing the filtered text and the analysis content if found.
    """
    pattern = r"(Analysis:[\s\S]*?)(?=Analysis:|$)"
    match = re.search(pattern, text)
    if not match:
        return text, None

    start_pos = match.start()
    prev_info = text[:start_pos]
    content = text[start_pos:]
    return prev_info.strip(), content


def extract_message(generated_text: str) -> Dict:
    """
    Extract message parts from the generated text.

    Parameters:
    - generated_text (str): The generated text from the AI model.

    Returns:
    - Dict: A dictionary containing the extracted message parts and their quality.
    """

    pattern = r"task_name:\s*(.*?)\s*instruction:\s*(.*?)\s*information:\s*(.*?)\s*solution:\s*(.*?)(?=\ntask_name:|\Z)"
    matches = re.findall(pattern, generated_text, re.DOTALL)

    # assert len(matches) <= 1
    if len(matches) > 1:
        print(f"INFO: matches={len(matches)}\ngenerated_text={generated_text}\n\n")
        return {"generated_text": generated_text, "quality": False, "result": None}

    if len(matches) == 0:
        return {"generated_text": generated_text, "quality": False, "result": None}

    try:
        task_name, instruction, information, solution = matches[0]
        # save the information in the dict
        result = {
            "task_name": task_name,
            "instruction": instruction,
            "information": information,
            "solution": solution,
        }
    except Exception as e:
        print(e)
        return None

    noinput_num = 0
    for key, value in result.items():
        if value.strip() in ["<noinput>", "<noinput>."]:
            noinput_num += 1
    if noinput_num > 1:
        print(f"noinput={noinput_num}")
        return {"generated_text": generated_text, "quality": False, "result": None}

    # check sentence numbers
    instruction_sentence = re.split(r"[.!?](?:\s|$)", instruction)
    instruction_sentence = [item for item in instruction_sentence if item != ""]
    print(f"instruction_sentence_num={len(instruction_sentence)}")

    if len(instruction_sentence) < 1 or len(instruction_sentence) > 2:
        return {"generated_text": generated_text, "quality": False, "result": result}

    # information
    if information.strip().lower() in [
        "none",
        "none.",
        "none,",
        "no additional information is required",
        "",
    ]:
        result["information"] = "<noinput>"
        information = "<noinput>"

    if information == "<noinput>":
        information_sentence_num = 0
    else:
        information_sentence = re.split(r"[.!?](?:\s|$)", information)
        information_sentence = [item for item in information_sentence if item != ""]
        information_sentence_num = len(information_sentence)
    # count sentence number
    print(f"information_sentence_num={information_sentence_num}")
    if information_sentence_num > 2 or information_sentence_num < 0:
        return {"generated_text": generated_text, "quality": False, "result": result}

    gen_text = ""
    # print(f"result={result}")
    for key, value in result.items():
        gen_text += f"{key}: {value}\n"

    return {"generated_text": gen_text, "quality": True, "result": result}


def extract_answer(feedback: str) -> (str, List[str], str):
    """
    Extract answers from the feedback text.

    Parameters:
    - feedback (str): The feedback text containing answers.

    Returns:
    - (str, List[str], str): A tuple containing the feedback, a list of part answers, and the overall answer.
    """

    # extract feedback
    pattern = r"(- step 1:.*?- reasons:.*?)\n"
    match = re.search(pattern, feedback + "\n", flags=re.DOTALL)
    # print(f"match={match.groups()}\n\n")
    if match:
        feedback = match.group(1)
    else:
        print("No match found")
        return None, None, None

    # extract answer
    answers = re.findall("<answer:\s*(yes|no)(?:,.*?)?>", feedback)
    overall_answer = re.search("-\s*Overall answer:\s*(.*?)\n", feedback).group(1)
    overall_answer = overall_answer.lower()

    part_answer = []
    for answer in answers:
        part_answer.append(answer.lower())

    return feedback, part_answer, overall_answer.strip()


def few_shot_task_gen(
    source_code: List[Dict],
    gen_prompt: str,
    dis_prompt: str,
    good_case_path: str,
    bad_case_path: str,
    sample_number: int,
    engine: str,
    api_key: str,
    base_url: str,
    gen_max_token: int = 800,
    data_stream_path: str = "data_stream.txt",
) -> List[Dict]:
    """
    Generate few-shot tasks and process the data.

    Parameters:
    - source_code (List[Dict]): List of source code data.
    - gen_prompt (str): The generator prompt text.
    - dis_prompt (str): The discriminator prompt text.
    - good_case_path (str): Path to save good cases.
    - bad_case_path (str): Path to save bad cases.
    - sample_number (int): Number of samples for few-shot learning.
    - engine (str): The OpenAI engine to use.
    - api_key (str): The API key for OpenAI.
    - base_url (str): The base URL for the OpenAI API.
    - gen_max_token (int): Maximum token length for the generator.
    - data_stream_path (str): Path to save the data stream.

    Returns:
    - List[Dict]: A list of dictionaries containing the generated data.
    """

    good_prompt = "Here are some good examples:\n"
    bad_prompt = "Here are some bad examples. In each example, I also provide an <Analysis> pointing out the reasons why the case is not good. Please do not generate data like this.\n"
    #
    GoodCaser = GoodCase(good_case_path, good_prompt, sample_number=sample_number)
    BadCaser = BadCase(bad_case_path, bad_prompt, sample_number=sample_number)
    print(f"Good Case: {len(GoodCaser.sample_list)}")
    print(f"Bad Case: {len(BadCaser.sample_list)}")
    g_prompt = gen_prompt
    d_prompt = dis_prompt

    data_list = []
    with open(data_stream_path, "w") as f:
        for data in tqdm(source_code):
            ids = data["id"]
            # print("The code for example " + str(ids+1) + " is generating.")

            example_code = data["text"]

            good_few_shot = GoodCaser.generate_fewshot_text()
            bad_few_shot = BadCaser.generate_fewshot_text()
            example = {
                "good_few_shot": good_few_shot,
                "bad_few_shot": bad_few_shot,
                "input": example_code,
            }

            message = g_prompt.format_map(example)
            print(message)
            text = make_request(
                message=message, model=engine, api_key=api_key, base_url=base_url
            )
            if not text:
                raise Exception("No text generated")
            # print(text)
            # analysis filter
            text, analysis_content = analysis_filter(text)
            filter_messages = extract_message(text)
            if filter_messages["quality"] == False:
                continue
            generated_text = filter_messages["generated_text"]
            example_case = f"Input: \n{example_code}\n\nOutput:\n{generated_text}"

            # example_case = f"Output:\n{text}"
            ans = discriminator(
                prompt=d_prompt,
                GoodCaser=GoodCaser,
                BadCaser=BadCaser,
                engine=engine,
                api_key=api_key,
                base_url=base_url,
                generated_text=example_case,
            )
            if "no" not in ans:
                result_data = filter_messages["result"]
                assert result_data != None and len(result_data) > 0
                data_list.append(
                    {
                        "id": ids,
                        "task_type": "code generation",
                        "source_code": example_code,
                        "generation_data": result_data,
                    }
                )
                f.write(
                    json.dumps(
                        {
                            "id": ids,
                            "task_type": "code generation",
                            "source_code": example_code,
                            "generation_data": result_data,
                        }
                    )
                )
                f.write(",\n")

                time.sleep(2)

    if not data_list:
        raise Exception("No data generated")

    else:
        print(f"data_list={len(data_list)}\nsome examples are:")
        for key, value in data_list[0].items():
            print(f"{key}:{value}")

    return data_list


def discriminator(
    prompt: str,
    GoodCaser: GoodCase,
    BadCaser: BadCase,
    engine: str,
    api_key: str,
    base_url: str,
    generated_text: str,
    max_token: int = 500,
) -> str:
    """
    Use discriminator to classify the generated text as good or bad.

    Parameters:
    - prompt (str): The discriminator prompt text.
    - GoodCaser (GoodCase): An instance of GoodCase for good examples.
    - BadCaser (BadCase): An instance of BadCase for bad examples.
    - engine (str): The OpenAI engine to use.
    - api_key (str): The API key for OpenAI.
    - base_url (str): The base URL for the OpenAI API.
    - generated_text (str): The generated text to be classified.
    - max_token (int): Maximum token length for the discriminator.

    Returns:
    - str: The overall answer from the discriminator.
    """

    prompt_format = """
{prompt}\n{bad_examples}\n\n{generated_text}\n\nAnalysis:\n
    """

    good_examples = GoodCaser.generate_fewshot_for_d()
    bad_examples = BadCaser.generate_fewshot_for_d()
    # generate instruction
    example = {
        "prompt": prompt,
        "bad_examples": bad_examples,
        "generated_text": generated_text,
    }
    message = prompt_format.format_map(example)
    # obtain answer
    ans = make_request(
        message=message, model=engine, api_key=api_key, base_url=base_url
    )
    feedback, part_answer, overall_answer = extract_answer(ans)
    print(f"part_answer={part_answer}\toverall_answer={overall_answer}")

    if feedback is None:
        print("no pattern information was extracted in the feedback")
        return None

    generated_text = f"{generated_text}\n\nAnalysis:\n{feedback}\n"

    if overall_answer == "yes":
        GoodCaser.add_case(generated_text)
    else:
        BadCaser.add_case(generated_text)

    return overall_answer


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process few-shot tasks and generate data."
    )
    parser.add_argument(
        "--source_data_path",
        type=str,
        default="fewshot_case/source_code.json",
        help="Path to the source code data file.",
    )
    parser.add_argument(
        "--gen_prompt_path",
        type=str,
        default="prompt/generator.txt",
        help="Path to the generator prompt text file.",
    )
    parser.add_argument(
        "--dis_prompt_path",
        type=str,
        default="prompt/discriminator.txt",
        help="Path to the discriminator prompt text file.",
    )
    parser.add_argument(
        "--good_case_path",
        type=str,
        default="prompt/generator.txt",
        help="Path to save good case.",
    )
    parser.add_argument(
        "--data_stream_path",
        type=str,
        default="data_stream.txt",
        help="Path to save txt file.",
    )
    parser.add_argument(
        "--bad_case_path",
        type=str,
        default="prompt/generator.txt",
        help="Path to the save bad case.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If true, save the result to json file.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1,
        help="Sample size for the few-shot prompt.",
    )
    parser.add_argument(
        "--gen_max_token",
        type=int,
        default=800,
        help="Maximum token length for the generator prompt.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="result.json",
        help="Path for the output data file.",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default="default_key",
        help="OpenAI api key.",
    )
    parser.add_argument(
        "--openai_url",
        type=str,
        default="https://api.openai.com/v1/engines/davinci/completions",
        help="OpenAI url.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="text-davinci-003",
        help="OpenAI url.",
    )
    return parser.parse_args()


def main(args):
    # load source code
    ds = load_source_data(path=args.source_data_path)

    # load generator and discriminator prompts
    gen_prompt = load_prompt(args.gen_prompt_path)
    dis_prompt = load_prompt(args.dis_prompt_path)

    # generate few-shot task
    generations = few_shot_task_gen(
        source_code=ds,
        gen_prompt=gen_prompt,
        dis_prompt=dis_prompt,
        good_case_path=args.good_case_path,
        bad_case_path=args.bad_case_path,
        sample_number=args.sample_size,
        engine=args.openai_model,
        api_key=args.openai_key,
        base_url=args.openai_url,
        gen_max_token=args.gen_max_token,
        data_stream_path=args.data_stream_path,
    )
    if args.save_json:
        with open(args.output_data_path, "w") as f:
            json.dump(generations, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
