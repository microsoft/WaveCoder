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
import random
import copy
import os


class CaseSample:
    def __init__(
        self,
        seed_path: str,
        pre_prompt: str,
        sample_number: int = 1,
        seed: int = 1024,
        prefix_name: str = "",
    ):
        """
        Initialize the CaseSample instance.

        Parameters:
        - seed_path (str): The path to the seed data directory.
        - pre_prompt (str): The prompt to be used before the few-shot examples.
        - sample_number (int): The number of samples to be taken from the seed data.
        - seed (int): The seed for random number generation.
        - prefix_name (str): The prefix name for the seed data files.
        """
        self.prompt = pre_prompt
        self.sample_number = sample_number
        self.seed_path = seed_path
        self.prefix_name = prefix_name
        # init sample
        self.init_sample(seed_path)
        # random.seed(seed)

    def init_sample(self, path: str):
        """
        Initialize the sample list from the given path.

        Parameters:
        - path (str): The path to the directory containing seed data files.
        """
        if path is None:
            raise ValueError("Can't init path to obtain seed case")
        # with open(path, 'r') as f:
        #     sample_list = json.load(f)
        # f.close()
        sample_list = []
        file_list = os.listdir(path)

        for file_path in file_list:
            tmp_path = f"{path}/{file_path}"
            with open(tmp_path, "r") as f:
                data = f.readlines()
            f.close()
            data = "".join(data)
            sample_list.append(data)

        self.sample_list = self.preprocess_seed_data(sample_list)

    def preprocess_seed_data(self, seed_data: list) -> list:
        """
        Preprocess the seed data.

        Parameters:
        - seed_data (list): The list of seed data to be preprocessed.

        Returns:
        - list: The preprocessed seed data.
        """
        return seed_data

    def fewshot_sample(self) -> list:
        """
        Sample a few examples from the seed data.

        Returns:
        - list: A list of sampled few-shot examples.
        """
        assert self.sample_number <= len(self.sample_list)
        fewshot_case = random.sample(self.sample_list, self.sample_number)

        return fewshot_case

    def preprocess_add_data(self, data: str) -> str:
        """
        Preprocess data before adding it to the sample list.

        Parameters:
        - data (str): The data to be preprocessed.

        Returns:
        - str: The preprocessed data.
        """
        return data

    def add_case(self, data: str):
        """
        Add a new case to the sample list.

        Parameters:
        - data (str): The data to be added as a new case.
        """
        with open(
            f"{self.seed_path}/{self.prefix_name}_{len(self.sample_list)}.txt",
            "w",
        ) as f:
            f.write(data)
        f.close()
        self.sample_list.append(self.preprocess_add_data(data))

    def generate_fewshot_text(self) -> str:
        """
        Generate the few-shot text using the sampled cases and the prompt.

        Returns:
        - str: The generated few-shot text.
        """
        fewshot_case = self.fewshot_sample()
        gen_texts = self.prompt
        for i in range(len(fewshot_case)):
            gen_texts += f"Case {i}:\n{fewshot_case[i]}\n"

        return gen_texts


class GoodCase(CaseSample):

    def __init__(
        self, seed_path, pre_prompt, sample_number=1, seed=1024, prefix_name="good_case"
    ):
        super().__init__(
            seed_path,
            pre_prompt,
            sample_number=sample_number,
            seed=seed,
            prefix_name=prefix_name,
        )

    def generate_fewshot_text(self):
        fewshot_case = self.fewshot_sample()
        gen_texts = self.prompt
        for i in range(len(fewshot_case)):
            gen_texts += f"Good case {i}:\n{fewshot_case[i]}\n"

        return gen_texts

    def generate_fewshot_for_d(self):
        fewshot_case = self.fewshot_sample()
        gen_texts = ""
        for i in range(len(fewshot_case)):
            gen_texts += f"Good Case {i}:\n{fewshot_case[i]}\n"

        return gen_texts


class BadCase(CaseSample):
    def __init__(
        self, seed_path, pre_prompt, sample_number=1, seed=1024, prefix_name="bad_case"
    ):
        super().__init__(
            seed_path,
            pre_prompt,
            sample_number=sample_number,
            seed=seed,
            prefix_name=prefix_name,
        )

    def preprocess_add_data(self, data):
        return super().preprocess_add_data(data)

    def generate_fewshot_text(self):
        fewshot_case = self.fewshot_sample()
        gen_texts = self.prompt
        for i in range(len(fewshot_case)):
            gen_texts += f"Bad case {i}:\n{fewshot_case[i]}\n"

        return gen_texts

    def fewshot_sample(self):
        assert self.sample_number <= len(self.sample_list)
        fewshot_case = random.sample(self.sample_list[-1], self.sample_number)
        return fewshot_case

    def generate_fewshot_for_d(self):
        fewshot_case = self.fewshot_sample()
        gen_texts = ""
        # analysis = "Analysis:\nanswer:yes\nreasons:all requirements are satisfied"
        for i in range(len(fewshot_case)):
            gen_texts += f"Bad Case {i}:\n{fewshot_case[i]}\n"

        return gen_texts
