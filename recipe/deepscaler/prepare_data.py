from datasets import load_dataset, concatenate_datasets
from os.path import join

VERL_HOME = "/workspace/mnt/verl_latest"

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs



def prepare_math_data():
    train_dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    aime_2025 = concatenate_datasets([
        load_dataset("opencompass/AIME2025", "AIME2025-I",split="test"),
        load_dataset("opencompass/AIME2025", "AIME2025-II",split="test"),
    ])
    aime_2025 = aime_2025.rename_column("question", "problem")
    test_dataset = concatenate_datasets([test_dataset, aime_2025])

    instruction_following = "Let's think step by step and output the final answer within **\\boxed{}**."

    def process_fn(example, idx):
        question = example.pop("problem")

        question = question + " " + instruction_following

        answer = example.pop("solution")
        solution = example["answer"]
        data = {
            "data_source": "deepscaler",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
        return data

    split = "train"
    train_dataset = train_dataset.map(process_fn, with_indices=True)
    split = "test"
    test_dataset = test_dataset.map(process_fn, with_indices=True)

    train_dataset.to_parquet(join(VERL_HOME, "data/deepscaler_math.parquet"))
    test_dataset.to_parquet(join(VERL_HOME, "data/aime2425.parquet"))

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(len(train_dataset))
    print(len(test_dataset))