from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score.arc import get_arc_compute_score
from verl.utils.reward_score.countdown import get_countdown_compute_score
from verl.utils.reward_score.zebra import get_zebra_compute_score
from verl.utils.reward_score.deepscaler import get_deepscaler_reward_fn
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

def _select_rm_score_fn(data_source, correct_reward, format_reward):

    if data_source.startswith('test'):
        return get_deepscaler_reward_fn(correct_reward=1.0, format_reward=0.0)
    elif data_source.startswith('countdown'):
        if 'train' in data_source:
            return get_countdown_compute_score(correct_score=correct_reward, format_score=format_reward)
        elif 'test' in data_source:
            return get_countdown_compute_score(correct_score=1.0, format_score=0.0)
        else:
            raise ValueError(f'Invalid data source: {data_source}')
    elif data_source.startswith('zebra'):
        if 'train' in data_source:
            return get_zebra_compute_score(correct_score=correct_reward, format_score=format_reward)
        elif 'test' in data_source:
            return get_zebra_compute_score(correct_score=1.0, format_score=0.0)
        else:
            raise ValueError(f'Invalid data source: {data_source}')
    elif data_source.startswith('arc'):
        if 'train' in data_source:
            return get_arc_compute_score(correct_score=correct_reward, format_score=format_reward)
        elif 'test' in data_source:
            return get_arc_compute_score(correct_score=1.0, format_score=0.0)
        else:
            raise ValueError(f'Invalid data source: {data_source}')
    elif data_source == 'math_train':      
        return get_deepscaler_reward_fn(correct_reward=1.0, format_reward=format_reward)
    else:
        raise ValueError(f'Invalid data source: {data_source}')

def default_compute_score(data_source, solution_str, ground_truth, extra_info):
    return _select_rm_score_fn(data_source, 1.0, 0.0)(solution_str, ground_truth)


@register("sec")
class SECRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
