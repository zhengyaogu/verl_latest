# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import math
import json
import os
import uuid
import itertools
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, List

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, pad_2d_list_to_length
from verl.utils.tracking import ValidationGenerationsLogger
import verl.utils.torch_functional as verl_F


def merge_meta_info(meta_info1, meta_info2):
    if "timing" not in meta_info1 and "timing" not in meta_info2: # both are None
        return None
    if "timing" not in meta_info1 or "timing" not in meta_info2: # one of them is None
        meta_info = meta_info1["timing"] if "timing" in meta_info1 else meta_info2["timing"]
        meta_info2["timing"] = meta_info
        meta_info1["timing"] = meta_info
        return meta_info
    timing_info1 = meta_info1["timing"]
    timing_info2 = meta_info2["timing"]
    for key in timing_info2.keys():
        if key in timing_info1:
            timing_info1[key] += timing_info2[key]
        else:
            timing_info1[key] = timing_info2[key]
    meta_info1["timing"] = timing_info1
    meta_info2["timing"] = timing_info1
    return timing_info1

@dataclass
class DISCDataPoint:
    # we opt to not include the prompt in the data point, since it is already prepared in batch
    steps: List[str]
    z_score: float
    alpha: float
    num_sampled: int

def split_list(lst, alpha):
    cutoff = int(len(lst) * alpha)
    return lst[:cutoff], lst[cutoff:]

def compute_z_score(
    batch,
    epsilon=1e-6
):
    index = batch.non_tensor_batch["uid"]
    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    print("TOKEN LEVEL SCORES: ", batch.batch["token_level_scores"])
    if "rm_scores" in batch.batch:
        token_level_scores = batch.batch["token_level_scores"].sum(dim=-1)
        print("NUM VALID RM SCORES BY ROW: ", (batch.batch["rm_scores"]!=0.).sum(dim=-1))
        rm_scores = torch.sigmoid(batch.batch["rm_scores"].sum(dim=-1)).to(token_level_scores.dtype)
        correct_mask = token_level_scores == 1.
        rm_scores[correct_mask] = 1.
    elif "token_level_scores" in batch.batch:
        rm_scores = batch.batch["token_level_scores"].sum(dim=-1)
    else:
        raise ValueError("No reward scores found in batch")
    
    scores = rm_scores

    id2score = defaultdict(list)
    id2reward = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2reward_std = {}
    uid2batch_idx = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            if rewards[i] != 0. and rewards[i] != 1.:
                raise ValueError(f"Reward is not 0 or 1: {rewards[i]}")
            id2reward[index[i]].append(rewards[i])
            uid2batch_idx[index[i]].append(i) # write down index of data point for each uid
        
        max_len = 0
        for idx in id2score:
            if len(id2score[idx]) > max_len:
                max_len = len(id2score[idx])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                if max_len == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                    id2reward_std[idx] = torch.tensor(1.0)
                else:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.tensor(0.0)
                    id2reward_std[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
                id2reward_std[idx] = torch.std(torch.tensor([id2reward[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for idx in id2score:
            max_i = max(range(len(id2score[idx])), key=lambda i: id2score[idx][i])
            id2score[idx] = (id2score[idx][max_i] - id2mean[idx]) / (id2std[idx] + epsilon)
            uid2batch_idx[idx] = uid2batch_idx[idx][max_i]


    return id2score, uid2batch_idx, id2mean, id2std, id2reward_std

def stochastic_topk(estimates, error_magnitude, k):
    """
    Single-run stochastic top-k selection
    
    Args:
        estimates: 1D tensor of estimates
        error_magnitude: scalar (uniform error for all items)
        k: number of items to select
    
    Returns:
        indices: top-k indices from this random realization
        perturbed_values: the perturbed values for selected items
    """
    n = len(estimates)
    
    # Sample one random error for each item from uniform distribution
    # Error range: [-error_magnitude, +error_magnitude]
    errors = torch.randn(n) * math.sqrt(error_magnitude) * math.sqrt(math.pi / 2)
    
    # Perturb estimates with sampled errors
    perturbed_estimates = estimates + errors
    
    # Select top-k from perturbed estimates
    values, indices = torch.topk(perturbed_estimates, k)
    
    return indices, values

def osmd_sampler(estimates, k, alpha=0.5, tau=1.0, base_probs=None):
    """
    Args:
        estimates: 1D tensor of estimates
        alpha: base probability
        tau: temperature parameter
    
    Returns:
        indices: top-k indices from this random realization
        perturbed_values: the perturbed values for selected items
    """
    n = len(estimates)
    if base_probs is None:
        base_probs = torch.ones_like(estimates) / n

    sample_probs = torch.exp(estimates / tau)
    sample_probs = base_probs * sample_probs

    sorted_indices = torch.argsort(sample_probs)
    prob_ranking = torch.argsort(sorted_indices)

    ranking_discount = 1 - alpha * prob_ranking / n
    discounted_probs = sample_probs * ranking_discount

    sorted_discounted_probs = discounted_probs[sorted_indices]
    sorted_sample_probs = sample_probs[sorted_indices]

    v = sorted_discounted_probs #
    u = sorted_sample_probs.flip(0).cumsum(0).flip(0) * alpha / n
    non_zero_idx = torch.nonzero(u < v, as_tuple=True)[0]
    i_star = non_zero_idx[0]

    above_threshold_probs = sorted_discounted_probs[i_star:]
    above_threshold_probs = (1 - alpha * i_star / n) * above_threshold_probs / above_threshold_probs.sum()
    
    final_probs = torch.zeros_like(discounted_probs)
    final_probs[i_star:] = above_threshold_probs
    final_probs[:i_star] = alpha / n
    final_probs = final_probs[prob_ranking] # restore the original order

    sampled_idx = torch.multinomial(final_probs, num_samples=k, replacement=False)

    return sampled_idx, final_probs, i_star


def num_correct_answers_by_group(batch: DataProto, rollout_n: int):
    """
    Check if a prompt has correct answers
    """

    id2_num_correct = dict()
    uids = np.unique(batch.non_tensor_batch["uid"]).tolist()
    id2_num_correct = {uid: 0 for uid in uids}

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
        if rewards[i] == 1.:
            id2_num_correct[uid] += 1
    
    id2_adv_level = dict()
    level_center = rollout_n // 2
    for uid, num_correct in id2_num_correct.items():
        level = level_center - abs(num_correct - level_center)
        id2_adv_level[uid] = level

    return id2_num_correct, id2_adv_level

def compute_weights(id2_adv_level: dict, adv_level_proportions: dict, alpha: float = 0.5):
    n = len(id2_adv_level)
    adv_level_proportions_curr = Counter([abs(int(v)) for v in id2_adv_level.values()]) # force integer levels,
    # this should not modify the original id2level dict if it contains adv levels
    # force integer levels for num correct levels
    for level in adv_level_proportions.keys():
        adv_level_proportions[level] = (
            alpha * adv_level_proportions[level] + (1 - alpha) * adv_level_proportions_curr[level] / n
        )
    
    num_levels = len(adv_level_proportions.keys())
    weights = {level: min(5., 1 / (num_levels * adv_level_proportions[level] + 1e-6)) for level in adv_level_proportions.keys()}
    return weights, adv_level_proportions

    

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # if self.config.trainer.sec.enable:
        #     from verl.utils.auto_curriculum.sampler import BanditSampler
        #     train_sampler = BanditSampler(
        #         self.train_dataset,
        #         batch_size=self.config.data.train_batch_size,
        #         shuffle=self.config.data.shuffle,
        #         drop_last=True,
        #         collate_fn=collate_fn,
        #         seed=42,
        #         max_steps=self.config.trainer.total_training_steps,
        #         **self.config.trainer.sec.bandit
        #     )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        replay_buffer = None
        ema_vf_loss_mean, ema_vf_loss_var = None, None
        difficulty_heatmap = []
        # difficulty_df = pd.DataFrame(columns=["raw_prompt", "difficulty", "step"])
        
        # # Get project_name and exp_name from config
        # project_name = getattr(self.config, "project_name", "default_project")
        # exp_name = getattr(self.config, "exp_name", "default_exp")

        # # Create the directory path under "df"
        # save_dir = os.path.join("df", project_name, exp_name)
        # os.makedirs(save_dir, exist_ok=True)

        # maintain a global index for observed levels, used to train the second head
        global_level_index = defaultdict(list)
        # maintain a global index for observed perf_diff values, used when target == "perf_diff"
        global_perf_diff_index = defaultdict(list)

        difficulty2avg_adv = defaultdict(int)

        if self.config.adv_predictor.get("log_level_every_problem", False):
            prompt2levels = defaultdict(list)
        
        prompt2times_sampled = defaultdict(int)

        if self.config.greso.get("enable", False):
            zero_var_streak_index = defaultdict(list)


        # record the proportion of each abs adv level for critic training
        adv_level_proportions = dict()
        level_delta_proportions = dict()

        num_levels = self.config.actor_rollout_ref.rollout.n // 2 + 1
        for i in range(num_levels):
            adv_level_proportions[i] = 1 / num_levels
        
        num_levels = self.config.actor_rollout_ref.rollout.n
        for i in range(num_levels):
            level_delta_proportions[i] = 1 / num_levels

        adv_predictor_batch = None

        for epoch in range(self.config.trainer.total_epochs):
            _dataloader_iter = iter(self.train_dataloader)
            for batch_dict in _dataloader_iter:
                metrics = {}
                timing_raw = {}
                
                with marked_timer("step_total", timing_raw):
                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )

                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    # add uid to batch
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    id2index = dict()
                    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                        id2index[uid] = batch.non_tensor_batch["index"][i]

                    # advantage predictor runs before rollout
                    with marked_timer("curator_total", timing_raw):
                        with marked_timer("curator_adv_predictor", timing_raw):
                            if self.config.adv_predictor.enable:
                                prev_adv_predictor_batch = adv_predictor_batch if adv_predictor_batch is not None else None
                                adv_predictor_batch = self.critic_wg.compute_values(batch)
                                adv_predictor_batch = adv_predictor_batch.union(batch)

                                logger.log_hist(
                                    data={
                                        "adv_predictor/pre-selection/values": adv_predictor_batch.batch["values"],
                                    },
                                    step=self.global_steps
                                )

                                if self.config.adv_predictor.train_critic_only:
                                    print("ONLY UPDATING CRITIC")
                                
                                if self.global_steps <= self.config.adv_predictor.dormant_steps + self.config.adv_predictor.critic_warmup: # randomly sample prompts during critic warmup
                                    sampled_idx = torch.randperm(len(adv_predictor_batch))[:self.config.adv_predictor.num_samples]
                                    sample_weights = torch.ones_like(adv_predictor_batch.batch["values"].squeeze(-1)) / len(adv_predictor_batch) # uniform weights
                                else:
                                    if not self.config.adv_predictor.train_critic_only:
                                        if self.config.adv_predictor.sampler == "softmax":
                                            if self.config.adv_predictor.get("temperature_annealing", False):
                                                t0 = self.config.adv_predictor.temperature
                                                t1 = self.config.adv_predictor.max_temperature
                                                tau = t0 + (t1 - t0) * self.global_steps / self.total_training_steps
                                            else:
                                                tau = self.config.adv_predictor.temperature
                                            
                                            sample_weights = torch.nn.functional.gumbel_softmax(
                                                adv_predictor_batch.batch["values"].squeeze(-1),
                                                tau=tau,
                                                dim=0
                                            )
                                            sampled_idx = torch.topk(
                                                sample_weights, 
                                                self.config.adv_predictor.num_samples, 
                                                dim=0,
                                                sorted=False
                                            ).indices.squeeze(-1)
                                            sampled_idx = sampled_idx[torch.randperm(sampled_idx.shape[0])]
                                        elif self.config.adv_predictor.sampler == "stochastic_topk":
                                            if self.config.adv_predictor.get("temperature_annealing", False):
                                                t0 = self.config.adv_predictor.temperature
                                                t1 = self.config.adv_predictor.max_temperature
                                                tau = t0 + (t1 - t0) * self.global_steps / self.total_training_steps
                                            else:
                                                tau = self.config.adv_predictor.temperature
                                            sample_weights = 1-torch.abs(adv_predictor_batch.batch["values"].squeeze(-1) - 0.5)
                                            sampled_idx = torch.topk(
                                                sample_weights,
                                                self.config.adv_predictor.num_samples, 
                                                dim=0,
                                                sorted=False
                                            ).indices.squeeze(-1)
                                            sampled_idx = sampled_idx[torch.randperm(sampled_idx.shape[0])]
                                        elif self.config.adv_predictor.sampler == "osmd":
                                            if self.config.adv_predictor.get("alpha_annealing", False):
                                                a0 = self.config.adv_predictor.alpha
                                                a1 = self.config.adv_predictor.final_alpha
                                                alpha = a0 + (a1 - a0) * self.global_steps / self.total_training_steps
                                            else:
                                                alpha = self.config.adv_predictor.alpha
                                            
                                            tau = self.config.adv_predictor.temperature

                                            sampled_idx, _, i_star = osmd_sampler(
                                                adv_predictor_batch.batch["values"].squeeze(-1),
                                                self.config.adv_predictor.num_samples,
                                                alpha=alpha,
                                                tau=tau
                                            )
                                            logger.log(
                                                data={
                                                    "osmd/i_star": i_star
                                                },
                                                step=self.global_steps
                                            )
                                        elif self.config.adv_predictor.sampler == "uniform":
                                            n = int(self.config.data.train_batch_size)
                                            k = int(self.config.adv_predictor.num_samples)
                                            tau = self.config.adv_predictor.temperature

                                            sample_weights = torch.nn.functional.softmax(
                                                adv_predictor_batch.batch["values"].squeeze(-1) / tau,
                                                dim=0
                                            )
                                            sorted_idx = torch.argsort(sample_weights)
                                            sorted_sample_weights = sample_weights[sorted_idx]
                                            top_weights = sorted_sample_weights.flip(0).cumsum(0).flip(0)

                                            if self.config.adv_predictor.get("top_p_annealing", False):
                                                p0 = self.config.adv_predictor.top_p
                                                p1 = self.config.adv_predictor.final_top_p
                                                top_p = p0 + (p1 - p0) * self.global_steps / self.total_training_steps
                                            else:
                                                top_p = self.config.adv_predictor.top_p
                                            
                                            sampling_threshold = torch.nonzero(top_weights <= top_p, as_tuple=True)[0]
                                            if sampling_threshold.shape[0] == 0:
                                                threshold_i = 0
                                            else:
                                                threshold_i = sampling_threshold[0]
                                            num_above_threshold = min(n - threshold_i, k)
                                            num_uniform = k - num_above_threshold
                                            
                                            idx_above_threshold = torch.multinomial(sample_weights, num_samples=num_above_threshold, replacement=False)
                                            rem_idx = torch.from_numpy(np.setdiff1d(np.arange(n), idx_above_threshold.numpy()))

                                            if num_uniform > 0:
                                                uniform_idx = torch.randperm(rem_idx.shape[0])[:num_uniform]
                                                idx_below_threshold = rem_idx[uniform_idx]
                                                sampled_idx = torch.cat([idx_above_threshold, idx_below_threshold])
                                            else:
                                                sampled_idx = idx_above_threshold
                                        elif self.config.adv_predictor.sampler == "metropolis":
                                            if prev_adv_predictor_batch is None:
                                                sampled_idx = torch.arange(len(adv_predictor_batch))
                                            else:
                                                if self.config.adv_predictor.get("temperature_annealing", False):
                                                    t0 = self.config.adv_predictor.temperature
                                                    t1 = self.config.adv_predictor.max_temperature
                                                    tau = t0 + (t1 - t0) * self.global_steps / self.total_training_steps
                                                else:
                                                    tau = self.config.adv_predictor.temperature
                                                
                                                old_adv = prev_adv_predictor_batch.batch["advantages"]
                                                rand_idx = torch.randperm(old_adv.shape[0])
                                                old_adv = old_adv[rand_idx]
                                                old_idx = torch.arange(old_adv.shape[0])[rand_idx]
                                                new_adv = adv_predictor_batch.batch["values"].squeeze(-1).detach()
                                                acceptance_probs = torch.min(
                                                    torch.exp(
                                                        1 / tau * (new_adv - old_adv)
                                                    ),
                                                    torch.ones_like(new_adv)
                                                )
                                                print("old_adv.shape: ", old_adv.shape)
                                                print("new_adv.shape: ", new_adv.shape)
                                                print("acceptance_probs.shape: ", acceptance_probs.shape)
                                                sample_mask = torch.rand_like(acceptance_probs) < acceptance_probs

                                                turnover_rate = (sample_mask.sum() / sample_mask.shape[0]).item()
                                                logger.log(
                                                    data={
                                                        "adv_predictor/metropolis/turnover_rate": turnover_rate
                                                    },
                                                    step=self.global_steps
                                                )
                                                print("sample_mask.shape: ", sample_mask.shape)
                                                print("old_idx.shape: ", old_idx.shape)
                                                
                                                selected_old_idx = old_idx[~sample_mask]
                                                sampled_idx = torch.arange(len(new_adv))[sample_mask]
                                                prev_adv_predictor_batch = prev_adv_predictor_batch.select_idxs(selected_old_idx)
                                                print("prev_adv_predictor_batch.batch.keys(): ", prev_adv_predictor_batch.batch.keys())
                                    
                                # curr_heatmap = np.zeros_like(adv_predictor_batch.non_tensor_batch["uid"], dtype=np.float32)
                                # heatmap_sort_idx = np.argsort(adv_predictor_batch.batch["difficulty"])

                                batch = batch.select_idxs(sampled_idx)
                                adv_predictor_batch = adv_predictor_batch.select_idxs(sampled_idx)
                                print(batch.non_tensor_batch["uid"] == adv_predictor_batch.non_tensor_batch["uid"])
                                print((batch.non_tensor_batch["uid"] == adv_predictor_batch.non_tensor_batch["uid"]).sum())
                                assert np.all(batch.non_tensor_batch["uid"] == adv_predictor_batch.non_tensor_batch["uid"]), "batch uid and adv_predictor_batch uid should be the same, got {} and {} instead".format(batch.non_tensor_batch["uid"], adv_predictor_batch.non_tensor_batch["uid"])

                                if self.config.adv_predictor.sampler != "metropolis":
                                    # here sample_weights should be sliced like the batch and adv_predictor_batch
                                    adv_predictor_batch.batch["sampled_probs"] = sample_weights[sampled_idx]

                                    if self.config.adv_predictor.get("temperature_annealing", False):
                                        t0 = self.config.adv_predictor.temperature
                                        t1 = self.config.adv_predictor.max_temperature
                                        tau = t0 + (t1 - t0) * self.global_steps / self.total_training_steps
                                    else:
                                        tau = self.config.adv_predictor.temperature

                                    adv_predictor_batch.batch["sampled_logits"] = adv_predictor_batch.batch["values"].squeeze(-1) / tau
                                    
                                for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                    prompt2times_sampled[id2index[uid]] += 1
                                
                                logger.log_hist(
                                    data={
                                        "adv_predictor/pred_abs_adv": adv_predictor_batch.batch["values"].squeeze(-1)
                                    },
                                    step=self.global_steps
                                )
                                logger.log_hist(
                                    data={
                                        "adv_predictor/sampled_probs": adv_predictor_batch.batch["sampled_probs"]
                                    },
                                    step=self.global_steps
                                )
                                if self.config.critic.model.get("style", "value_head") == "ordinal":
                                    logger.log_hist(
                                        data={
                                            "adv_predictor/levels1": adv_predictor_batch.batch["levels1"],
                                            "adv_predictor/levels2": adv_predictor_batch.batch["levels2"],
                                        },
                                        step=self.global_steps
                                    )
                                elif self.config.critic.model.get("style", "value_head") == "value_head":
                                    logger.log_hist(
                                        data={
                                            "adv_predictor/probs1": adv_predictor_batch.batch["probs1"],
                                            "adv_predictor/probs2": adv_predictor_batch.batch["probs2"],
                                        },
                                        step=self.global_steps
                                    )
                                logger.log(
                                    data={
                                        "adv_predictor/pred_abs_adv/mean": torch.mean(adv_predictor_batch.batch["values"].squeeze(-1)).item(),
                                        "adv_predictor/pred_abs_adv/max": torch.max(adv_predictor_batch.batch["values"].squeeze(-1)).item(),
                                        "adv_predictor/pred_abs_adv/min": torch.min(adv_predictor_batch.batch["values"].squeeze(-1)).item()
                                    },
                                    step=self.global_steps
                                )
                                
                                # curr_heatmap[uid_mask] = 1
                                # curr_heatmap = curr_heatmap[heatmap_sort_idx]
                                # difficulty_heatmap.append(torch.from_numpy(curr_heatmap))
                                # print("HEATMAP SHAPE: ", torch.stack(difficulty_heatmap).shape)
                                # logger.log_heatmap(
                                #     data={
                                #         "adv_predictor/heatmap": torch.stack(difficulty_heatmap).unsqueeze(0)
                                #     },
                                #     step=self.global_steps
                                # )
                                
                                
                                logger.log_hist(
                                    data={
                                        "adv_predictor/difficulty": batch.batch["difficulty"]
                                    },
                                    step=self.global_steps
                                )
                                difficulty_dict = torch.bincount(batch.batch["difficulty"].int())[1:]
                                difficulty_dict = {"adv_predictor/difficulty/{}".format(diff_level+1): count.item() for diff_level, count in enumerate(difficulty_dict)}
                                logger.log(data=difficulty_dict, step=self.global_steps)

                                batch_keys = list(set(adv_predictor_batch.batch.keys()) - set(["values"]))
                                non_tensor_batch_keys = list(adv_predictor_batch.non_tensor_batch.keys())
                                adv_predictor_batch = adv_predictor_batch.pop(
                                    batch_keys=batch_keys, 
                                    non_tensor_batch_keys=non_tensor_batch_keys
                                )

                                if self.global_steps > self.config.adv_predictor.dormant_steps + self.config.adv_predictor.critic_warmup:
                                    if self.config.adv_predictor.sampler == "metropolis" and prev_adv_predictor_batch is not None:
                                        print("prev_adv_predictor_batch.batch.keys(): ", prev_adv_predictor_batch.batch.keys())
                                        print("prev_adv_predictor_batch.non_tensor_batch.keys(): ", prev_adv_predictor_batch.non_tensor_batch.keys())
                                        
                                        print("adv_predictor_batch.batch.keys(): ", adv_predictor_batch.batch.keys())
                                        print("adv_predictor_batch.non_tensor_batch.keys(): ", adv_predictor_batch.non_tensor_batch.keys())

                                        print("batch.batch.keys(): ", batch.batch.keys())
                                        print("batch.non_tensor_batch.keys(): ", batch.non_tensor_batch.keys())

                                        batch_keys = list(batch.batch.keys())
                                        non_tensor_batch_keys = list(batch.non_tensor_batch.keys())

                                        prev_adv_predictor_batch = prev_adv_predictor_batch.pop(
                                            batch_keys=batch_keys, 
                                            non_tensor_batch_keys=non_tensor_batch_keys
                                        )
                                        adv_predictor_batch = adv_predictor_batch.pop(
                                            batch_keys=batch_keys, 
                                            non_tensor_batch_keys=non_tensor_batch_keys
                                        )
                                        adv_predictor_batch = DataProto.concat([adv_predictor_batch, prev_adv_predictor_batch])

                                        batch = DataProto.concat([batch, prev_adv_predictor_batch])
                                        assert len(batch) == len(adv_predictor_batch), "batch size should be equal to adv_predictor_batch size, got {} instead".format(len(batch))

                                        new_uids = np.array(
                                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                                        )
                                        adv_predictor_batch.non_tensor_batch["uid"] = new_uids
                                        batch.non_tensor_batch["uid"] = new_uids.copy()
                            elif self.config.adv_predictor.get("use_difficulty_curriculum", False):
                                avg_adv_curr = []
                                for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                    avg_adv_curr.append(
                                        difficulty2avg_adv[
                                            int(batch.batch["difficulty"][i].item())
                                        ]
                                    )
                                avg_adv_curr = torch.tensor(avg_adv_curr)
                                tau = self.config.adv_predictor.temperature
                                sample_weights = torch.nn.functional.softmax(
                                    avg_adv_curr / tau,
                                    dim=0
                                )
                                num_samples = self.config.adv_predictor.train_batch_size
                                sampled_idx = torch.multinomial(sample_weights, num_samples=num_samples, replacement=False)
                                batch = batch.select_idxs(sampled_idx)

                    # inference: greedy sampling simply do rollouts in one go, disc sampling do rollouts in multiple iterations
                    sampling_method = self.config.actor_rollout_ref.rollout.get("sampling_method", "greedy")
                    if sampling_method == "greedy":
                        # assert len(batch) == 32, "batch size should be equal to 32, got {} instead".format(len(batch))
                        gen_batch = self._get_gen_batch(batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps # pass global_steps to trace
                        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # assert len(gen_batch) == 256, "gen_batch size should be equal to 256, got {} instead".format(len(gen_batch))
                    elif sampling_method == "disc":
                        # prepare variables for disc sampling
                        alpha = self.config.actor_rollout_ref.rollout.alpha0
                        unique_uids = np.unique(batch.non_tensor_batch["uid"])
                        id2data_points = {}
                        for uid in unique_uids:
                            id2data_points[uid] = DISCDataPoint(
                                steps=[],
                                z_score=float("inf"),
                                alpha=alpha,
                                num_sampled=0
                            )
                        remaining_uids = unique_uids.tolist()
                        batch_total = None

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if sampling_method == "greedy":
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                
                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)

                                # repeat to align with repeated responses in rollout
                                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                batch = batch.union(gen_batch_output)

                                if "response_mask" not in batch.batch.keys():
                                    batch.batch["response_mask"] = compute_response_mask(batch)
                                # Balance the number of valid tokens across DP ranks.
                                # NOTE: This usually changes the order of data in the `batch`,
                                # which won't affect the advantage calculation (since it's based on uid),
                                # but might affect the loss calculation (due to the change of mini-batching).
                                # TODO: Decouple the DP balancing and mini-batching.
                                if self.config.trainer.balance_batch:
                                    self._balance_batch(batch, metrics=metrics)

                                # compute global_valid tokens
                                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                                with marked_timer("reward", timing_raw, color="yellow"):
                                    # compute reward model score
                                    if self.use_rm:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                        batch = batch.union(reward_tensor)

                                    if self.config.reward_model.launch_reward_fn_async:
                                        future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                                    else:
                                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            elif sampling_method == "greso":
                                n_easy, n_hard, n_total = 0, 0, 0
                                b_r_default = self.config.greso.get("default_rollout_bsz", None) #init b_r with default rollout bsz
                                b_r = b_r_default
                                p_easy = self.config.greso.get("p_easy", None)
                                p_hard = self.config.greso.get("p_hard", None)
                                p_delta = self.config.greso.get("p_delta", None)
                                assert b_r is not None and p_easy is not None and p_hard is not None and p_delta is not None, "greso sampling method requires default_rollout_bsz, p_easy, p_hard and p_delta to be specified in config.greso"
                                batch_curr = None
                                n_added = 0

                                def greso_filter(batch, zero_var_streak_index, p_easy, p_hard, b_r):
                                    #global indices
                                    # can I do global indices not using uid but original indexes?
                                    global_indices = batch.non_tensor_batch["index"].tolist()

                                    # each zero_var_streak_index[idx] is a 2-tuple (streak_length, is easy), get a list ps which picks p_easy if the streak is easy and p_hard if the streak is hard, then sample a mask based on ps to select data points for the next rollout batch
                                    ps = []
                                    for idx in global_indices:
                                        if idx in zero_var_streak_index:
                                            streak_length, is_easy = zero_var_streak_index[idx]
                                            if is_easy:
                                                p = 1 - (p_easy ** streak_length)
                                            else:
                                                p = 1 - (p_hard ** streak_length)
                                                
                                        else:
                                            p = 1 # if the data point has never had zero variance output, we consider it easy and use p_easy
                                        ps.append(p)
                                    
                                    # filter batch dataset based on ps (ps is the filtering probability) until we have b_r data points
                                    ps = np.array(ps)
                                    if len(ps) <= b_r:
                                        return batch
                                    else:
                                        sampled_mask = np.random.rand(len(ps)) < ps
                                        while sampled_mask.sum() < b_r:
                                            sampled_mask = np.random.rand(len(ps)) < ps
                                        selected_indices = np.where(sampled_mask)[0][:b_r]
                                        batch_selected = batch.select_idxs(selected_indices)
                                        return batch_selected
                                
                                def update_streak_index(zero_var_streak_index, batch):
                                    id2score, uid2best_batch_idx, id2mean, id2std, id2reward_std = compute_z_score(batch)
                                    # use id2reward_std to update zero_var_streak_index, if reward std is 0, increase the streak length by 1, otherwise reset the streak length to 0, also update whether it's easy or hard based on whether the mean score is above a certain threshold
                                    for uid in id2score:
                                        idx = id2index[uid]
                                        if id2reward_std[uid] <= 1e-2: # consider it zero variance if reward std is less than 1e-2
                                            is_easy_new = id2mean[uid] > 1e-2 # consider it easy if mean score is less than 1e-2
                                            if idx in zero_var_streak_index:
                                                streak_length, is_easy = zero_var_streak_index[idx]
                                                if is_easy == is_easy_new:
                                                    zero_var_streak_index[idx] = (streak_length + 1, is_easy)
                                                else:
                                                    zero_var_streak_index[idx] = (1, is_easy_new) # reset streak length to 1, update easy/hard based on new mean score
                                            else:
                                                zero_var_streak_index[idx] = (1, is_easy_new) # init with streak length 1 and easy
                                    return zero_var_streak_index


                                while n_added < self.config.data.get("train_batch_size", None):
                                    batch = greso_filter(batch, zero_var_streak_index, p_easy, p_hard, b_r)
                                    gen_batch = self._get_gen_batch(batch)
                                    gen_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]
                                    n_added += len(gen_batch)
                                    if n_added > self.config.data.get("train_batch_size", None):
                                        gen_batch = gen_batch[:self.config.data.train_batch_size - (n_added - len(gen_batch))]
                                    gen_batch.meta_info["global_steps"] = self.global_steps # pass global_steps to trace
                                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                                    # generate rollout & compute reward
                                    with marked_timer("gen", timing_raw, color="red"):
                                        if not self.async_rollout_mode:
                                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                        else:
                                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                        
                                        timing_raw.update(gen_batch_output.meta_info["timing"])
                                        gen_batch_output.meta_info.pop("timing", None)

                                        # repeat to align with repeated responses in rollout
                                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                        batch = batch.union(gen_batch_output)

                                        if "response_mask" not in batch.batch.keys():
                                            batch.batch["response_mask"] = compute_response_mask(batch)
                                        # Balance the number of valid tokens across DP ranks.
                                        # NOTE: This usually changes the order of data in the `batch`,
                                        # which won't affect the advantage calculation (since it's based on uid),
                                        # but might affect the loss calculation (due to the change of mini-batching).
                                        # TODO: Decouple the DP balancing and mini-batching.
                                        if self.config.trainer.balance_batch:
                                            self._balance_batch(batch, metrics=metrics)

                                        # compute global_valid tokens
                                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                                        with marked_timer("reward", timing_raw, color="yellow"):
                                            # compute reward model score
                                            if self.use_rm:
                                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                                batch = batch.union(reward_tensor)

                                            if self.config.reward_model.launch_reward_fn_async:
                                                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                                            else:
                                                token_level_scores, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                                                batch.batch["token_level_scores"] = token_level_scores
                                        
                                        batch_curr = batch if batch_curr is None else DataProto.concat([batch_curr, batch])
                                        if len(batch_curr) >= self.config.data.train_batch_size:
                                            break
                                        
                                        #alpha is the current zero-variance example ratio in this iteration (as some rollouts have already occurred in this iteration)
                                        id2score, uid2best_batch_idx, id2mean, id2std, id2reward_std = compute_z_score(batch_curr)
                                        #compute the ratio of uids with near 0 reward_std from id2reward_std
                                        alpha = len([v for v in id2reward_std.values() if v < 1e-2]) / len(id2reward_std)
                                        n_easy = len([k for k in id2reward_std.keys() if id2reward_std[k] < 1e-2 and id2mean[k] > 1e-2])
                                        n_hard = len([k for k in id2reward_std.keys() if id2reward_std[k] < 1e-2 and id2mean[k] <= 1e-2])
                                        n_total = len(id2reward_std)
                                        if n_easy / n_total > 1/12:
                                            p_easy -= p_delta
                                        else:
                                            p_easy += p_delta
                                        if n_hard / n_total > 1/6:
                                            p_hard -= p_delta
                                        else:
                                            p_hard += p_delta
                                        b_r = min(
                                            b_r_default,
                                            1.25 * (self.config.data.get("train_batch_size", None) - n_added) / (1 - alpha)
                                        )
                                        # next batch
                                        batch_dict = next(_dataloader_iter)
                                        batch = DataProto.from_single_dict(batch_dict)

                                        # add uid to batch
                                        batch.non_tensor_batch["uid"] = np.array(
                                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                                        )
                                        
                                id2index = dict()
                                for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                    id2index[uid] = batch.non_tensor_batch["index"][i]
                                zero_var_streak_index = update_streak_index(zero_var_streak_index, batch_curr)

                                batch = batch_curr

                            elif sampling_method == "disc":
                                if self.async_rollout_mode:
                                    self.async_rollout_manager.wake_up() # manually wake up the rollout manager

                                round_num = 0
                                total_traj_remaining = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                max_num_rounds = self.config.actor_rollout_ref.rollout.max_num_rounds
                                rollout_size_per_round = self.config.actor_rollout_ref.rollout.disc_rollout_size_per_round
                                finished_uids = set()

                                while round_num < max_num_rounds and total_traj_remaining > 0:
                                    print(f"NUM UIDS STILL TO SAMPLE THIS BATCH: {len(unique_uids) - len(finished_uids)}")
                                    remaining_uids = np.array([uid for uid in unique_uids if uid not in finished_uids])
                                    if remaining_uids.shape[0] == 0: # if all uids are finished, sample all uids to make the batch size
                                        remaining_uids = unique_uids.copy()
                                    actual_rollout_size_curr_round, rollout_remainder = total_traj_remaining // remaining_uids.shape[0], total_traj_remaining % remaining_uids.shape[0]

                                    if round_num < max_num_rounds - 1 and actual_rollout_size_curr_round >= rollout_size_per_round: 
                                        select_mask = np.isin(batch.non_tensor_batch["uid"], remaining_uids)
                                        batch_curr_round = batch.select_idxs(select_mask)
                                        batch_curr_round = batch_curr_round.repeat(
                                            repeat_times=rollout_size_per_round,
                                            interleave=True,
                                        )
                                    else: # sample evenly what's remaining, sample remainders randomly
                                        remainder_uids = np.random.choice(remaining_uids, size=rollout_remainder, replace=False)
                                        main_uids = np.array([uid for uid in remaining_uids if uid not in remainder_uids])
                                        remainder_mask = np.isin(batch.non_tensor_batch["uid"], remainder_uids)
                                        main_mask = np.isin(batch.non_tensor_batch["uid"], main_uids)
                                        main_batch = batch.select_idxs(main_mask)
                                        remainder_batch = batch.select_idxs(remainder_mask)
                                        remainder_batch = remainder_batch.repeat(
                                            repeat_times=actual_rollout_size_curr_round + 1,
                                            interleave=True,
                                        )
                                        main_batch = main_batch.repeat(
                                            repeat_times=actual_rollout_size_curr_round,
                                            interleave=True,
                                        )
                                        batch_curr_round = DataProto.concat([main_batch, remainder_batch])
                                    
                                    #prepare input_ids for rollout (add partial solution & re-pad left-padded prompt ids)
                                    prompt_ids = []
                                    raw_prompts = []
                                    left_padded_prompt_ids = batch_curr_round.batch["input_ids"]
                                    prompt_attention_mask = batch_curr_round.batch["attention_mask"]
                                    prompt_lengths = prompt_attention_mask.sum(dim=-1)

                                    for i, uid in enumerate(batch_curr_round.non_tensor_batch["uid"]):
                                        if len(id2data_points[uid].steps) > 0:
                                            split_last_step, _ = split_list(id2data_points[uid].steps[-1], id2data_points[uid].alpha)
                                            partial_solution = list(itertools.chain.from_iterable(id2data_points[uid].steps[:-1]))
                                            partial_solution = partial_solution + split_last_step
                                            new_prompt_ids = left_padded_prompt_ids[i, -prompt_lengths[i]:].tolist() + partial_solution
                                            prompt_ids.append(new_prompt_ids)
                                            raw_prompts.append([self.tokenizer.decode(new_prompt_ids, skip_special_tokens=True)])
                                        else:
                                            new_prompt_ids = left_padded_prompt_ids[i, -prompt_lengths[i]:].tolist()
                                            prompt_ids.append(new_prompt_ids)
                                            raw_prompts.append([self.tokenizer.decode(new_prompt_ids, skip_special_tokens=True)])

                                    
                                    prompt_ids = pad_2d_list_to_length(prompt_ids, pad_token_id=self.tokenizer.pad_token_id, max_length=self.config.data.max_prompt_length, left_pad=True)
                                    batch_curr_round.non_tensor_batch["raw_prompt"] = np.array(raw_prompts)
                                    batch_curr_round.batch["input_ids"] = prompt_ids
                                    gen_batch_curr_round = self._get_gen_batch(batch_curr_round)
                                    if not self.async_rollout_mode:
                                        gen_batch_output_curr_round = self.actor_rollout_wg.generate_sequences(gen_batch_curr_round)
                                    else:
                                        gen_batch_output_curr_round = self.async_rollout_manager.generate_sequences(gen_batch_curr_round)
                                    
                                    # is_different = False
                                    # for key in gen_batch_output_curr_round.meta_info.keys():
                                    #     if key in batch_curr_round.meta_info:
                                    #         if gen_batch_output_curr_round.meta_info[key] != batch_curr_round.meta_info[key]:
                                    #             is_different = True
                                    #             print(f"meta info {key} is not the same")
                                    #             print(gen_batch_output_curr_round.meta_info[key])
                                    #             print(batch_curr_round.meta_info[key])
                                    merge_meta_info(batch_curr_round.meta_info, gen_batch_output_curr_round.meta_info) # this operation is inplace, second argument is modified
                                    batch_curr_round = batch_curr_round.union(gen_batch_output_curr_round)

                                    if "response_mask" not in batch_curr_round.batch.keys():
                                        batch_curr_round.batch["response_mask"] = compute_response_mask(batch_curr_round)

                                    if self.config.trainer.balance_batch:
                                        self._balance_batch(batch_curr_round, metrics=metrics)
                                    
                                    with marked_timer("reward", timing_raw, color="yellow"):
                                        # compute reward model score
                                        if self.use_rm:
                                            reward_tensor = self.rm_wg.compute_rm_score(batch_curr_round)

                                        if self.config.reward_model.launch_reward_fn_async:
                                            future_reward = compute_reward_async.remote(data=batch_curr_round, reward_fn=self.reward_fn)

                                        # we union the reward_tensor with the batch_curr_round later to compute the token level scores
                                        token_level_scores, reward_extra_infos_dict = compute_reward(batch_curr_round, self.reward_fn)
                                        batch_curr_round = batch_curr_round.union(reward_tensor)
                                        batch_curr_round.batch["token_level_scores"] = token_level_scores
                                    
                                    with marked_timer("adv", timing_raw, color="brown"):
                                        # we combine with rule-based rm
                                        reward_extra_infos_dict: dict[str, list]
                                        if self.config.reward_model.launch_reward_fn_async:
                                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                            batch_curr_round.batch["rm_scores"] = reward_tensor

                                        if reward_extra_infos_dict:
                                            batch_curr_round.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})   

                                    for uid in batch_curr_round.non_tensor_batch["uid"]:
                                        id2data_points[uid].num_sampled += 1
                                    
                                    id2score, uid2best_batch_idx, id2mean, id2std, id2reward_std = compute_z_score(batch_curr_round, epsilon=1e-6)

                                    token_level_scores = batch_curr_round.batch["token_level_scores"].sum(dim=-1)
                                    for i, uid in enumerate(batch_curr_round.non_tensor_batch["uid"]):
                                        # if id2reward_std[uid] <= 1e-2:
                                        #     finished_uids.add(uid)
                                        if token_level_scores[i] >= 1:
                                            finished_uids.add(uid)
                                    print("NUM UIDS FINISHED: ", len(finished_uids))
                                    
                                    for uid in id2score:
                                        print(f"ROUND: {round_num}, UID: {uid}, SCORE: {id2score[uid]}, MEAN: {id2mean[uid]}, STD: {id2std[uid]}, REWARD STD: {id2reward_std[uid]}")
                                    
                                    unique_uids_curr_round = np.unique(batch_curr_round.non_tensor_batch["uid"])
                                    for uid in unique_uids_curr_round:
                                        new_z_score = id2score[uid]
                                        # if new z score is better, or if the last step is too short, update the data point
                                        if new_z_score < id2data_points[uid].z_score or (
                                            len(id2data_points[uid].steps) > 0 and len(id2data_points[uid].steps[-1]) <= 1
                                        ):
                                            id2data_points[uid].z_score = new_z_score
                                            response_ids = batch_curr_round.batch["responses"]
                                            response_lengths = batch_curr_round.batch["response_attention_mask"].sum(dim=-1)
                                            unpadded_response = response_ids[uid2best_batch_idx[uid], :response_lengths[i]].tolist()
                                            if len(id2data_points[uid].steps) > 0:
                                                second_last_step, _ = split_list(id2data_points[uid].steps[-1], id2data_points[uid].alpha)
                                                id2data_points[uid].steps[-1] = second_last_step
                                                id2data_points[uid].steps.append(unpadded_response)
                                            else:
                                                id2data_points[uid].steps.append(unpadded_response)
                                            id2data_points[uid].alpha = self.config.actor_rollout_ref.rollout.alpha0
                                        else:
                                            id2data_points[uid].alpha = id2data_points[uid].alpha * self.config.actor_rollout_ref.rollout.alpha0
                                    
                                    if batch_total is None:
                                        batch_total = batch_curr_round
                                    else:
                                        batch_total = DataProto.concat([batch_total, batch_curr_round])
                                    for uid in unique_uids:
                                        print(f"UID: {uid}, NUM SAMPLED: {id2data_points[uid].num_sampled}")
                                    
                                    round_num += 1
                                    total_traj_remaining -= batch_curr_round.non_tensor_batch["uid"].shape[0]
                
                                if self.async_rollout_mode:
                                    self.async_rollout_manager.sleep()

                                batch = batch_total
                                # compute global_valid tokens
                                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output

                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                        
                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        with marked_timer("curator_total", timing_raw):
                            with marked_timer("compute_values", timing_raw):
                                if self.use_critic and not self.config.adv_predictor.enable:
                                    with marked_timer("values", timing_raw, color="cyan"):
                                        values = self.critic_wg.compute_values(batch)
                                        batch = batch.union(values)
                        
                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            if not (self.config.actor_rollout_ref.rollout.get("sampling_method", "greedy") == "disc" or
                                    self.config.greso.get("enable", False)):
                                reward_extra_infos_dict: dict[str, list]
                                if self.config.reward_model.launch_reward_fn_async:
                                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                batch.batch["token_level_scores"] = reward_tensor

                                if reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            if self.use_rm and self.config.reward_model.add_rm_score_to_adv:
                                token_level_scores = batch.batch["token_level_scores"]
                                rm_scores = batch.batch["rm_scores"].clone().to(dtype=token_level_scores.dtype)
                                correct_mask = (token_level_scores.sum(dim=-1) == 1)
                                rm_scores[correct_mask] = token_level_scores[correct_mask]
                                batch.batch["token_level_rewards"] = rm_scores.to(dtype=token_level_scores.dtype)

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            print("batch.batch['advantages'].shape: ", batch.batch["advantages"].shape)
                            
                        def compute_abs_adv_by_group(batch: DataProto, cliprange_lo=None, cliprange_hi=None):
                            adv = verl_F.masked_mean(batch.batch["advantages"].abs(), batch.batch["response_mask"], axis=-1)
                            if "perf_diff_unit" in batch.batch.keys():
                                adv = verl_F.masked_mean(batch.batch["perf_diff_unit"], batch.batch["response_mask"], axis=-1)
                                cliprange_lo = float('-inf') if cliprange_lo is None else cliprange_lo
                                cliprange_hi = float('inf') if cliprange_hi is None else cliprange_hi
                                # adv = torch.clamp(adv, min=cliprange_lo, max=cliprange_hi)
                            id2adv = defaultdict(list)
                            for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                id2adv[uid].append(adv[i])
                            id2adv_new = dict()
                            for uid, advs in id2adv.items():
                                avg_adv = np.mean(advs).item()
                                # assert avg_adv <= 1, "avg_adv should be less than 1., got {} instead".format(avg_adv)
                                id2adv_new[uid] = avg_adv
                            return id2adv_new

                        print("batch.non_tensor_batch.keys(): ", batch.non_tensor_batch.keys())
                        id2adv = compute_abs_adv_by_group(
                            batch
                        )

                        id2_num_correct, id2_adv_level = num_correct_answers_by_group(batch, self.config.actor_rollout_ref.rollout.n)
                        num_effective = sum([(v > 0 and v < self.config.actor_rollout_ref.rollout.n) for v in id2_num_correct.values()])

                        if self.config.adv_predictor.get("use_difficulty_curriculum", False):
                            id2difficulty = dict()
                            for i, uid in enumerate(batch.non_tensor_batch["uid"]):
                                id2difficulty[uid] = batch.batch["difficulty"][i].item()
                            
                            avg_adv_by_difficulty_curr = defaultdict(list)
                            for uid, difficulty in id2difficulty.items():
                                avg_adv_by_difficulty_curr[difficulty].append(id2adv[uid])
                            
                            avg_adv_by_difficulty_curr = {
                                difficulty: sum(advs) / len(advs) if len(advs) > 0 else 0 
                                for difficulty, advs in avg_adv_by_difficulty_curr.items()
                            }
                            
                            for k, v in difficulty2avg_adv.items():
                                old_v = v
                                alpha = self.config.adv_predictor.ema_coeff
                                difficulty2avg_adv[k] = (1 - alpha) * old_v + alpha * avg_adv_by_difficulty_curr[k]

                        print("NUM EFFECTIVE: ", num_effective)
                        print("id2_adv_level: ", id2_adv_level)

                        # update global_level_index


                        logger.log(
                            data={
                                "adv_predictor/num_effective": num_effective
                            },
                            step=self.global_steps
                        )

                        level2_weights, adv_level_proportions = compute_weights(id2_adv_level, adv_level_proportions)
                        if not self.config.adv_predictor.get("use_level_weights", True):
                            level2_weights = {level: 1. for level in level2_weights.keys()}

                        level_weights_logging_dict = {
                            "adv_predictor/level_weights/{}".format(level): level2_weights[level] for level in level2_weights.keys()
                        }
                        adv_level_proportions_logging_dict = {
                            "adv_predictor/level_proportions/{}".format(level): adv_level_proportions[level] for level in adv_level_proportions.keys()
                        }
                        level_info_logging_dict = {
                            **level_weights_logging_dict,
                            **adv_level_proportions_logging_dict
                        }
                        logger.log(
                            data=level_info_logging_dict,
                            step=self.global_steps
                        )

                        logger.log_hist(
                            data={
                                "adv_predictor/abs_adv": torch.tensor(list(id2adv.values()))
                            },
                            step=self.global_steps
                        )

                        correct_rate = sum([v > 0 for v in id2_num_correct.values()]) / len(id2_num_correct)

                        # add the level history for each prompt
                        if self.config.adv_predictor.get("log_level_every_problem", False):
                            unique_uids = np.unique(batch.non_tensor_batch["uid"]).tolist()
                            for uid in unique_uids:
                                level = id2_num_correct[uid]
                                prompt2levels[id2index[uid]].append(level)

                        if self.config.adv_predictor.enable:
                            avg_adv = []
                            num_correct = []
                            adv_level = []
                            target_probs = []
                            
                            id2_level_delta = dict()
                            
                            history_length = self.config.adv_predictor.get("history_length", 1)
                            target_probs_delta = []
                            target_levels_delta = []
                            adv_level_by_difficulty_curr = defaultdict(list)
                            for i, uid in enumerate(adv_predictor_batch.non_tensor_batch["uid"]):
                                # append to global_level_index
                                level = id2_num_correct[uid]
                                global_level_index[id2index[uid]].append(level)
                                if len(global_level_index[id2index[uid]]) > history_length:
                                    global_level_index[id2index[uid]].pop(0)

                                # data for first head
                                avg_adv.append(id2adv[uid])
                                adv_level_by_difficulty_curr[int(batch.batch["difficulty"][i].item())].append(id2adv[uid])
                                level_window_avg = sum(global_level_index[id2index[uid]]) / len(global_level_index[id2index[uid]])
                                num_correct.append(level_window_avg)
                                target_probs.append(level_window_avg / self.config.actor_rollout_ref.rollout.n)
                                adv_level.append(id2_adv_level[uid])
                                
                                # data for second head
                                if len(global_level_index[id2index[uid]]) > 1:
                                    level_delta = (global_level_index[id2index[uid]][-1] - global_level_index[id2index[uid]][0]) / (history_length - 1)
                                    target_levels_delta.append(level_delta)
                                    target_probs_delta.append(level_delta / self.config.actor_rollout_ref.rollout.n)

                                    id2_level_delta[uid] = level_delta
                                    
                                else:
                                    target_levels_delta.append(0)
                                    target_probs_delta.append(0)
                                    id2_level_delta[uid] = 0

                            
                            level_delta_weights, level_delta_proportions = compute_weights(id2_level_delta, level_delta_proportions)
                            print("level_delta_weights: ", level_delta_weights)
                            print("level_delta_proportions: ", level_delta_proportions)

                            level_delta_weights_logging_dict = {
                                "adv_predictor/level_delta_weights/{}".format(level): level_delta_weights[level] for level in level_delta_weights.keys()
                            }
                            level_delta_proportions_logging_dict = {
                                "adv_predictor/level_delta_proportions/{}".format(level): level_delta_proportions[level] for level in level_delta_proportions.keys()
                            }
                            level_delta_info_logging_dict = {
                                **level_delta_weights_logging_dict,
                                **level_delta_proportions_logging_dict
                            }

                            logger.log(
                                data=level_delta_info_logging_dict,
                                step=self.global_steps
                            )

                            logger.log_hist(
                                data={
                                    "adv_predictor/target_levels_delta": torch.tensor(target_levels_delta),
                                    "adv_predictor/target_levels": torch.tensor(num_correct),
                                },
                                step=self.global_steps
                            )
                            
                            logger.log(
                                data={
                                    "adv_predictor/target_level": torch.tensor(num_correct).float().mean().item(),
                                },
                                step=self.global_steps
                            )

                            
                            if self.config.trainer.critic_warmup <= self.global_steps and (not self.config.adv_predictor.enable or not self.config.adv_predictor.train_critic_only):
                                # update actor
                                with marked_timer("update_actor", timing_raw, color="red"):
                                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                            if self.config.adv_predictor.target == "perf_diff":
                                with marked_timer("curator_total", timing_raw):
                                    with marked_timer("compute_new_log_prob", timing_raw, color="pink"):
                                        new_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                        new_log_prob.batch.pop("entropys")
                                        new_log_prob = DataProto.from_dict(tensors={"new_log_probs": new_log_prob.batch["old_log_probs"]})
                                        batch = batch.union(new_log_prob)
                                        
                                        response_mask = batch.batch["response_mask"].to(bool)
                                        new_log_prob = batch.batch["new_log_probs"]
                                        advantages = batch.batch["advantages"]
                                        old_log_prob = batch.batch["old_log_probs"]
                                        advantages = batch.batch["advantages"]

                                        log_importance_ratio = new_log_prob - old_log_prob
                                        # compute sequence-level importance ratio:
                                        # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
                                        # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
                                        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
                                        log_importance_ratio = torch.sum(log_importance_ratio * response_mask, dim=-1) / seq_lengths
                                        importance_ratio = torch.exp(log_importance_ratio)
                                        #target_importance_ratio_cliprange = self.config.adv_predictor.get("importance_ratio_cliprange", 10.0)
                                        #importance_ratio = torch.clamp(importance_ratio, -target_importance_ratio_cliprange, target_importance_ratio_cliprange)

                                        perf_diff_unit = advantages * importance_ratio.unsqueeze(-1)
                                        batch.union(DataProto.from_single_dict({
                                            "perf_diff_unit": perf_diff_unit.detach().clone()
                                        }))
                                        
                                        perf_diff_unit_cliprange_lo = self.config.adv_predictor.get("perf_diff_unit_cliprange_lo", None)
                                        perf_diff_unit_cliprange_hi = self.config.adv_predictor.get("perf_diff_unit_cliprange_hi", None)
                                        id2perf_diff = compute_abs_adv_by_group(
                                            batch, 
                                            cliprange_lo=perf_diff_unit_cliprange_lo, 
                                            cliprange_hi=perf_diff_unit_cliprange_hi
                                        )
                                        perf_diff = []
                                        perf_diff_window_avg = []
                                        for i, uid in enumerate(adv_predictor_batch.non_tensor_batch["uid"]):
                                            perf_diff.append(id2perf_diff[uid])
                                            global_perf_diff_index[id2index[uid]].append(id2perf_diff[uid])
                                            if len(global_perf_diff_index[id2index[uid]]) > history_length:
                                                global_perf_diff_index[id2index[uid]].pop(0)
                                            perf_diff_window_avg.append(sum(global_perf_diff_index[id2index[uid]]) / len(global_perf_diff_index[id2index[uid]]))
                                        perf_diff = torch.tensor(perf_diff)
                                        perf_diff_window_avg = torch.tensor(perf_diff_window_avg)

                                        if self.config.adv_predictor.get("use_window_avg_target", False):
                                            perf_diff = perf_diff_window_avg
                                        
                                        if self.config.adv_predictor.get("use_sampling_prior", False):
                                            # inv_sampling_prior = []
                                            # for i, uid in enumerate(adv_predictor_batch.non_tensor_batch["uid"]):
                                            #     inv_sampling_prior.append(prompt2times_sampled[id2index[uid]])
                                            # k = len(inv_sampling_prior)
                                            # inv_sampling_prior = torch.tensor(inv_sampling_prior)
                                            # inv_sampling_prior = inv_sampling_prior / inv_sampling_prior.sum() * k
                                            # inv_sampling_prior = 1 / inv_sampling_prior
                                            # inv_sampling_prior = torch.clamp(inv_sampling_prior, 0.1, 10)
                                            # sampled_probs = torch.clamp(adv_predictor_batch.batch["sampled_probs"], 1e-6, 1)
                                            #inv_sampling_prior = 1 / adv_predictor_batch.batch["sampled_probs"]
                                            inv_sampling_prior = 1.
                                            perf_diff = perf_diff * inv_sampling_prior
                                
                                perf_diff_amplifier = self.config.adv_predictor.get("perf_diff_amplifier", 1.0)
                                perf_diff = perf_diff * perf_diff_amplifier

                                
                                logger.log_hist(
                                    # log importance ratio distribution
                                    data={
                                        "adv_predictor/importance_ratio": importance_ratio
                                    },
                                    step=self.global_steps
                                )
                                logger.log_hist(
                                    data={
                                        "adv_predictor/perf_diff": perf_diff
                                    },
                                    step=self.global_steps
                                )

                                log_perf_diff_unit = verl_F.masked_mean(batch.batch["perf_diff_unit"], batch.batch["response_mask"], axis=-1)
                                perf_diff_unit_cliprange_lo = self.config.adv_predictor.get("perf_diff_unit_cliprange_lo", float('-inf'))
                                perf_diff_unit_cliprange_hi = self.config.adv_predictor.get("perf_diff_unit_cliprange_hi", float('inf'))
                                log_perf_diff_unit = torch.clamp(log_perf_diff_unit, perf_diff_unit_cliprange_lo, perf_diff_unit_cliprange_hi)
                                logger.log_hist(
                                    data={
                                        "adv_predictor/perf_diff_unit": log_perf_diff_unit
                                    },
                                    step=self.global_steps
                                )

                            
                            critic_infos = DataProto.from_single_dict({
                                "advantages": torch.tensor(avg_adv),
                                "target_probs": torch.tensor(target_probs),
                                "target_levels": torch.tensor(num_correct, dtype=torch.int32),
                                "adv_level": torch.tensor(adv_level, dtype=torch.int32),
                                "target_levels_delta": torch.tensor(target_levels_delta),
                                "target_probs_delta": torch.tensor(target_probs_delta),
                            })
                            if self.config.adv_predictor.get("target", "abs_adv") == "perf_diff":
                                critic_infos = critic_infos.union(DataProto.from_single_dict({
                                    "perf_diff": perf_diff,
                                }))
                            adv_predictor_batch = adv_predictor_batch.union(critic_infos)
                            print("adv_predictor_batch.batch.keys(): ", adv_predictor_batch.batch.keys())

                            adv_predictor_batch_size = int(self.config.adv_predictor.train_batch_size)
                            assert adv_predictor_batch_size >= len(adv_predictor_batch), "self.config.adv_predictor.train_batch_size should be greater than or equal to len(adv_predictor_batch)={}, got {} instead".format(len(adv_predictor_batch), adv_predictor_batch_size)
                            old_examples_size = adv_predictor_batch_size - len(adv_predictor_batch)

                            fix_critic = self.config.adv_predictor.get("fix_critic", False)
                            if not fix_critic and (
                                self.config.adv_predictor.dormant_steps <= self.global_steps and 
                                replay_buffer is not None and len(replay_buffer) >= old_examples_size
                            ):
                                with marked_timer("curator_total", timing_raw):
                                    with marked_timer("update_curator", timing_raw, color="pink"):
                                        replay_idx = torch.randperm(len(replay_buffer))[:old_examples_size]
                                        replay_batch = replay_buffer.select_idxs(replay_idx)
                                        replay_batch = DataProto.concat([replay_batch, adv_predictor_batch])
                                        adv_level = replay_batch.batch["adv_level"].tolist()
                                        replay_batch.batch["weights"] = torch.tensor([level2_weights[level] for level in adv_level]).to(
                                            dtype=replay_batch.batch["advantages"].dtype
                                        )
                                        replay_batch.batch["level_delta_weights"] = torch.tensor([
                                            level_delta_weights[abs(int(level_delta))] for level_delta in target_levels_delta],
                                            dtype=replay_batch.batch["advantages"].dtype
                                        )
                                        print("replay_batch.batch['level_delta_weights'].shape: ", replay_batch.batch["level_delta_weights"].shape)
                                        
                                        critic_output = self.critic_wg.update_critic(replay_batch)
                                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                        
                                # if self.config.adv_predictor.sampler == "stochastic_topk":
                                #     print("EMA VF LOSS MEAN updated")
                                #     ema_coeff = self.config.adv_predictor.ema_coeff
                                #     ema_vf_loss_mean = (
                                #         critic_output_metrics["critic/vf_loss"] 
                                #         if ema_vf_loss_mean is None
                                #         else ema_vf_loss_mean * ema_coeff + critic_output_metrics["critic/vf_loss"] * (1 - ema_coeff)
                                #     )
                                #     # ema_vf_loss_var = (
                                #     #     critic_output_metrics["critic/vf_loss_var"] 
                                #     #     if ema_vf_loss_var is None 
                                #     #     else ema_vf_loss_var * ema_coeff + critic_output_metrics["critic/vf_loss_var"] * (1 - ema_coeff)
                                #     # )

                                metrics.update(critic_output_metrics)


                            if replay_buffer is None:
                                replay_buffer = deepcopy(adv_predictor_batch)
                            else:
                                replay_buffer = DataProto.concat([replay_buffer, deepcopy(adv_predictor_batch)])
                                max_buffer_size = self.config.adv_predictor.replay_buffer_size
                                if len(replay_buffer) > max_buffer_size:
                                    replay_buffer = replay_buffer[-max_buffer_size:]
                        
                        else:
                            # update critic
                            if self.use_critic and not self.config.adv_predictor.enable:
                                with marked_timer("update_critic", timing_raw, color="pink"):
                                    critic_output = self.critic_wg.update_critic(batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                metrics.update(critic_output_metrics)
                            
                                                    # implement critic warmup
                            if self.config.trainer.critic_warmup <= self.global_steps and (not self.config.adv_predictor.enable or not self.config.adv_predictor.train_critic_only):
                                # update actor
                                with marked_timer("update_actor", timing_raw, color="red"):
                                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                sample_gts = [
                                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                    for item in batch
                                ]

                                if "request_id" in batch.non_tensor_batch:
                                    reward_extra_infos_dict.setdefault(
                                        "request_id",
                                        batch.non_tensor_batch["request_id"].tolist(),
                                    )

                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    gts=sample_gts,
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic and not self.config.adv_predictor.enable))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
            
            
            # # plot the level history for each prompt
            # if self.config.adv_predictor.get("log_level_every_problem", False):
            #     print("saving level history to tensorboard... epoch: ", epoch)
            #     project_name = getattr(self.config, "project_name", "default_project")
            #     exp_name = getattr(self.config, "exp_name", "default_exp")
            #     save_dir = os.path.join("df", project_name, exp_name)
            #     os.makedirs(save_dir, exist_ok=True)
            #     with open(os.path.join(save_dir, "prompt2levels.json"), "w") as f:
            #         json.dump(prompt2levels, f)
                
            #     def create_batches(data, batch_size):
            #         return [data[i:min(i + batch_size, len(data))] for i in range(0, len(data), batch_size)]

            #     prompt_ids = list(prompt2levels.keys())
            #     prompt_ids_mini_batches = create_batches(prompt_ids, 50)
            #     for i, prompt_ids_mini_batch in enumerate(prompt_ids_mini_batches):
            #         fig, ax = plt.subplots(figsize=(12, 8))
            #         for prompt_id in prompt_ids_mini_batch:
            #             levels = prompt2levels[prompt_id]
            #             assert len(levels) == epoch + 1, "len(levels) should be equal to epoch + 1, got {} instead".format(len(levels))
            #             print("levels: ", levels)
            #             ax.plot(levels, label=prompt_id, alpha=0.6)

            #             ax.set_xlabel('Step')
            #             ax.set_ylabel('Level')

            #             # Add legend (you may want to adjust this since 100 entries is a lot)
            #             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            #             # Adjust layout to prevent legend cutoff
            #             plt.tight_layout()

            #         logger.log_figure(
            #             data={
            #                 "adv_predictor/level_history/{}".format(i): fig,
            #             },
            #             step=self.global_steps
            #         )

            #         plt.close(fig)
            # logger.flush()
