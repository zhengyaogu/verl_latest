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
The main entry point to run the PPO algorithm
"""

import logging
import os

import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo import core_algos
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_nccl_backend,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.engine import EngineRegistry
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def average_abs_advantage(levels: torch.Tensor=None, n: int=None, probs: torch.Tensor = None) -> float:
    if probs is None:
        assert levels is not None and n is not None
        p = levels / n
    else:
        p = probs
    return 2 * torch.sqrt(p * (1 - p))


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config
        self.engine = EngineRegistry.new(self.config.strategy, self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model()

    def _post_fn_values(self, micro_batch, preds):
        response_length = micro_batch["responses"].size(-1)
        values = preds[:, -response_length - 1 : -1]

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        if not use_remove_padding:
            values = values.squeeze(-1)
        values = F.softplus(values) # make sure the values are positive

        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.infer_batch(data, post_fn=self._post_fn_values)
            response_mask = data.batch["response_mask"]
            values = output["values"] * response_mask  # Only action tokens have values
            output = DataProto.from_dict(tensors={"values": values})

            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output

    def loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        old_values = batch["values"]
        returns = batch["returns"]
        response_mask = batch["response_mask"]
        micro_batch_metrics = {}

        values, _ = self._post_fn_values(batch, vpreds)

        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=values,
            values=old_values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=self.config.cliprange_value,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        if self.config.use_dynamic_bsz:
            # relative to the dynamic bsz
            loss = vf_loss * (len(batch) / self.config.ppo_mini_batch_size)
        else:
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            loss = vf_loss / gradient_accumulation

        micro_batch_metrics = {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(values, response_mask).detach().item(),
        }

        return loss, micro_batch_metrics

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                select_keys = [
                    "input_ids",
                    "responses",
                    "response_mask",
                    "attention_mask",
                    "position_ids",
                    "values",
                    "returns",
                ]
                batch = data.select(batch_keys=select_keys).batch
                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                if has_multi_modal_inputs:
                    num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
                    non_tensor_select_keys = ["multi_modal_inputs"]
                    dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
                else:
                    dataloader = batch.split(self.config.ppo_mini_batch_size)

                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(dataloader):
                        self.engine.optimizer_zero_grad()
                        mini_batch_metrics = self.engine.train_batch(mini_batch, self.loss_fn)
                        grad_norm = self.engine.optimizer_step()
                        mini_batch_metrics["critic/grad_norm"] = grad_norm.detach().item()
                        append_to_dict(metrics, mini_batch_metrics)
                self.engine.optimizer_zero_grad()
            delta_time = timer.last

            # TODO: should not access engine's flops_counter
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.engine.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            metrics["critic/lr"] = self.engine.lr_scheduler_step()[0]
            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.engine.unshard_data(data=output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class AdvPredictorWorker(CriticWorker):

    def value_loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor],
        second_head: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adv = batch["advantages"].unsqueeze(-1) # this corresponds to old_values
        target_probs = batch["target_probs"].unsqueeze(-1)
        if second_head:
            target_probs = batch["target_probs_delta"].unsqueeze(-1)
        response_mask = torch.ones_like(adv)
        micro_batch_metrics = {}

        probs, _ = self._post_fn_values(batch, vpreds)
        print("inside value_loss_fn")
        print("probs.shape: ", probs.shape)
        print("target_probs.shape: ", target_probs.shape)
        print("target_probs_delta: ", batch["target_probs_delta"].unsqueeze(-1).shape)
        clamped_probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
        values = average_abs_advantage(probs=clamped_probs)

        loss_config = self.config.get("loss", None)
        if loss_config is None:
            tau = 0.5
        else:
            tau = loss_config.get("tau", 0.5)
        
        weights = batch.get("weights", None) if not second_head else batch.get("level_delta_weights", None)

        if second_head:
            print("content of weights: ", weights)
        
        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=probs,
            values=probs,
            returns=target_probs, # returns are all zeros
            response_mask=response_mask, # response_mask are all ones
            cliprange_value=1e6, # no clipping
            loss_agg_mode=self.config.loss_agg_mode, # this does not matter as loss is computed on only one token
            weights=weights,
        )

        with torch.no_grad():
            skewness = (values - adv).mean()
        
        if self.config.use_dynamic_bsz:
            # relative to the dynamic bsz
            loss = vf_loss * (len(batch) / self.config.ppo_mini_batch_size)
        else:
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            loss = vf_loss / gradient_accumulation

        if second_head:
            micro_batch_metrics = {
                "critic/vf_loss2": vf_loss.detach().item(),
                "critic/vf_clipfrac2": vf_clipfrac.detach().item(),
                "critic/vpred_mean2": masked_mean(values, response_mask).detach().item(),
                #"critic/vf_loss_var": vf_loss_var.detach().item(),
                "critic/skewness2": skewness.detach().item(),
            }
        else:
            micro_batch_metrics = {
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": masked_mean(values, response_mask).detach().item(),
                #"critic/vf_loss_var": vf_loss_var.detach().item(),
                "critic/skewness": skewness.detach().item(),
            }

        return loss, micro_batch_metrics
    
    def ordinal_loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor],
        second_head: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        vpreds: (batch_size, num_levels) - predicted values for each level
        """
        logits, _ = self._post_fn_ordinal(batch, vpreds)
        levels = (logits > 0.5).sum(dim=-1)
        values = average_abs_advantage(levels, logits.shape[-1])

        target_levels = batch["target_levels"]
        if second_head:
            target_levels = batch["target_levels_delta"]
        adv = batch["advantages"].unsqueeze(-1)
        weights = batch.get("weights", None) if not second_head else batch.get("level_delta_weights", None)

        if second_head:
            print("content of weights: ", weights)

        print("logits.shape: ", logits.shape)
        print("target_levels.shape: ", target_levels.shape)

        loss = core_algos.compute_ordinal_loss(
            logits=logits,
            target_levels=target_levels,
            loss_agg_mode=self.config.loss_agg_mode,
            weights=weights,
        )

        with torch.no_grad():
            vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                vpreds=values,
                values=values,
                returns=adv,
                response_mask=torch.ones_like(adv),
                cliprange_value=1e6,
                loss_agg_mode=self.config.loss_agg_mode,
                weights=weights,
            )

            skewness = (values - adv).mean()

        vpred_mean = masked_mean(values, torch.ones_like(values)).detach().item()

        if self.config.use_dynamic_bsz:
            # relative to the dynamic bsz
            loss = loss * (len(batch) / self.config.ppo_mini_batch_size)
        else:
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            loss = loss / gradient_accumulation

        if second_head:
            micro_batch_metrics = {
                "critic/ordinal_loss2": loss.detach().item(),
                "critic/vpred_mean2": vpred_mean,
                "critic/vf_loss2": vf_loss.detach().item(),
                "critic/skewness2": skewness.detach().item(),
            }
        else:
            micro_batch_metrics = {
                "critic/ordinal_loss": loss.detach().item(),
                "critic/vpred_mean": vpred_mean,
                "critic/vf_loss": vf_loss.detach().item(),
                "critic/skewness": skewness.detach().item(),
            }
        
        print("micro_batch_metrics.keys(): ", micro_batch_metrics.keys())

        return loss, micro_batch_metrics
    
    def osmd_loss_fn(
        self, 
        batch: DataProto, vpreds: dict[str, torch.Tensor],
        second_head: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probs, _ = self._post_fn_osmd(batch, vpreds)
        sampled_probs = batch["sampled_probs"].unsqueeze(-1)
        print("inside osmd_loss_fn", probs.shape, sampled_probs.shape)
        ratio = probs / sampled_probs
        if "perf_diff" in batch.keys():
            advantages = batch["perf_diff"].unsqueeze(-1)
        else:
            advantages = batch["adv_level"].unsqueeze(-1)
        policy_losses1 = -advantages * ratio
        clip_range = self.config.get("clip_range", 0.5)
        print("clip_range: ", clip_range)
        policy_losses2 = -advantages * torch.clamp(
            ratio, 1 - clip_range, 1 + clip_range
        )
        clip_pg_losses = torch.maximum(
            policy_losses1, policy_losses2
        )

        response_mask = torch.ones_like(clip_pg_losses)
        from verl.trainer.ppo.core_algos import agg_loss
        loss = agg_loss(
            loss_mat=clip_pg_losses, 
            loss_mask=response_mask, 
            loss_agg_mode=self.config.loss_agg_mode, 
            weights=None
        )

        micro_batch_metrics = {
            "critic/osmd_loss": loss.detach().item(),
        }
        return loss, micro_batch_metrics
    
    def _post_fn_values(self, micro_batch, preds):
        probs = preds[:, -1].unsqueeze(-1)
        probs = F.softplus(probs) # make sure the values are positive
        return probs, {"probs": probs.clone().detach()}
    
    def _post_fn_ordinal(self, micro_batch, preds):
        logits = preds[:, -1].unsqueeze(1)
        
        # Method 1: Use torch.cat to avoid in-place operation
        logits = torch.cat([
            logits[:, :, :1],  # Keep first column as-is
            F.softplus(logits[:, :, 1:])  # Apply softplus to rest
        ], dim=-1)
        
        logits = - (torch.cumsum(logits, dim=-1))
        probs = F.sigmoid(logits)
        return logits, {"probs": probs.clone().detach()}
    
    def _post_fn_osmd_infer(self, micro_batch, preds):
        logits = preds[:, -1].unsqueeze(-1)
        return logits, {"logits": logits.clone().detach()}

    
    def _post_fn_osmd(self, micro_batch, preds):
        preds = preds[:, -1].unsqueeze(-1)
        probs = F.log_softmax(preds, dim=0)
        return probs, {"probs": probs.clone().detach()}

    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)

            if self.config.model.get("style", "value_head") == "ordinal":
                post_fn = self._post_fn_ordinal
            elif self.config.model.get("style", "value_head") == "value_head":
                post_fn = self._post_fn_values
            elif self.config.model.get("style", "value_head") == "osmd":
                post_fn = self._post_fn_osmd_infer
            else:
                raise ValueError(f"Invalid style: {self.config.model.get('style', 'value_head')}")
            output = self.engine.infer_batch(data, post_fn=post_fn)

            if self.config.model.get("style", "value_head") == "ordinal":
                probs = torch.clamp(output["probs"], min=1e-6, max=1-1e-6)
                levels = (probs > 0.5).sum(dim=-1)
                levels1 = levels
                if "probs/2" in output:
                    probs2 = torch.clamp(output["probs/2"], min=1e-6, max=1-1e-6)
                    levels2 = (probs2 > 0.5).sum(dim=-1)
                    levels = torch.clamp(levels + levels2, min=0)
                values = average_abs_advantage(levels=levels, n=probs.shape[-1])
                output = DataProto.from_dict(tensors={
                    "values": values,
                    "probs": probs,
                    "levels": levels,
                    "levels1": levels1,
                    "levels2": levels2,
                })
            elif self.config.model.get("style", "value_head") == "value_head":
                probs = output["probs"]
                probs1 = probs
                probs2 = output["probs/2"]
                if "probs/2" in output:
                    probs = output["probs"] + output["probs/2"]
                clamped_probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
                values = average_abs_advantage(probs=clamped_probs)
                output = DataProto.from_dict(tensors={
                    "probs": probs,
                    "values": values,
                    "probs1": probs1,
                    "probs2": probs2,
                })
            elif self.config.model.get("style", "value_head") == "osmd":
                logits = output["logits"]
                output = DataProto.from_dict(tensors={
                    "probs": logits,
                    "values": logits,
                    "logits": logits,
                })

            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                select_keys = [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "advantages",
                    "target_levels",
                    "weights",
                    "level_delta_weights",
                    "target_probs",
                    "target_probs_delta",
                    "target_levels_delta",
                    "sampled_probs",
                    "adv_level",
                ]
                batch = data.select(batch_keys=select_keys).batch
                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                if has_multi_modal_inputs:
                    num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
                    non_tensor_select_keys = ["multi_modal_inputs"]
                    dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
                else:
                    dataloader = batch.split(self.config.ppo_mini_batch_size)

                if self.config.model.get("style", "value_head") == "ordinal":
                    loss_fn = self.ordinal_loss_fn
                elif self.config.model.get("style", "value_head") == "value_head":
                    loss_fn = self.value_loss_fn
                elif self.config.model.get("style", "value_head") == "osmd":
                    loss_fn = self.osmd_loss_fn
                else:
                    raise ValueError(f"Invalid style: {self.config.model.get('style', 'value_head')}")

                for epoch in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(dataloader):
                        self.engine.optimizer_zero_grad()
                        mini_batch_metrics = self.engine.train_batch(mini_batch, loss_fn)
                        grad_norm = self.engine.optimizer_step()
                        mini_batch_metrics["critic/grad_norm"] = grad_norm.detach().item()
                        append_to_dict(metrics, mini_batch_metrics)
                self.engine.optimizer_zero_grad()
            delta_time = timer.last

            # TODO: should not access engine's flops_counter
            # global_num_tokens = data.meta_info["global_token_num"]
            # estimated_flops, promised_flops = self.engine.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            metrics["critic/lr"] = self.engine.lr_scheduler_step()[0]
            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.engine.unshard_data(data=output)

        output = output.to("cpu")
        return output