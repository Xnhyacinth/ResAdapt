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
Single Process Actor
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
###
from visionthink.adaptive.utils import to_numpy_cpu
###

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)

        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        
        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    ###
    @GPUMemoryLogger(role="dp predictor", logger=logger)
    def scale_multi_modal(self, data: DataProto, eval_mode=False):
        """
        Batch multi-modal scaling pre-processing:
        Safely and efficiently generate updated `input_ids/attention_mask/position_ids`
        and corresponding `multi_modal_inputs/multi_modal_data` for subsequent policy updates.
        """
        import numpy as np

        self.actor_module.eval()

        video2list = data.meta_info.pop("video2list", False)
        video2image = data.meta_info.pop("video2image", False)
        return_mm_data = data.meta_info.pop("return_mm_data", True)
        # Use dynamic micro-batch size if available; fallback to 8
        micro_batch_size = data.meta_info.pop("micro_batch_size", 8) if hasattr(data, "meta_info") else 8
        micro_batches = data.split(micro_batch_size)

        # Backup original fields (safe fallback if optional keys are missing)
        original_data = {
            "ori_prompt": data.non_tensor_batch.get("raw_prompt", None).copy(),

        }

        processed_micro_batches = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    if "raw_prompt" in model_inputs:
                        messages = list(model_inputs["raw_prompt"])
                    else:
                        raise ValueError("raw_prompt is required for scaling multi-modal data.")
                    # elif "prompt" in model_inputs:
                    #     messages = list(model_inputs["prompt"])

                    scaled_inputs = self.actor_module(
                        messages=messages,
                        eval_mode=eval_mode,
                        video2list=video2list,
                        video2image=video2image,
                        return_mm_data=return_mm_data,
                    )

                scale_one_mask = model_inputs.get("scale_one_mask", None)
                if scale_one_mask is not None:
                    scale_one_mask_cpu = (
                        scale_one_mask.detach().cpu().numpy()
                        if isinstance(scale_one_mask, torch.Tensor)
                        else np.array(scale_one_mask, dtype=bool)
                    )
                    if scale_one_mask_cpu.any():
                        if return_mm_data:
                            if "multi_modal_data" in scaled_inputs and "multi_modal_data" in model_inputs:
                                original_mm = model_inputs["multi_modal_data"]
                                if hasattr(original_mm, "tolist"):
                                    original_mm = original_mm.tolist()
                                scaled_mm = scaled_inputs.get("multi_modal_data")
                                if hasattr(scaled_mm, "tolist"):
                                    scaled_mm = scaled_mm.tolist()
                                updated_mm = list(scaled_mm)
                                for i, keep_one in enumerate(scale_one_mask_cpu):
                                    if keep_one:
                                        updated_mm[i] = original_mm[i]
                                scaled_inputs["multi_modal_data"] = updated_mm
                            if "scaled_messages" in scaled_inputs:
                                scaled_messages = list(scaled_inputs.get("scaled_messages"))
                                for i, keep_one in enumerate(scale_one_mask_cpu):
                                    if keep_one:
                                        scaled_messages[i] = messages[i]
                                scaled_inputs["scaled_messages"] = scaled_messages
                        else:
                            scales = scaled_inputs.get("scales", None)
                            if isinstance(scales, torch.Tensor):
                                scale_one_mask_tensor = torch.as_tensor(
                                    scale_one_mask_cpu, device=scales.device, dtype=torch.bool
                                )
                                scales = scales.clone()
                                scales[scale_one_mask_tensor] = 1.0
                                scaled_inputs["scales"] = scales

                # Write back tensor fields; rename 'log_probs' -> 'predictor_old_log_probs'
                tensor_updates = {
                    (key if key != "log_probs" else "predictor_old_log_probs"): value
                    for key, value in scaled_inputs.items()
                    if isinstance(value, torch.Tensor)
                }

                non_tensor_updates = {}
                if return_mm_data:
                    if "multi_modal_data" in scaled_inputs:
                        non_tensor_updates["multi_modal_data"] = np.array(scaled_inputs.get("multi_modal_data"), dtype=object)
                    if "scaled_messages" in scaled_inputs:
                        # print(f"before scale raw_prompt: {messages}")
                        non_tensor_updates["raw_prompt"] = np.array(scaled_inputs.get("scaled_messages"), dtype=object)
                        # print(f"after scale raw_prompt: {non_tensor_updates["raw_prompt"]}")
                else:
                    non_tensor_updates["scales"] = to_numpy_cpu(scaled_inputs.get("scales"))
                    non_tensor_updates["scale_mask"] = to_numpy_cpu(scaled_inputs.get("scale_mask"))

                micro_batch.batch.update(tensor_updates)
                micro_batch.non_tensor_batch.update(non_tensor_updates)

            processed_micro_batches.append(micro_batch.to("cpu"))

        # Concatenate micro-batches and restore original keys
        updated_data = DataProto.concat(processed_micro_batches)
        updated_data.non_tensor_batch["ori_prompt"] = original_data["ori_prompt"]

        return updated_data

    def compute_pred_log_prob(self, micro_batch):
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            messages = micro_batch["ori_prompt"]
            actions = micro_batch["actions"]
            scale_mask = micro_batch["scale_mask"]
            video2list = micro_batch["video2list"]
            video2image = micro_batch["video2image"]
            compute_frame_metrics = micro_batch["compute_frame_metrics"]
            # return_pred_extras = micro_batch["return_pred_extras"]

            scaled_inputs = self.actor_module(
                messages=messages,
                actions=actions,
                scale_mask=scale_mask,
                video2list=video2list,
                video2image=video2image,
                compute_frame_metrics=compute_frame_metrics,
                # return_pred_extras=return_pred_extras,
            )
            predictor_log_probs = scaled_inputs['log_probs']

        return predictor_log_probs, scaled_inputs
    ###

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    @GPUMemoryLogger(role="dp predictor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch:
            non_tensor_select_keys.append("uid")

        ###
        # non_tensor_select_keys.append("multi_modal_data") "ori_attention_mask", "ori_position_ids"
        is_pred = data.meta_info.pop("is_pred", False)
        video2list = data.meta_info.pop("video2list", False)
        video2image = data.meta_info.pop("video2image", False)
        compute_frame_metrics = data.meta_info.pop("compute_frame_metrics", False)
        contrastive_coef = data.meta_info.pop("contrastive_coef", 0.0)
        sim_scale_coef = data.meta_info.pop("sim_scale_coef", 0.0)
        concentration_coef = data.meta_info.pop("concentration_coef", 0.0)

        # print(data.batch.keys())
        # print(data.batch['actions'].shape)
        select_keys.extend(
            ["actions", "predictor_old_log_probs", "scale_mask", "predictor_advantages", "predictor_update_mask"]
        )
        non_tensor_select_keys.extend(["ori_prompt"])
        ###
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "predictor/pg_loss": 0.0,
            "predictor/kl_loss": 0.0,
            "predictor/sim_loss": 0.0,
            "predictor/contrastive_loss": 0.0,
        }
        ###
        predictor_log_probs = []
        frame_metrics_acc = {}
        ###
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                # print("self.config.use_dynamic_bsz", self.config.use_dynamic_bsz)
                # print(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    # response_mask = model_inputs["response_mask"]
                    # old_log_prob = model_inputs["old_log_probs"]
                    # advantages = model_inputs["advantages"]
                    ###
                    model_inputs["video2list"] = video2list
                    model_inputs["video2image"] = video2image
                    model_inputs["compute_frame_metrics"] = compute_frame_metrics
                    response_mask = model_inputs.get("scale_mask", model_inputs["response_mask"])
                    predictor_update_mask = model_inputs.get("predictor_update_mask", None)
                    if predictor_update_mask is not None:
                        if response_mask is None:
                            response_mask = predictor_update_mask
                        else:
                            if predictor_update_mask.dim() < response_mask.dim():
                                predictor_update_mask = predictor_update_mask.unsqueeze(-1).expand_as(response_mask)
                            response_mask = response_mask & predictor_update_mask
                    old_log_prob = model_inputs.get("predictor_old_log_probs", None)
                    advantages = model_inputs.get("predictor_advantages", None)
                    ###

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # entropy, log_prob = self._forward_micro_batch(
                    #     model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    # )
                    
                    ### model_inputs['ori_multi_modal_inputs'][0]['pixel_values']
                    log_prob, pred_outputs = self.compute_pred_log_prob(model_inputs)
                    if is_pred:
                        predictor_log_probs.append(log_prob.detach())
                    frame_features = pred_outputs.get("frame_features", None) if pred_outputs is not None else None
                    scales = pred_outputs.get("scales", None) if pred_outputs is not None else None
                    entropy = pred_outputs.get("entropy", None) if pred_outputs is not None else None
                    fm = pred_outputs.get("frame_metrics", None) if pred_outputs is not None else None
                    if compute_frame_metrics and frame_metrics_acc is not None and fm is not None:
                        for k, v in fm.items():
                            if isinstance(v, torch.Tensor):
                                frame_metrics_acc.setdefault(k, []).append(v.detach())
                    # print("log_prob", log_prob.shape)

                    if advantages is None:
                        print("scale is None!")
                        advantages = model_inputs["advantages"]

                        advantages = (advantages * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
                        advantages = advantages.unsqueeze(-1).expand_as(log_prob)
                    ###

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        ###
                        old_log_prob = model_inputs["predictor_old_log_probs"]
                        ###
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            ###
                            old_log_prob = model_inputs["predictor_old_log_probs"]
                            ###

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    if loss_mode == "frame_pg":
                        pg_loss = agg_loss(
                            loss_mat=-(log_prob * advantages),
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            **self.config.global_batch_info,
                        )
                        pg_metrics = {}
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                    # print("pg_loss", pg_loss)
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if entropy_coeff != 0 and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["predictor/entropy"] = entropy_agg.detach().item()
                        policy_loss -= entropy_agg * entropy_coeff 

                    concentration_loss = pred_outputs.get("concentration_loss", None) if pred_outputs else None
                    if concentration_coef != 0 and concentration_loss is not None:
                        policy_loss = policy_loss + concentration_loss * concentration_coef
                        micro_batch_metrics["predictor/concentration_loss"] = concentration_loss.detach().item()

                    # Sim scale loss: similar frames -> later frame lower scale (compress redundancy)
                    # Try to get from model (V2), fallback to local computation (V1)
                    # sim_scale_coef = self.config.get("sim_scale_coef", 0.0)
                    if sim_scale_coef > 0:
                        sim_loss = pred_outputs.get("sim_scale_loss", None) if pred_outputs else None
                        
                        # Fallback: compute locally if model didn't return it
                        if sim_loss is None and frame_features is not None and scales is not None:
                            frame_mask = response_mask.bool()
                            if frame_mask.dim() < scales.dim():
                                frame_mask = frame_mask.unsqueeze(-1).expand_as(scales)
                            f0 = frame_features[:, :-1]
                            f1 = frame_features[:, 1:]
                            sim = F.cosine_similarity(f0, f1, dim=-1)
                            tau = self.config.get("sim_tau", 0.5)
                            temp = self.config.get("sim_temp", 0.1)
                            gamma = self.config.get("sim_gamma", 0.05)
                            w = torch.sigmoid((sim - tau) / temp)
                            m = (frame_mask[:, :-1] & frame_mask[:, 1:]).float()
                            s = scales.clamp_min(1e-6).log()
                            target = s[:, :-1] - gamma * w
                            err = (s[:, 1:] - target).abs()
                            sim_loss = (err * w * m).sum() / (m.sum() + 1e-6)
                        
                        if sim_loss is not None and isinstance(sim_loss, torch.Tensor):
                            # FSDP-safe: always add loss to ensure consistent backward across ranks
                            # Avoid requires_grad check which could cause different code paths
                            # print(f"[sim_loss] value={sim_loss}")
                            policy_loss = policy_loss + sim_loss * sim_scale_coef
                            metrics["predictor/sim_loss"] += sim_loss.detach().item()
                            # if sim_loss.requires_grad:
                            #     print(f"[sim_loss] value={sim_loss.item():.6f}, policy_loss={policy_loss.item():.4f}")

                    # Contrastive loss for frame differentiation (V2 predictor)
                    # Read directly from pred_outputs dict (cleaner, FSDP-safe)
                    # contrastive_coef = self.config.get("contrastive_coef", 0.0)
                    if contrastive_coef > 0:
                        contrastive_loss = pred_outputs.get("contrastive_loss", None) if pred_outputs else None
                        # print(f"[contrastive_loss] value={contrastive_loss}")
                        if contrastive_loss is not None and isinstance(contrastive_loss, torch.Tensor):
                            # FSDP-safe: always add loss to ensure consistent backward across ranks
                            policy_loss = policy_loss + contrastive_loss * contrastive_coef
                            metrics["predictor/contrastive_loss"] += contrastive_loss.detach().item()
                            # if contrastive_loss.requires_grad:
                            #     print(f"[contrastive_loss] value={contrastive_loss.item():.6f}")

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["predictor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["predictor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    ###
                    # breakpoint()
                    final_metrics = {}
                    for k, v in micro_batch_metrics.items():
                        if k.startswith("actor/"):
                            new_key = k.replace("actor/", "predictor/")
                            final_metrics[new_key] = v
                        else:
                            final_metrics[k] = v
                    micro_batch_metrics = final_metrics
                    ###
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"predictor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        # return metrics
        output = (metrics,)
        if is_pred:
            output += (torch.cat(predictor_log_probs, dim=0),)
        if compute_frame_metrics and frame_metrics_acc:
            frame_metrics_cat = {k: torch.cat(vs, dim=0) for k, vs in frame_metrics_acc.items()}
            output += (frame_metrics_cat,)
        return output
        ###
        # if hasattr(data.batch, "is_locked") and data.batch.is_locked:
        #     data.batch.unlock_()
        
        # if hasattr(data.batch, "unlock_"):
        #     data.batch.unlock_()

        # was_locked = False
        # if hasattr(data.batch, "is_locked"):
        #     was_locked = data.batch.is_locked
        
        # if was_locked and hasattr(data.batch, "unlock_"):
        #     data.batch.unlock_()
            
        # data.batch["predictor_log_probs"] = torch.cat(predictor_log_probs, dim=0)
        # data.meta_info["metrics"] = metrics

        # if was_locked and hasattr(data.batch, "lock_"):
        #     data.batch.lock_()
        
        # return data
        ###

    @GPUMemoryLogger(role="dp predictor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs
