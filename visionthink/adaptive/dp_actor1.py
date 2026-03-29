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

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, predictor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"
        ###
        self.predictor_optimizer = predictor_optimizer
        ###

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

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

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
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

                ###
                multi_modal_data = micro_batch.pop("ori_multi_modal_data", None)
                # predictor_log_probs = None
                if multi_modal_data is not None:
                    # ori_input_ids = micro_batch["ori_input_ids"]
                    # ori_attention_mask = micro_batch["ori_attention_mask"]
                    # ori_position_ids = micro_batch["ori_position_ids"]
                    # ori_multi_modal_inputs = extract_multi_modal_inputs(micro_batch["ori_multi_modal_inputs"])
                    # actions = micro_batch["actions"]
                    extra_args.update({
                        "ori_input_ids": micro_batch["ori_input_ids"],
                        "ori_attention_mask": micro_batch["ori_attention_mask"],
                        "ori_position_ids": micro_batch["ori_position_ids"],
                        "ori_multi_modal_inputs": extract_multi_modal_inputs(micro_batch["ori_multi_modal_inputs"]),
                        "actions": micro_batch["actions"],
                        "multi_modal_data": multi_modal_data
                    })
                ###

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
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
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
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
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

            return entropy, log_probs, output.log_probs

    ###
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def scale_multi_modal(self, data: DataProto, calculate_entropy=False):
        """
        Batch multi-modal scaling pre-processing:
        Safely and efficiently generate updated `input_ids/attention_mask/position_ids`
        and corresponding `multi_modal_inputs/multi_modal_data` for subsequent policy updates.
        """
        import numpy as np

        self.actor_module.eval()

        # Use dynamic micro-batch size if available; fallback to 8
        micro_batch_size = data.meta_info.get("micro_batch_size", 8) if hasattr(data, "meta_info") else 8
        micro_batches = data.split(micro_batch_size)

        # Backup original fields (safe fallback if optional keys are missing)
        ori_mm_inputs = data.non_tensor_batch.get("multi_modal_inputs", None)
        ori_mm_data = data.non_tensor_batch.get("multi_modal_data", None)
        original_data = {
            "ori_input_ids": data.batch["input_ids"].clone(),
            "ori_attention_mask": data.batch["attention_mask"].clone(),
            "ori_position_ids": data.batch["position_ids"].clone(),
            "ori_multi_modal_inputs": ori_mm_inputs.copy() if isinstance(ori_mm_inputs, dict) else ori_mm_inputs,
            "ori_multi_modal_data": ori_mm_data.copy() if isinstance(ori_mm_data, list) else ori_mm_data,
        }

        processed_micro_batches = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # Inference-only pre-processing; wrap model forward in autocast for performance
            with torch.inference_mode():
                from verl.utils.model import extract_multi_modal_inputs
                multi_modal_inputs = {}
                if "multi_modal_inputs" in model_inputs:
                    multi_modal_inputs = extract_multi_modal_inputs(model_inputs["multi_modal_inputs"])

                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    input_ids = model_inputs["input_ids"]
                    attention_mask = model_inputs["attention_mask"]
                    position_ids = model_inputs["position_ids"]
                    multi_modal_data = model_inputs.get("multi_modal_data", None)

                    scaled_inputs = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        multi_modal_data=multi_modal_data,
                        **multi_modal_inputs,
                    )

                # Write back tensor fields; rename 'log_probs' -> 'predictor_old_log_probs'
                tensor_updates = {
                    (key if key != "log_probs" else "predictor_old_log_probs"): value
                    for key, value in scaled_inputs.items()
                    if isinstance(value, torch.Tensor)
                }

                # Handle non-tensor fields: per-sample split of pixel_values and image_grid_thw
                non_tensor_updates = {}
                updated_mm_inputs = scaled_inputs.get("multi_modal_inputs")
                updated_mm_data = scaled_inputs.get("multi_modal_data")

                if updated_mm_inputs is not None and "pixel_values" in updated_mm_inputs:
                    # Split pixel_values per sample using patch counts from image_grid_thw
                    pixel_values = updated_mm_inputs["pixel_values"]
                    grid_thw_batch = updated_mm_inputs.get("image_grid_thw")

                    split_sizes = grid_thw_batch.prod(-1).tolist()
                    pixel_values_batch = pixel_values.split(split_sizes, dim=0)
                    batch_size = grid_thw_batch.shape[0]

                    list_of_mm_dicts = []
                    for i in range(batch_size):
                        sample_dict = {
                            "pixel_values": pixel_values_batch[i].cpu()
                        }
                        # Preserve shape (1, 3) and exact values for image_grid_thw
                        sample_dict["image_grid_thw"] = grid_thw_batch[i].unsqueeze(0).cpu()
                        list_of_mm_dicts.append(sample_dict)

                    non_tensor_updates["multi_modal_inputs"] = np.array(list_of_mm_dicts, dtype=object)

                if updated_mm_data is not None:
                    non_tensor_updates["multi_modal_data"] = np.array(updated_mm_data, dtype=object)

                micro_batch.batch.update(tensor_updates)
                micro_batch.non_tensor_batch.update(non_tensor_updates)

            processed_micro_batches.append(micro_batch.to("cpu"))

        # Concatenate micro-batches and restore original keys
        updated_data = DataProto.concat(processed_micro_batches)
        updated_data.batch["ori_input_ids"] = original_data["ori_input_ids"]
        updated_data.batch["ori_attention_mask"] = original_data["ori_attention_mask"]
        updated_data.batch["ori_position_ids"] = original_data["ori_position_ids"]

        if original_data["ori_multi_modal_inputs"] is not None:
            updated_data.non_tensor_batch["ori_multi_modal_inputs"] = original_data["ori_multi_modal_inputs"]
        if original_data["ori_multi_modal_data"] is not None:
            updated_data.non_tensor_batch["ori_multi_modal_data"] = original_data["ori_multi_modal_data"]

        # Safely remove unused fields
        # updated_data.non_tensor_batch.pop("raw_prompt_ids", None)
        return updated_data
    
    def compute_pred_log_prob(self, micro_batch):
        ori_multi_modal_data = micro_batch.pop("ori_multi_modal_data", None)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            predictor_log_probs = None
            if ori_multi_modal_data is not None:
                from verl.utils.model import extract_multi_modal_inputs

                ori_input_ids = micro_batch["ori_input_ids"]
                ori_attention_mask = micro_batch["ori_attention_mask"]
                ori_position_ids = micro_batch["ori_position_ids"]
                ori_multi_modal_inputs = extract_multi_modal_inputs(micro_batch["ori_multi_modal_inputs"])
                actions = micro_batch["actions"]

                scaled_inputs = self.actor_module(
                    input_ids=ori_input_ids,
                    attention_mask=ori_attention_mask,
                    position_ids=ori_position_ids,
                    multi_modal_data=ori_multi_modal_data,
                    actions=actions,
                    **ori_multi_modal_inputs,
                )
                predictor_log_probs = scaled_inputs['log_probs']

            return predictor_log_probs
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
            ###
            if self.predictor_optimizer is not None:
                self.scaler.step(self.predictor_optimizer)
            ###
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
                ###
                if self.predictor_optimizer is not None:
                    self.predictor_optimizer.zero_grad()
                ###
            else:
                self.actor_optimizer.step()
                ###
                if self.predictor_optimizer is not None:
                    self.predictor_optimizer.step()
                ###
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                ###
                entropy, log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
                ###
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
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
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        ###
        # non_tensor_select_keys.append("multi_modal_data")
        if "ori_input_ids" in data.batch.keys():
        # if self.config.get("scale_multi_modal_data", None):
            # data.non_tensor_batch['ori_multi_modal_inputs'] = updated_data.non_tensor_batch['extra_info'].pop('ori_multi_modal_inputs', None)
            select_keys.extend(["ori_input_ids", "ori_attention_mask", "ori_position_ids", "actions", "predictor_old_log_probs", "scales"])
            non_tensor_select_keys.extend(["ori_multi_modal_inputs", "ori_multi_modal_data"])

            if self.config.get("scale_n", 1) > 1:
                select_keys.extend(["predictor_advantages", "predictor_update_mask"])
        ###

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
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

                self.actor_optimizer.zero_grad()
                ###
                if self.predictor_optimizer is not None:
                    self.predictor_optimizer.zero_grad()
                ###

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    ###
                    if self.actor_module._debug_nan(optimizer=self.actor_optimizer) or self.actor_module._debug_nan(optimizer=self.predictor_optimizer):
                        with FSDP.summon_full_params(self.actor_module, writeback=False, rank0_only=False):
                            breakpoint()
                    ###

                    # all return: (bsz, response_length)
                    ###
                    entropy, log_prob, predictor_log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    ###

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    ### update predictor
                    # predictor_log_prob = self.compute_pred_log_prob(model_inputs)
                    if self.predictor_optimizer is not None and predictor_log_prob is not None:
                        predictor_old_log_probs = model_inputs.get("predictor_old_log_probs", None)
                        if predictor_old_log_probs is not None:
                            if on_policy:
                                predictor_old_log_prob_used = predictor_log_prob.detach()
                            else:
                                predictor_old_log_prob_used = predictor_old_log_probs

                        sample_advantage = (advantages * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8) # (B,)

                        if predictor_log_prob.dim() == 1:
                            # Case A: Sample-level (B,)
                            predictor_adv = sample_advantage
                            predictor_mask = torch.ones_like(predictor_log_prob, dtype=torch.bool) # (B,)

                            predictor_log_prob_sum = predictor_log_prob.detach()
                            predictor_old_log_prob_sum = predictor_old_log_prob_used
                            
                        else:
                            # Case B: Frame-level (B, T)
                            T = predictor_log_prob.shape[1]
                            predictor_adv = sample_advantage.unsqueeze(1).expand(-1, T) # (B, 1) -> (B, T)
                            predictor_mask = torch.ones_like(predictor_log_prob, dtype=torch.bool) # (B, T)

                            predictor_log_prob_sum = predictor_log_prob.sum(dim=-1).detach()
                            predictor_old_log_prob_sum = predictor_old_log_prob_used.sum(dim=-1)

                        negative_approx_kl = predictor_log_prob_sum - predictor_old_log_prob_sum # (B,)
                        # Clamp negative_approx_kl for stability
                        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                        ratio = torch.exp(negative_approx_kl).unsqueeze(1).expand(-1, advantages.shape[-1])
                        advantages = advantages * ratio

                        pred_pg_loss, pred_pg_metrics = policy_loss_fn(
                            old_log_prob=predictor_old_log_prob_used,
                            log_prob=predictor_log_prob,
                            advantages=predictor_adv,  # (B,) or (B, T)
                            response_mask=predictor_mask,  # (B,) or (B, T)
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=None,
                        )

                        scale_cost_loss = model_inputs['scales'].mean()
                        # policy_loss = pred_pg_loss

                        # if self.config.use_dynamic_bsz:
                        #     # relative to the dynamic bsz
                        #     pred_loss = pred_pg_loss * loss_scale_factor
                        # else:
                        #     pred_loss = pred_pg_loss * loss_scale_factor
                        # if self.scaler is not None:
                        #     self.scaler.scale(pred_loss).backward()
                        # else:
                        #     pred_loss.backward()
                        
                        # for k, v in pred_pg_metrics.items():
                        #     micro_batch_metrics[f"predictor/{k}"] = v
                        # micro_batch_metrics[f"predictor/pg_loss"] = pred_loss.detach().item() * loss_scale_factor

                    ###

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
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
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
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

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                    def check_graph_connection(loss_a, loss_b, tensor_name="Backbone Output"):
                        """
                        检查两个 Loss 是否通过计算图连接到了一起。
                        原理：遍历 grad_fn 链条（简单版）。
                        """
                        print(f"\n🕸️ 正在分析计算图依赖...")
                        
                        # 获取两个 Loss 的梯度函数 (Grad Function)
                        roots_a = set()
                        roots_b = set()
                        
                        def traverse(grad_fn, visited_set, depth=0, max_depth=500):
                            if grad_fn is None or depth > max_depth:
                                return
                            if grad_fn in visited_set:
                                return
                            visited_set.add(grad_fn)
                            
                            # 递归遍历父节点
                            if hasattr(grad_fn, 'next_functions'):
                                for f, _ in grad_fn.next_functions:
                                    traverse(f, visited_set, depth + 1)
                        # def traverse(grad_fn, visited_set, depth=0, max_depth=500):
                        #     indent = "|  " * depth  # 生成缩进，每深一层多一个 "|  "
                            
                        #     if grad_fn is None:
                        #         return
                            
                        #     if depth > max_depth:
                        #         print(f"{indent} ... (达到最大深度限制 {max_depth})")
                        #         return

                        #     # 获取节点名称 (例如 AddBackward0, MmmBackward0)
                        #     node_name = type(grad_fn).__name__
                            
                        #     # === 特殊处理叶子节点 (AccumulateGrad) ===
                        #     # AccumulateGrad 代表梯度累积到了具体的 Parameter 上
                        #     # 我们打印出这个 Parameter 的形状，这对于识别是哪个层非常有帮助
                        #     extra_info = ""
                        #     if "AccumulateGrad" in node_name and hasattr(grad_fn, 'variable'):
                        #         param = grad_fn.variable
                        #         extra_info = f" ---> 🟢 目标参数 Shape: {tuple(param.shape)} | Dtype: {param.dtype}"
                            
                        #     # === 打印逻辑 ===
                        #     if grad_fn in visited_set:
                        #         # 如果已经访问过，打印一个标记并返回，避免无限循环打印
                        #         # print(f"{indent} -> {node_name} (已访问，跳过展开)")
                        #         return

                        #     print(f"{indent} -> {node_name}{extra_info}")
                            
                        #     visited_set.add(grad_fn)
                            
                        #     # 递归遍历父节点 (next_functions)
                        #     if hasattr(grad_fn, 'next_functions'):
                        #         for i, (f, _) in enumerate(grad_fn.next_functions):
                        #             # 为了不让输出太乱，可以只打印前几个分支，或者全部打印
                        #             traverse(f, visited_set, depth + 1)

                        print("正在遍历 Loss A 的计算图...")
                        traverse(loss_a.grad_fn, roots_a)
                        print(f"Loss A 涉及 {len(roots_a)} 个计算节点")

                        print("正在遍历 Loss B 的计算图...")
                        traverse(loss_b.grad_fn, roots_b)
                        print(f"Loss B 涉及 {len(roots_b)} 个计算节点")
                        
                        # 找交集
                        intersection = roots_a.intersection(roots_b)
                        
                        if len(intersection) > 0:
                            print(f"🚨 严重警告：发现 {len(intersection)} 个共享的计算图节点！")
                            print("这意味着两个 Loss 依赖于同一组中间计算结果。")
                            print("❌ 你绝对不能分别调用 backward()！必须相加后调用一次 backward()。")
                            
                            # 尝试打印一些节点名字看看是什么层
                            print("部分共享节点类型:", [type(fn).__name__ for fn in list(intersection)[:5]])
                        else:
                            print("✅ 未发现计算图交集。这两个 Loss 看起来是完全独立的计算路径（不太可能）。")
                    if self.predictor_optimizer is not None and self.actor_optimizer is not None:
                        check_graph_connection(pred_pg_loss, policy_loss)

                    tensors_to_check = [
                        ("advantages", advantages),
                        ("response_mask", response_mask)
                    ]
                    if self.predictor_optimizer is not None and predictor_log_prob is not None:
                        tensors_to_check.extend([
                            ("predictor_old_log_prob_used", predictor_old_log_prob_used),
                            ("predictor_adv", predictor_adv),
                            ("predictor_mask", predictor_mask),
                            ("predictor_log_prob", predictor_log_prob),
                            ("pred_pg_loss", pred_pg_loss),
                        ])
                    if self.actor_optimizer is not None:
                        tensors_to_check.extend([
                            ("policy_loss", policy_loss),
                            ("old_log_prob", old_log_prob),
                            ("log_prob", log_prob)
                        ])

                    if any(torch.isnan(t[1]).any() for t in tensors_to_check):
                        # with FSDP.summon_full_params(self.actor_module, writeback=False, rank0_only=True):
                        #     breakpoint()
                        # model_inputs['input_ids'][0][1500:2100]
                        # model_inputs['ori_input_ids'][0][800:]
                        # model_inputs['input_ids'][8:12]
                        # self.actor_module.vision_tower.blocks[0]._fsdp_wrapped_module.attn.qkv.weight
                        # self.actor_module.visual.blocks[0]._fsdp_wrapped_module.attn.qkv.weight
                        print("\n" + "!"*20 + " NaN DETECTED " + "!"*20)
                        for var_name, t in tensors_to_check:
                            if torch.isnan(t).any():
                                print(f"NaN detected in {var_name}: {t}")
                        # print("old_log_prob.shape:", old_log_prob.shape)
                        # print("log_prob.shape:", log_prob.shape)
                        # print("advantages.shape:", advantages.shape)
                        # print("response_mask.shape", response_mask.shape)

                        # print("log_prob", log_prob)
                        # print("old_log_prob:", old_log_prob)
                        # print("policy_loss", policy_loss)
                        # print("advantages", advantages)
                        # print("response_mask", response_mask)


                        # if self.predictor_optimizer is not None and predictor_log_prob is not None:
                        #     print("predictor_old_log_prob_used.shape", predictor_old_log_prob_used.shape)
                        #     print("predictor_adv.shape", predictor_adv.shape)
                        #     print("predictor_mask.shape", predictor_mask.shape)

                        #     print("predictor_adv", predictor_adv)
                        #     print("predictor_old_log_prob_used", predictor_old_log_prob_used)
                        #     print("predictor_log_prob", predictor_log_prob)

                        #     print("pred_pg_loss", pred_pg_loss)
                        #     print("scale_cost_loss", scale_cost_loss)

                        # print("policy_loss", policy_loss)
                        print("!"*54 + "\n")
                    # policy_loss += pred_pg_loss
                    def gradient_check():
                        print("\n=== Gradient Check ===")
                        for name, param in self.actor_module.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                print(f"[有梯度] {name} | Grad Norm: {grad_norm:.6f}")
                            # else:
                            #     print(f"[无梯度] {name}")
                        print("======================\n")


                    if self.actor_module._debug_nan(optimizer=self.actor_optimizer) or self.actor_module._debug_nan(optimizer=self.predictor_optimizer):
                        with FSDP.summon_full_params(self.actor_module, writeback=False, rank0_only=False):
                            breakpoint()
                        # self.actor_module.predictor.mlp._fsdp_wrapped_module.net[1].weight
                        # self.actor_module.predictor.scorer.layers[2].ff_spatio_temporal[1].weight
                        # self.actor_module.vision_tower.blocks[4].attn.qkv.weight
                        # self.actor_module.language_model.layers[35].self_attn.v_proj.weight   self.actor_module.language_model.layers[35].mlp.down_proj.weight.grad   self.actor_module.language_model.layers[35].mlp.up_proj.weight.grad
                        # self.actor_module.predictor.scorer.layers[2].spatial_attn.to_out.weight.grad
                        # self.actor_module.visual.blocks[1].mlp.up_proj.weight.grad
                        # torch.isnan(self.actor_module.language_model.embed_tokens.weight).any()
                        # breakpoint()
                        # self.actor_module._debug_nan(optimizer=self.actor_optimizer)
                        # self.actor_module._debug_nan(optimizer=self.predictor_optimizer)
                        # output = self.actor_module(input_ids=input_ids_rmpad,attention_mask=None,position_ids=position_ids_rmpad,**multi_modal_inputs,use_cache=False,**extra_args,)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        ###
        if self.predictor_optimizer is not None:
            self.predictor_optimizer.zero_grad()
        ###
        return metrics