import os
from typing import Any, Optional, Union
import vllm.inputs.preprocess
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.multimodal.inputs import (
    MultiModalUUIDDict,
)
from vllm.inputs.data import (
    ProcessorInputs,
    PromptType,
)
from transformers.image_transforms import convert_to_rgb
    
import vllm.model_executor.models.qwen2_5_vl
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor

from PIL import Image


MAX_PIXELS = 14 * 14 * 4 * 16384

_original_init = Qwen2_5_VLMultiModalProcessor.__init__
def __init__(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)

    predictor_path = os.getenv("PREDICTOR_PATH", None)
    print("predictor_path", predictor_path)

    if predictor_path is None:
        # raise ValueError("PREDICTOR_PATH is not set")
        print("PREDICTOR_PATH is not set, use ENABLE_BASELINE_SCALE")
        if os.environ.get("ENABLE_BASELINE_SCALE", None) is None:
            raise ValueError("ENABLE_BASELINE_SCALE is not set")
        
    else:
        self.predictor = AutoModel.from_pretrained(
            predictor_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        self.predictor.eval()


def preprocess(
    self,
    prompt: PromptType,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    *,
    mm_uuids: Optional[MultiModalUUIDDict] = None,
) -> ProcessorInputs:
    """Preprocess the input prompt."""
    if self.model_config.is_encoder_decoder:
        # Encoder-decoder model requires special mapping of
        # input prompts to encoder & decoder.
        return self._process_encoder_decoder_prompt(
            prompt,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    if is_explicit_encoder_decoder_prompt(prompt):
        raise ValueError("Cannot pass encoder-decoder prompt "
                            "to decoder-only models")

    ###
    if self.predictor:
        text = prompt['prompt']
        image_inputs = prompt['multi_modal_data'].get('image', None)
        video_inputs = prompt['multi_modal_data'].get('video', None)
        images = [convert_to_rgb(image) for image in image_inputs]
        inputs = self.processor(
            text=[text],
            images=images,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.predictor.device)
        inputs.update({"multi_modal_data": [{"image": images}], "text": [text], "eval_mode": True})
        scaled_images = self.predictor(**inputs)
        prompt = {"prompt": text, "multi_modal_data": {"image": scaled_images}}
        print("prompt", prompt)
    ###
    
    # Decoder-only operation
    res = self._process_decoder_only_prompt(
        prompt,
        tokenization_kwargs=tokenization_kwargs,
        mm_uuids=mm_uuids,
    )

    return res


def _apply_hf_processor_main(
    self,
    prompt,
    mm_items,
    hf_processor_mm_kwargs,
    tokenization_kwargs,
    *,
    enable_hf_prompt_update: bool,
):
    """
    Apply the HF processor on the prompt text and multi-modal data.

    In addition, return whether prompt updates have been applied
    (for most HF processors, this should be `True`).

    Note:
        If `enable_hf_prompt_update=False`, we use HF processor
        to perform prompt updates if available; HF processor requires
        that the prompt corresponds to multi-modal items.
    """
    processor_data, passthrough_data = self._get_hf_mm_data(mm_items)
    # images = processor_data.get("images", None)
    # videos = processor_data.get("videos", None)
    # if images is not None:
    #     if os.environ.get("ENABLE_BASELINE_SCALE", None):
    #         fixed_scale = float(os.environ.get("BASELINE_SCALE_FACTOR", "1.5"))
    #         print(f"[Baseline Mode] Scaling all images by factor: {fixed_scale}")

    #         scaled_images = []
    #         for img in images:
    #             w, h = img.size
    #             new_w = int(w * fixed_scale)
    #             new_h = int(h * fixed_scale)
                
    #             new_w = max(14, new_w)
    #             new_h = max(14, new_h)

    #             resized_h, resized_w = smart_resize(new_h, new_w, 14 * 2, max_pixels=MAX_PIXELS)

    #             resized_img = img.resize((resized_w, resized_h), resample=Image.BICUBIC)
    #             scaled_images.append(resized_img)
    #             print("before", img)
    #             print("after", resized_img)
                
    #     else:
    #         inputs = self.predictor.processor(
    #             text=[prompt],
    #             images=images,
    #             videos=videos,
    #             padding=True,
    #             return_tensors="pt",
    #         ).to(self.predictor.device)
            
    #         inputs.update({"multi_modal_data": [{"image": images}], "text": [prompt], "eval_mode": True})

    #         scaled_multi_modal_data = self.predictor(**inputs)["multi_modal_data"]
    #         scaled_images = [img for mm_item in scaled_multi_modal_data for img in mm_item["image"]]
        
    #     mm_items = self._to_mm_items({"image": scaled_images})

        # # print("images", images)
        # images = [convert_to_rgb(image) for image in images]
        # inputs = self.predictor.processor(
        #     text=[prompt],
        #     images=images,
        #     videos=videos,
        #     padding=True,
        #     return_tensors="pt",
        # ).to(self.predictor.device)
        # inputs.update({"multi_modal_data": [{"image": images}], "text": [prompt], "eval_mode": True})

        # scaled_multi_modal_data = self.predictor(**inputs)["multi_modal_data"]
        # scaled_images = [img for mm_item in scaled_multi_modal_data for img in mm_item["image"]]
        
        # mm_items = self._to_mm_items({"image": scaled_images})

    if isinstance(prompt, str):
        if enable_hf_prompt_update:
            return self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )

        prompt_ids = self._apply_hf_processor_text_only(
            prompt, tokenization_kwargs)
    else:
        prompt_ids = self._apply_hf_processor_tokens_only(prompt)

    mm_processed_data = self._apply_hf_processor_mm_only(
        mm_items=mm_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        tokenization_kwargs=tokenization_kwargs,
    )
    # breakpoint()
    print("hf_processor_mm_kwargs", hf_processor_mm_kwargs)
    print("self.info.get_hf_processor(**mm_kwargs)", self.info.get_hf_processor(**hf_processor_mm_kwargs))
    return prompt_ids, mm_processed_data, False


# Apply monkey patch to InputPreprocessor
# vllm.inputs.preprocess.InputPreprocessor.__init__ = __init__
# vllm.inputs.preprocess.InputPreprocessor.preprocess = preprocess


# Qwen2_5_VLMultiModalProcessor.__init__ = __init__
# vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._apply_hf_processor_main = _apply_hf_processor_main


from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.sampling_params import SamplingType
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.model_executor.models.interfaces import (SupportsMultiModal,
                                                   is_mixture_of_experts,
                                                   supports_eagle3,
                                                   supports_mrope,
                                                   supports_multimodal_pruning,
                                                   supports_transcription)
def _update_states(self, scheduler_output) -> None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Free the cached encoder outputs.
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    reqs_to_add: list[CachedRequestState] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if sampling_params and \
            sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)
        print("use_mrope:", self.uses_mrope)
        print("self.model_config", self.model_config)
        print("supports_mrope(self.model)", supports_mrope(self.model))
        print("self.model", self.model)
        reqs_to_add.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_data.resumed_from_preemption[i]

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            # When using PP, the scheduler sends the sampled tokens back,
            # because there's no direct communication between the first-
            # stage worker and the last-stage worker.
            new_token_ids = req_data.new_token_ids[i]
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec tokens.
            num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    new_token_ids[-num_new_tokens:])

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                                new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            reqs_to_add.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = (
            num_computed_tokens)
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(
                new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = new_token_ids
            self.input_batch.num_tokens_no_spec[
                req_index] = end_token_index
            self.input_batch.num_tokens[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        if spec_token_ids:
            num_spec_tokens = len(spec_token_ids)
            start_index = self.input_batch.num_tokens_no_spec[req_index]
            end_token_index = start_index + num_spec_tokens
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec tokens.
            self.input_batch.num_tokens[req_index] += num_spec_tokens

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()

# from vllm.v1.worker.gpu_model_runner import GPUModelRunner
# vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states = _update_states


def add_request(
    self,
    request_id: str,
    prompt,
    params,
    arrival_time = None,
    lora_request = None,
    tokenization_kwargs = None,
    trace_headers = None,
    priority: int = 0,
) -> None:
    # Validate the request_id type.
    if not isinstance(request_id, str):
        raise TypeError(
            f"request_id must be a string, got {type(request_id)}")
    breakpoint()
    # Process raw inputs into the request.
    prompt_str, request = self.processor.process_inputs(
        request_id, prompt, params, arrival_time, lora_request,
        tokenization_kwargs, trace_headers, priority)

    n = params.n if isinstance(params, SamplingParams) else 1

    if n == 1:
        # Make a new RequestState and queue.
        self.output_processor.add_request(request, prompt_str, None, 0)
        # Add the request to EngineCore.
        self.engine_core.add_request(request)
        return

    # Fan out child requests (for n>1).
    parent_req = ParentRequest(request_id, params)
    for idx in range(n):
        request_id, params = parent_req.get_child_info(idx)
        child_request = request if idx == n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = params

        # Make a new RequestState and queue.
        self.output_processor.add_request(child_request, prompt_str,
                                            parent_req, idx)
        # Add the request to EngineCore.
        self.engine_core.add_request(child_request)

# import vllm.v1.engine.llm_engine
# vllm.v1.engine.llm_engine.LLMEngine.add_request = add_request


from functools import lru_cache, partial
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model)
def _preprocess(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors = None,
    ubatch_slices = None,
    num_tokens_after_padding = None,
):

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    if ubatch_slices:
        assert num_tokens_after_padding is not None
        num_input_tokens = int(num_tokens_after_padding[0].item() * 2)
        self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
    elif ubatch_slices is None:
        num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
        num_pad, num_tokens_after_padding = self.get_dp_padding(
            num_input_tokens)
        num_input_tokens += num_pad

    # _prepare_inputs may reorder the batch, so we must gather multi
    # modal outputs after that to ensure the correct order
    if (self.supports_mm_inputs and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder):
        # Run the multimodal encoder if any.
        self._execute_mm_encoder(scheduler_output)
        mm_embeds = self._gather_mm_embeddings(scheduler_output)

        # NOTE(woosuk): To unify token ids and soft tokens (vision
        # embeddings), we always use embeddings (rather than token ids)
        # as input to the multimodal model, even when the input is text.
        inputs_embeds_scheduled = self.model.get_input_embeddings(
            input_ids=self.input_ids.gpu[:num_scheduled_tokens],
            multimodal_embeddings=mm_embeds or None,
        )

        # TODO(woosuk): Avoid the copy. Optimize.
        self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
            inputs_embeds_scheduled)

        input_ids = None
        inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
        model_kwargs = {
            **self._init_model_kwargs(num_scheduled_tokens),
            **self._extract_mm_kwargs(scheduler_output),
        }
    elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
        # Get the input embeddings for the tokens that are not input embeds,
        # then put them into the appropriate positions.
        # TODO(qthequartermasterman): Since even when prompt embeds are
        # enabled, (a) not all requests will use prompt embeds, and (b)
        # after the initial prompt is processed, the rest of the generated
        # tokens will be token ids, it is not desirable to have the
        # embedding layer outside of the CUDA graph all the time. The v0
        # engine avoids this by "double compiling" the CUDA graph, once
        # with input_ids and again with inputs_embeds, for all num_tokens.
        # If a batch only has token ids, then including the embedding layer
        # in the CUDA graph will be more performant (like in the else case
        # below).
        token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens] \
            .nonzero(as_tuple=False) \
            .squeeze(1)
        # Some tokens ids may need to become embeds
        if token_ids_idx.numel() > 0:
            token_ids = self.input_ids.gpu[token_ids_idx]
            tokens_to_embeds = self.model.get_input_embeddings(
                input_ids=token_ids)
            self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

        inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
        model_kwargs = self._init_model_kwargs(num_input_tokens)
        input_ids = None
    else:
        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the CUDA graph.
        input_ids = self.input_ids.gpu[:num_input_tokens]
        inputs_embeds = None
        model_kwargs = self._init_model_kwargs(num_input_tokens)
    if self.uses_mrope:
        positions = self.mrope_positions.gpu[:, :num_input_tokens]
    else:
        positions = self.positions.gpu[:num_input_tokens]

    if get_pp_group().is_first_rank:
        intermediate_tensors = None
    else:
        intermediate_tensors = self.sync_and_slice_intermediate_tensors(
            num_input_tokens, intermediate_tensors, True)

    if (self.model_config.is_encoder_decoder
            and scheduler_output.scheduled_encoder_inputs):
        encoder_inputs = self._extract_encoder_inputs(scheduler_output)
        model_kwargs.update(encoder_inputs)
    # breakpoint()
    print()
    return (
        num_scheduled_tokens,
        num_input_tokens,
        num_tokens_after_padding,
        input_ids,
        inputs_embeds,
        positions,
        intermediate_tensors,
        model_kwargs,
    )
# import vllm.v1.worker.gpu_model_runner
# vllm.v1.worker.gpu_model_runner.GPUModelRunner._preprocess = _preprocess

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias
import torch
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
)
from transformers import BatchFeature, Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLImageInputs
from vllm.utils.tensor_schema import TensorSchema, TensorShape


def get_mrope_input_positions(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    kwargs = MultiModalFeatureSpec.gather_kwargs(
        mm_features,
        {"image_grid_thw", "video_grid_thw", "second_per_grid_ts", "image_timestamps", "video_timestamps"},
    )
    image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
    video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
    second_per_grid_ts = kwargs.get("second_per_grid_ts", [])
    # image_timestamps = kwargs.get("image_timestamps", None)
    # video_timestamps = kwargs.get("video_timestamps", None)
    # video_timestamps = [float(i * 2) for i in range(len(second_per_grid_ts))]
    # video_timestamps = [0, 2, 4, 6, 0, 2, 4, 6]

    hf_config = self.config
    image_token_id = hf_config.image_token_id
    video_token_id = hf_config.video_token_id
    vision_start_token_id = hf_config.vision_start_token_id
    spatial_merge_size = hf_config.vision_config.spatial_merge_size
    tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

    input_tokens_tensor = torch.tensor(input_tokens)
    vision_start_indices = torch.argwhere(
        input_tokens_tensor == vision_start_token_id
    ).squeeze(1)
    vision_tokens = input_tokens_tensor[vision_start_indices + 1]
    image_nums = (vision_tokens == image_token_id).sum()
    video_nums = (vision_tokens == video_token_id).sum()
    llm_pos_ids_list: list = []

    video_grid_ptr, image_grid_ptr = 0, 0
    video_timestamps, image_timestamps = [], []
    temporal_patch_size = hf_config.vision_config.temporal_patch_size
    
    if len(video_grid_thw) > 1 or len(image_grid_thw) > 1:
        start_indices = torch.where(input_tokens_tensor == vision_start_token_id)[0].tolist()
        seq_len = len(input_tokens)

        for start_idx in start_indices:
            curr = start_idx + 1
            if curr >= seq_len:
                continue

            current_video_accumulated_time = 0.0
            content_token = input_tokens[curr]
            if content_token == video_token_id:
                while (
                    curr < seq_len
                    and input_tokens[curr] == video_token_id
                    and video_grid_ptr < len(video_grid_thw)
                ):
                    video_timestamps.append(current_video_accumulated_time)

                    t, h, w = video_grid_thw[video_grid_ptr]
                    current_video_accumulated_time += (
                        temporal_patch_size * t * second_per_grid_ts[video_grid_ptr].item()
                    )

                    token_stride = (h * w) // (spatial_merge_size ** 2)
                    curr += token_stride
                    video_grid_ptr += 1

            elif content_token == image_token_id:
                while (
                    curr < seq_len
                    and input_tokens[curr] == image_token_id
                    and image_grid_ptr < len(image_grid_thw)
                ):
                    image_timestamps.append(current_video_accumulated_time)

                    _, h, w = image_grid_thw[image_grid_ptr]
                    current_video_accumulated_time += temporal_patch_size

                    token_stride = (h * w) // (spatial_merge_size ** 2)
                    curr += token_stride
                    image_grid_ptr += 1
                    
    if image_timestamps is not None:
        image_nums = len(image_grid_thw)
        # print("image_grid_thw", image_grid_thw)

    if video_timestamps is not None:
        video_nums = len(video_grid_thw)
        # print("second_per_grid_ts", second_per_grid_ts)
        # print("video_grid_thw", video_grid_thw)
        # print("video_timestamps", video_timestamps)

    st = 0
    remain_images, remain_videos = image_nums, video_nums

    image_index, video_index = 0, 0
    current_video_anchor_pos = None

    # print(f"[DEBUG M-RoPE] Total Images: {image_nums}, Timestamps: {image_timestamps}")

    for i in range(image_nums + video_nums):
        video_second_per_grid_t = 0.0
        current_image_ts, current_video_ts_list = None, None 
        
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1
        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1
        
        is_image_mode, is_video_mode = False, False

        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            
            if image_timestamps is not None and len(image_timestamps) > image_index:
                raw_ts = image_timestamps[image_index]
                if isinstance(raw_ts, torch.Tensor):
                    current_image_ts = raw_ts.item()
                else:
                    current_image_ts = float(raw_ts)

                if abs(current_image_ts) < 1e-6:
                    current_st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len_temp = ed_image - st
                    current_video_anchor_pos = current_st_idx + text_len_temp
                    # print(f"[DEBUG] Frame {image_index} (Start): Set Anchor = {current_video_anchor_pos}")

            image_index += 1
            remain_images -= 1
            ed = ed_image
            is_image_mode = True
        else:
            t, h, w = video_grid_thw[video_index]
            ts_source = None
            if video_timestamps and len(video_timestamps) > video_index:
                ts_source = video_timestamps[video_index]
            elif second_per_grid_ts and len(second_per_grid_ts) > video_index:
                ts_source = second_per_grid_ts[video_index]

            if ts_source is not None:
                # Normalize to Tensor
                if not isinstance(ts_source, torch.Tensor):
                    ts_source = torch.tensor(ts_source)
                if ts_source.ndim == 0:
                    ts_source = ts_source.view(1)

                current_video_ts_list = ts_source

                if len(current_video_ts_list) > 0:
                    start_time = float(current_video_ts_list.min())
                    
                    # [Anchor Logic]
                    if abs(start_time) < 1e-6:
                        # Case: New Video Start (0.0) -> Set New Anchor
                        current_st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len_temp = ed_video - st
                        current_video_anchor_pos = current_st_idx + text_len_temp
                        # print(f"[DEBUG] New Video detected (TS=0). Anchor: {current_video_anchor_pos}")
                        

            video_index += 1
            remain_videos -= 1
            ed = ed_video
            is_video_mode = True

        llm_grid_t, llm_grid_h, llm_grid_w = (
            t,
            h // spatial_merge_size,
            w // spatial_merge_size,
        )
        text_len = ed - st
        
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        )

        use_anchor_logic = False
        if current_video_anchor_pos is not None:
            if is_image_mode and current_image_ts is not None:
                use_anchor_logic = True
            elif is_video_mode and current_video_ts_list is not None:
                use_anchor_logic = True

        if use_anchor_logic:
            base_pos = current_video_anchor_pos
            
            if is_image_mode:
                t_index = torch.full_like(h_index, current_image_ts, dtype=torch.long)
            else:
                # current_video_ts_list shape: [T]
                # We need shape: [T * H * W]
                frame_spatial_size = llm_grid_h * llm_grid_w
                ts_tensor = current_video_ts_list.to(h_index.device).long()
                # Expand [t1, t2] -> [t1...t1, t2...t2]
                t_index = ts_tensor.view(-1, 1).expand(-1, frame_spatial_size).flatten()

            # print(f"[DEBUG] Using Anchor {base_pos}. Type: {'IMG' if is_image_mode else 'VID'}")
            
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + base_pos
            )
        else:
            t_vals = (torch.arange(llm_grid_t).float() * video_second_per_grid_t)
            t_index = (
                (
                    t_vals
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    * tokens_per_second
                )
                .long()
                .flatten()
            )
            
            debug_st_idx = st_idx + text_len
            # print(f"[DEBUG] Frame {image_index-1} (Standard): St_Idx {debug_st_idx}. (Why? Mode={is_image_mode}, TS={current_image_ts}, Anchor={current_video_anchor_pos})")
            
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + debug_st_idx
            )
        
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        
    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)
    return llm_positions, mrope_position_delta


class Qwen2_5_VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - ni: Number of images
        - cps: Number of channels * patch_size * patch_size

    Historical context:
        - pixel_values shape: (num_patches, num_channels * patch_size *
          patch_size)
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format.
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]

    image_timestamps: Optional[Annotated[
        torch.Tensor,
        TensorShape("ni"),
    ]] = None

class Qwen2_5_VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images

    Historical context:
        - image_embeds shape: (num_image_features, hidden_size)
        - num_image_features varies based on the number and resolution of the
          images.
        - hidden_size must match the hidden size of language model backbone.
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]

    image_timestamps: Optional[Annotated[
        torch.Tensor,
        TensorShape("ni"),
    ]] = None

def _parse_and_validate_image_input(
    self, **kwargs: object
) -> Qwen2_5_VLImageInputs | None:
    pixel_values = kwargs.pop("pixel_values", None)
    image_embeds = kwargs.pop("image_embeds", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    image_timestamps = kwargs.pop("image_timestamps", None)

    if pixel_values is None and image_embeds is None:
        return None

    if pixel_values is not None:
        return Qwen2_5_VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_timestamps=image_timestamps,
        )

    if image_embeds is not None:
        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
            image_timestamps=image_timestamps,
        )

def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    if "image_timestamps" in hf_processor_mm_kwargs:
        hf_inputs["image_timestamps"] = hf_processor_mm_kwargs["image_timestamps"]
    if "video_timestamps" in hf_processor_mm_kwargs:
        hf_inputs["video_timestamps"] = hf_processor_mm_kwargs["video_timestamps"]
    base_config = Qwen2VLMultiModalProcessor._get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs)
    # print("hf_inputs", hf_inputs)
    # print("hf_processor_mm_kwargs", hf_processor_mm_kwargs)
    return dict(
        **base_config,
        second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        image_timestamps=MultiModalFieldConfig.batched("image"),
        video_timestamps=MultiModalFieldConfig.batched("video"),
        )

def get_mrope_input_positions00(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    kwargs = MultiModalFeatureSpec.gather_kwargs(
        mm_features,
        {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"},
    )
    image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
    video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
    second_per_grid_ts = kwargs.get("second_per_grid_ts", [])

    hf_config = self.config
    image_token_id = hf_config.image_token_id
    video_token_id = hf_config.video_token_id
    vision_start_token_id = hf_config.vision_start_token_id
    spatial_merge_size = hf_config.vision_config.spatial_merge_size
    tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

    input_tokens_tensor = torch.tensor(input_tokens)
    vision_start_indices = torch.argwhere(
        input_tokens_tensor == vision_start_token_id
    ).squeeze(1)
    vision_tokens = input_tokens_tensor[vision_start_indices + 1]
    image_nums = (vision_tokens == image_token_id).sum()
    video_nums = (vision_tokens == video_token_id).sum()
    llm_pos_ids_list: list = []

    st = 0
    remain_images, remain_videos = image_nums, video_nums

    image_index, video_index = 0, 0
    for _ in range(image_nums + video_nums):
        video_second_per_grid_t = 0.0
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1
        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1
        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = video_grid_thw[video_index]
            video_second_per_grid_t = 1.0
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t, llm_grid_h, llm_grid_w = (
            t,
            h // spatial_merge_size,
            w // spatial_merge_size,
        )
        text_len = ed - st

        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

        t_index = (
            (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                * video_second_per_grid_t
                * tokens_per_second
            )
            .long()
            .flatten()
        )

        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        )
        llm_pos_ids_list.append(
            torch.stack([t_index, h_index, w_index]) + text_len + st_idx
        )
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)
    return llm_positions, mrope_position_delta



from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions
# vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration._parse_and_validate_image_input = _parse_and_validate_image_input
# vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
# vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates


from transformers import PretrainedConfig
@classmethod
def _vl_get_input_positions_tensor(
    cls,
    input_tokens: list[int],
    hf_config: PretrainedConfig,
    image_grid_thw: Union[list[list[int]], torch.Tensor],
    video_grid_thw: Union[list[list[int]], torch.Tensor],
    second_per_grid_ts: list[float],
    context_len: int = 0,
    seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    """Get mrope input positions and delta value."""

    image_token_id = hf_config.image_token_id
    video_token_id = hf_config.video_token_id
    vision_start_token_id = hf_config.vision_start_token_id
    spatial_merge_size = hf_config.vision_config.spatial_merge_size
    tokens_per_second = getattr(hf_config.vision_config,
                                "tokens_per_second", 1.0)

    input_tokens_tensor = torch.tensor(input_tokens)
    vision_start_indices = torch.argwhere(
        input_tokens_tensor == vision_start_token_id).squeeze(1)
    vision_tokens = input_tokens_tensor[vision_start_indices + 1]
    image_nums = (vision_tokens == image_token_id).sum()
    video_nums = (vision_tokens == video_token_id).sum()
    llm_pos_ids_list: list = []

    st = 0
    remain_images, remain_videos = image_nums, video_nums

    image_index, video_index = 0, 0
    for _ in range(image_nums + video_nums):
        video_second_per_grid_t = 0.0
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1
        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1
        if ed_image < ed_video:
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = (
                video_grid_thw[video_index][0],
                video_grid_thw[video_index][1],
                video_grid_thw[video_index][2],
            )
            video_second_per_grid_t = 1.0
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t, llm_grid_h, llm_grid_w = \
            t, h // spatial_merge_size, w // spatial_merge_size
        text_len = ed - st

        st_idx = llm_pos_ids_list[-1].max() + 1 if len(
            llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        t_index = (torch.arange(llm_grid_t).view(-1, 1).expand(
            -1, llm_grid_h * llm_grid_w) * video_second_per_grid_t *
                    tokens_per_second).long().flatten()

        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
            llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
            llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(
            torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(
            llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 -
                            len(input_tokens)).item()
    llm_positions = llm_positions[:, context_len:seq_len]
    print("%"*50)
    return llm_positions, mrope_position_delta

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding._vl_get_input_positions_tensor = _vl_get_input_positions_tensor

if __name__ == "__main__":
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # image.png
    # from vllm import LLM
    # from PIL import Image
    # from transformers import AutoProcessor

    # # Qwen2.5-VL example with two images
    # # llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # video_path ="/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"
    # video_messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": [
    #             {"type": "text", "text": "描述这个视频。"},
    #             {
    #                 "type": "video", 
    #                 "video": "file:///" + video_path, 
    #                 "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28  
    #             }
    #         ]
    #     },
    # ]
    # # breakpoint()
    # # processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-1.5-8B", trust_remote_code=True)
    # llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.6, dtype="bfloat16", )
    # # llm = LLM(model="Kwai-Keye/Keye-VL-1.5-8B", trust_remote_code=True, gpu_memory_utilization=0.8, dtype="bfloat16", )


    # prompt = "USER: <|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>\nWhat is the content of this image?\nASSISTANT:"
    # prompt = "USER: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>\nWhat is the content of this image?\nASSISTANT:"
    # image = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/VisionThink/scissor.png")
    # image1 = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Pretrain/images/00000/000000010.jpg")
    # from transformers.image_utils import (
    #     PILImageResampling,
    #     get_image_size,
    #     infer_channel_dimension_format,
    #     to_numpy_array,
    #     ChannelDimension,
    #     SizeDict
    # )
    # image = to_numpy_array(image)
    # image1 = to_numpy_array(image1)

    # # Single prompt inference
    # outputs = llm.generate({
    #     "prompt": prompt,
    #     "multi_modal_data": {"image": [image, image1]},
    #     "mm_processor_kwargs": {
    #         "image_timestamps": [0.0, 1.0],
    #     }
    # })

    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(generated_text)


    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    # from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions


    MODEL_PATH = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/qwen2.5_vl-3b"
    video_path ="/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"

    llm = LLM(
        model=MODEL_PATH,  
        gpu_memory_utilization=0.8,  
        tensor_parallel_size=1, 
        max_model_len=16384,  
        dtype="bfloat16", 
        mm_processor_cache_gb=0,
        # enforce_eager=True,  
        # limit_mm_per_prompt={"image": 10, "video": 10},  
    )


    sampling_params = SamplingParams(
        temperature=0.1,  
        top_p=0.001,  
        repetition_penalty=1.05,  
        max_tokens=8192,  
        stop_token_ids=[],  
    )

    video_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": "描述这个视频。"},
                {
                    "type": "video", 
                    "video": "file:///" + video_path, 
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                    "max_frames": 8,
                },
                {
                    "type": "video", 
                    "video": "file:///" + video_path, 
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                    "max_frames": 8,
                }
            ]
        },
    ]


    messages = video_messages


    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, videos, video_kwargs = process_vision_info(messages, return_video_metadata=True, return_video_kwargs=True)
    # breakpoint()
    from visionthink.adaptive.utils import tensor_to_pil_list, expand_video_prompt, tensor_to_tensor_list, expand_image_prompt, tensor_to_temporal_stack_list, split_video_metadata
    images = []
    video_inputs = [video[0] for video in videos]
    video_metadata = [video[1] for video in videos]
    
    # for video in video_inputs:
    #     images.extend(tensor_to_tensor_list(video))
    # prompt = expand_image_prompt(prompt, video_inputs)
    # video_inputs = None


    video_metadata = split_video_metadata(video_metadata, 2)
    # video_timestamps = [video['video_timestamps'] for video in video_metadata]

    # new_videos = []
    # x_videos = []
    # for video in video_inputs:
    #     new_videos.extend(tensor_to_temporal_stack_list(video, 2))
    #     x_videos.append(tensor_to_temporal_stack_list(video, 2))
    # prompt = expand_video_prompt(prompt, x_videos, 2)
    # # video_inputs = new_videos
    # video_inputs = [(video, metadata) for video, metadata in zip(new_videos, video_metadata)]

    # # breakpoint()
    # print(video_inputs[0][0].shape)
    # print(prompt)
    # print(video_timestamps)

    # if os.environ.get("REMOVEPAD", None):
    #     prompt = expand_video_prompt(prompt, video_inputs)
    # else:
    #     new_messages = []
    #     for msg in messages:
    #         new_msg = msg.copy()
    #         if msg['role'] == 'user' and isinstance(msg.get('content'), list):
    #             new_content = []
    #             for content_item in msg['content']:
    #                 if content_item.get('type') == 'video':
    #                     for image in images:
    #                         new_content.append({"type": "image", "image": image})
    #                 else:
    #                     new_content.append(content_item)
    #             new_msg['content'] = new_content
    #         new_messages.append(new_msg)

    #     messages = new_messages
    #     prompt = processor.apply_chat_template(
    #         new_messages, add_generation_prompt=True, tokenize=False
    #     )

    # breakpoint()
    # print("images", images)


    mm_data = {}
    video_timestamps_list = []
    
    temporal_merge_size = processor.video_processor.temporal_patch_size
    print("processor.temporal_patch_size", processor.video_processor.temporal_patch_size)
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    #     for i, video_tensor in enumerate(video_inputs):
    #         curr_t = video_tensor.shape[0] 
    #         start_frame_idx = i * temporal_merge_size
            
    #         timestamps = tuple(float(start_frame_idx + j) for j in range(curr_t))
    #         video_timestamps_list.append(timestamps)
    # print("video_timestamps_list", video_timestamps_list)

    if len(images) > 0:
        mm_data["image"] = images

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # "mm_processor_kwargs": {
        #     # "image_timestamps": [float(i * 2) for i in range(len(images))],
        #     # "video_timestamps": video_timestamps_list,
        #     "video_timestamps": video_timestamps,
        # }
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    # outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    # generated_text = outputs[0].outputs[0].text

    print(generated_text)
