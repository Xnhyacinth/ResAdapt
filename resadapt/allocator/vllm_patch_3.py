import torch
import numpy as np
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from vllm.multimodal.inputs import MultiModalFeatureSpec

from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
    compute_retention_mask,
    recompute_mrope_positions,
)
from vllm.utils.collection_utils import is_list_of

def _get_prompt_updates(
    self,
    mm_items,
    hf_processor_mm_kwargs,
    out_mm_kwargs,
):
    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    tokenizer = self.info.get_tokenizer()
    hf_config = self.info.get_hf_config()

    video_token_id = hf_config.video_token_id
    vision_start_token_id = hf_config.vision_start_token_id
    vision_end_token_id = hf_config.vision_end_token_id

    merge_length = image_processor.merge_size**2

    def get_image_replacement_qwen3vl(item_idx: int):
        out_item = out_mm_kwargs["image"][item_idx]
        grid_thw = out_item["image_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        num_tokens = int(grid_thw.prod()) // merge_length
        return [hf_processor.image_token_id] * num_tokens

    def get_video_replacement_qwen3vl(item_idx: int):
        out_item = out_mm_kwargs["video"][item_idx]
        grid_thw = out_item["video_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        video, metadata = mm_items["video"][item_idx]
        do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
        sampled_fps = hf_processor_mm_kwargs.get("fps")
        if is_list_of(sampled_fps, float):
            sampled_fps = sampled_fps[item_idx]
        timestamps = self.info._get_video_second_idx(
            metadata, out_item, do_sample_frames, sampled_fps
        )

        assert len(timestamps) == grid_thw[0], (
            f"The timestamps length({len(timestamps)}) should be equal "
            f"video length ({grid_thw[0]})."
        )

        frames_idx_token = [
            tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False)
            for curr_time in timestamps
        ]
        tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
        per_frame_token_counts = [tokens_per_frame for _ in frames_idx_token]

        video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
        if video_pruning_rate is not None and video_pruning_rate > 0.0:
            total_retained = compute_retained_tokens_count(
                tokens_per_frame,
                len(frames_idx_token),
                video_pruning_rate,
            )
            if len(frames_idx_token) == 0:
                per_frame_token_counts = []
            elif len(frames_idx_token) == 1:
                per_frame_token_counts = [tokens_per_frame]
            else:
                first_frame_tokens = tokens_per_frame
                remaining_tokens = max(total_retained - first_frame_tokens, 0)
                base = remaining_tokens // (len(frames_idx_token) - 1)
                remainder = remaining_tokens % (len(frames_idx_token) - 1)
                per_frame_token_counts = [first_frame_tokens]
                for frame_idx in range(1, len(frames_idx_token)):
                    extra = base + (1 if (frame_idx - 1) < remainder else 0)
                    per_frame_token_counts.append(extra)

        placeholder = []
        for frame_idx, timestamp_tokens in enumerate(frames_idx_token):
            placeholder.extend(timestamp_tokens)
            tokens_this_frame = per_frame_token_counts[
                frame_idx if frame_idx < len(per_frame_token_counts) else -1
            ]
            placeholder.extend(
                [vision_start_token_id]
                + [video_token_id] * tokens_this_frame
                + [vision_end_token_id]
            )
        return PromptUpdateDetails.select_token_id(placeholder, video_token_id)

    return [
        PromptReplacement(
            modality="image",
            target=hf_processor.image_token,
            replacement=get_image_replacement_qwen3vl,
        ),
        PromptReplacement(
            modality="video",
            target="<|vision_start|><|video_pad|><|vision_end|>",
            replacement=get_video_replacement_qwen3vl,
        ),
    ]


def iter_mm_grid_hw(
    self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
) -> Iterator[tuple[int, int, int]]:
    """
    Iterate over multimodal features and yield grid information.

    For videos with EVS (Efficient Video Sampling) enabled, this function
    computes the offset based on the pruned token count rather than relying
    on input_tokens.index(), which would fail when tokens are pruned.

    Args:
        input_tokens: List of token IDs in the prompt
        mm_features: List of multimodal feature specifications

    Yields:
        Tuple of (offset, grid_h, grid_w) for each frame/image
    """
    video_token_id = self.config.video_token_id
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        offset = mm_feature.mm_position.offset
        if mm_feature.modality == "image":
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            assert t == 1, f"Image must have 1 frame, got {t}"
            yield offset, h // spatial_merge_size, w // spatial_merge_size
        elif mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            # Check if EVS (Efficient Video Sampling) is enabled
            is_evs_enabled = (
                hasattr(self, "video_pruning_rate")
                and self.video_pruning_rate is not None
                and self.video_pruning_rate > 0.0
            )

            if is_evs_enabled:
                frame_offsets = self._extract_frame_offsets_from_mask(
                    mm_feature.mm_position, t
                )
                if frame_offsets is not None:
                    for rel_offset in frame_offsets:
                        yield offset + rel_offset, llm_grid_h, llm_grid_w
                    continue

                # If EVS is enabled but mask is missing, this indicates a bug
                # in the prompt processing pipeline. The is_embed mask should
                # always be present when video_pruning_rate > 0.
                raise RuntimeError(
                    f"EVS is enabled (pruning_rate={self.video_pruning_rate}) "
                    "but is_embed mask is missing from mm_position. "
                    "This indicates a bug in prompt processing."
                )
            else:
                # Non-EVS mode: Use original logic with input_tokens.index()
                for _ in range(t):
                    offset = input_tokens.index(video_token_id, offset)
                    yield offset, llm_grid_h, llm_grid_w
                    offset += llm_grid_h * llm_grid_w
        else:
            raise ValueError(f"Unsupported modality: {mm_feature.modality}")


# def expand_video_prompt_blocks(prompt, video_inputs):
#     video_block = "<|vision_start|><|video_pad|><|vision_end|>"
#     parts = prompt.split(video_block)
#     if len(parts) - 1 != len(video_inputs):
#         raise ValueError(
#             f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided."
#         )
#     new_prompt = parts[0]
#     for i, video in enumerate(video_inputs):
#         count_n = len(video) if isinstance(video, list) else 1
#         new_prompt += (video_block * count_n) + parts[i + 1]
#     return new_prompt


# def _get_actual_frame_token_counts(
#         self, mm_position: PlaceholderRange, expected_frames: int
#     ) -> list[int] | None:
#         """Return actual token count for each EVS-retained frame.

#         This function calculates the actual number of tokens per frame by
#         analyzing the is_embed mask, accounting for EVS pruning. Each frame
#         may have a different token count due to content-aware pruning.

#         Args:
#             mm_position: MultiModal position containing the is_embed mask
#             expected_frames: Expected number of frames

#         Returns:
#             List of token counts for each frame, or None if EVS is not enabled.
#         """
#         segments = self._get_evs_mask_segments(mm_position, expected_frames)
#         if segments is None:
#             return None

#         return [len(seg) for seg in segments]

def get_mrope_input_positions00(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    # Pre-collect actual frame token counts for EVS mode
    frame_token_counts_map = {}
    for mm_feature in mm_features:
        if mm_feature.modality == "video":
            is_evs_enabled = (
                hasattr(self, "video_pruning_rate")
                and self.video_pruning_rate is not None
                and self.video_pruning_rate > 0.0
            )
            if is_evs_enabled:
                t = mm_feature.data["video_grid_thw"].data.tolist()[0]
                token_counts = self._get_actual_frame_token_counts(
                    mm_feature.mm_position, t
                )
                assert token_counts is not None, (
                    "EVS enabled but failed to extract frame token counts "
                    "from is_embed mask"
                )
                frame_token_counts_map[mm_feature.mm_position.offset] = token_counts

    llm_pos_ids_list = []
    st = 0
    frame_counts_idx = {}

    for offset, llm_grid_h, llm_grid_w in self.iter_mm_grid_hw(
        input_tokens, mm_features
    ):
        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

        # Determine actual token count for this frame
        base_offset = None
        for feat_offset in frame_token_counts_map:
            if offset >= feat_offset:
                base_offset = feat_offset

        if base_offset is not None:
            # EVS mode: use actual token count from is_embed mask
            assert base_offset in frame_token_counts_map, (
                f"Found base_offset {base_offset} but not in frame_token_counts_map"
            )

            if base_offset not in frame_counts_idx:
                frame_counts_idx[base_offset] = 0

            counts = frame_token_counts_map[base_offset]
            idx = frame_counts_idx[base_offset]

            assert idx < len(counts), (
                f"EVS frame index {idx} out of range (total frames: {len(counts)})"
            )

            actual_frame_tokens = counts[idx]
            frame_counts_idx[base_offset] += 1
        else:
            # Non-EVS mode (or image): use theoretical grid size
            actual_frame_tokens = llm_grid_h * llm_grid_w

        # Add text segment
        text_positions = (
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )
        llm_pos_ids_list.append(text_positions)
        st_idx += text_len

        # Add frame segment with actual token count (not theoretical)
        grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
        # Only take the first actual_frame_tokens positions
        frame_positions = grid_indices[:, :actual_frame_tokens] + st_idx
        llm_pos_ids_list.append(frame_positions)

        # Update st using actual token count
        st = offset + actual_frame_tokens

    # Handle final text segment
    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        final_text_positions = (
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )
        llm_pos_ids_list.append(final_text_positions)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)

    return torch.from_numpy(llm_positions), mrope_position_delta

def get_mrope_input_positions(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    # Pre-collect actual frame token counts for EVS mode
    frame_token_counts_map = {}
    feature_map = {}
    for mm_feature in mm_features:
        feature_map[mm_feature.mm_position.offset] = mm_feature
        if mm_feature.modality == "video":
            is_evs_enabled = (
                hasattr(self, "video_pruning_rate")
                and self.video_pruning_rate is not None
                and self.video_pruning_rate > 0.0
            )
            if is_evs_enabled:
                t = mm_feature.data["video_grid_thw"].data.tolist()[0]
                token_counts = self._get_actual_frame_token_counts(
                    mm_feature.mm_position, t
                )
                assert token_counts is not None, (
                    "EVS enabled but failed to extract frame token counts "
                    "from is_embed mask"
                )
                frame_token_counts_map[mm_feature.mm_position.offset] = token_counts

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    temporal_patch_size = getattr(self.config.vision_config, "temporal_patch_size", 1)
    default_video_ts_map = {}
    video_grid_t_map = {}
    video_time_cursor = 0
    prev_video_end_offset = None
    prev_modality = None
    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        if mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            token_count = int(t) * llm_h * llm_w
            offset = mm_feature.mm_position.offset
            is_contiguous_video = (
                prev_modality == "video"
                and prev_video_end_offset is not None
                and (offset == prev_video_end_offset or offset == prev_video_end_offset + 2)
            )
            if not is_contiguous_video:
                video_time_cursor = 0
            default_video_ts_map[offset] = [
                video_time_cursor + i * temporal_patch_size for i in range(int(t))
            ]
            video_grid_t_map[offset] = int(t)
            video_time_cursor += int(t) * temporal_patch_size
            prev_video_end_offset = offset + token_count
            prev_modality = "video"
        else:
            prev_modality = mm_feature.modality

    llm_pos_ids_list = []
    st = 0
    frame_counts_idx = {}
    
    # New structures for tracking video timestamps and splits
    feat_offsets = sorted(list(feature_map.keys()))
    local_frame_idx_map = {}
    current_video_last_max_pos = None

    for offset, llm_grid_h, llm_grid_w in self.iter_mm_grid_hw(
        input_tokens, mm_features
    ):
        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

        # Determine features actual token count for this frame
        base_offset = None
        for fo in feat_offsets:
            if offset >= fo:
                base_offset = fo
            else:
                break

        if base_offset is not None:
            # EVS mode: use actual token count from is_embed mask
            if base_offset in frame_token_counts_map:
                if base_offset not in frame_counts_idx:
                    frame_counts_idx[base_offset] = 0

                counts = frame_token_counts_map[base_offset]
                idx = frame_counts_idx[base_offset]

                assert idx < len(counts), (
                    f"EVS frame index {idx} out of range (total frames: {len(counts)})"
                )

                actual_frame_tokens = counts[idx]
                frame_counts_idx[base_offset] += 1
            else:
                actual_frame_tokens = llm_grid_h * llm_grid_w
        else:
            actual_frame_tokens = llm_grid_h * llm_grid_w

        # Add text segment (before we freeze the anchor, let the text consume positions normally)
        if text_len > 0:
            text_positions = (
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )
            llm_pos_ids_list.append(text_positions)
            st_idx += text_len

        # --- Anchor Logics Start ---
        raw_ts = None
        is_temporal_list = False
        mm_feature = feature_map.get(base_offset)
        local_frame_idx = None
        ts_elem = None

        if mm_feature is not None:
            if base_offset not in local_frame_idx_map:
                local_frame_idx_map[base_offset] = 0
            local_frame_idx = local_frame_idx_map[base_offset]

            if mm_feature.modality == "video":
                ts_elem = mm_feature.data.get("video_timestamps")
            elif mm_feature.modality == "image":
                ts_elem = mm_feature.data.get("image_timestamps")

            # Unwrap generic tensor/numpy values
            if hasattr(ts_elem, "data"):
                ts_elem = ts_elem.data
            if hasattr(ts_elem, "tolist"):
                ts_elem = ts_elem.tolist()
            elif hasattr(ts_elem, "item"):
                ts_elem = ts_elem.item()

            if mm_feature.modality == "video":
                default_ts = default_video_ts_map.get(base_offset)
                expected_t = video_grid_t_map.get(base_offset)
                if (
                    not isinstance(ts_elem, (list, tuple))
                    or (expected_t is not None and len(ts_elem) != expected_t)
                ):
                    ts_elem = default_ts

            is_temporal_list = isinstance(ts_elem, (list, tuple))
            if is_temporal_list:
                if local_frame_idx < len(ts_elem):
                    raw_ts = ts_elem[local_frame_idx]
                else:
                    raw_ts = ts_elem[-1]
            else:
                raw_ts = ts_elem

            local_frame_idx_map[base_offset] += 1

        is_video_mode = mm_feature is not None and mm_feature.modality == "video"
        is_video_start = False

        if raw_ts is not None:
            # Detect video beginnings using actual timestamp logic
            if is_temporal_list and local_frame_idx == 0:
                if abs(ts_elem[0]) < 1e-6:
                    is_video_start = True
            elif isinstance(raw_ts, (int, float)) and abs(raw_ts) < 1e-6:
                is_video_start = True

            # Logic Override: Force 'st_idx'
            if is_video_start:
                # Video Group naturally starts its own base pos, clear any preceding anchor
                current_video_last_max_pos = None  
            elif current_video_last_max_pos is not None and (is_temporal_list or is_video_mode):
                # We are continuing a video but we encountered text/splits! Override 'st_idx'.
                st_idx = current_video_last_max_pos + 1
        else:
            current_video_last_max_pos = None
        # --- Anchor Logics End ---

        # Add frame segment with actual token count (not theoretical)
        grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
        # Only take the first actual_frame_tokens positions (since it's Qwen3VL `t` is always 1 natively per loop here)
        frame_positions = grid_indices[:, :actual_frame_tokens] + st_idx
        llm_pos_ids_list.append(frame_positions)

        # Update the contiguous anchor for if the next split frame belongs to this session
        if raw_ts is not None and (is_temporal_list or is_video_mode):
            current_video_last_max_pos = int(frame_positions.max().item())

        # Update absolute string token pointer using actual offset
        st = offset + actual_frame_tokens

    # Handle final text segment
    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        final_text_positions = (
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )
        llm_pos_ids_list.append(final_text_positions)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)

    return torch.from_numpy(llm_positions), mrope_position_delta

def expand_video_prompt_blocks(prompt, video_inputs, video_metadatas=None, ts_key="video_timestamps"):
    video_block = "<|vision_start|><|video_pad|><|vision_end|>"
    if video_metadatas and isinstance(video_metadatas[0], dict) and ts_key in video_metadatas[0]:
        from resadapt.utils.utils import group_videos_by_timestamps
        video_inputs = group_videos_by_timestamps(
            video_inputs,
            video_metadatas,
            ts_key=ts_key,
            reset_value=0,
        )
    parts = prompt.split(video_block)
    if len(parts) - 1 != len(video_inputs):
        raise ValueError(
            f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided."
        )
    new_prompt = parts[0]
    for i, video in enumerate(video_inputs):
        count_n = len(video) if isinstance(video, list) else 1
        new_prompt += (video_block * count_n) + parts[i + 1]
    return new_prompt

import vllm.model_executor.models.qwen3_vl
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions00
# vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.iter_mm_grid_hw = iter_mm_grid_hw
# vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates

if __name__ == "__main__":
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions22
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw22


    # from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    # import vllm.model_executor.models.qwen3_vl
    # from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    # vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions00
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw22
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration._parse_and_validate_image_input = _parse_and_validate_image_input
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates

    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # 官方示例里有

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
    video_path = "YOUR_WORKSPACE_PATH/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": "描述这个视频。"},
                {"type": "text", "text": "pls describe this video in details"},
                {
                    "type": "video",
                    "video": f"file://{video_path}",   # ✅ 标准 file URI
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                    "max_frames": 8,
                },
                # {
                #     "type": "video",
                #     "video": f"file://{video_path}",   # ✅ 标准 file URI
                #     "total_pixels": 20480 * 28 * 28,
                #     "min_pixels": 16 * 28 * 28,
                #     "max_frames": 8,
                # },
            ],
        },
    ]

    # ✅ 建议加 trust_remote_code
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ✅ 先生成带多模态占位符的 prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ✅ Qwen3-VL 官方示例关键点：传 image_patch_size
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_metadata=True,
        return_video_kwargs=True,
    )

    from resadapt.utils.utils import (
        split_video_metadata,
        tensor_to_temporal_stack_list,
    )

    temporal_patch_size = 2
    if video_inputs is not None:
        video_metadata = None
        if (
            isinstance(video_inputs, list)
            and len(video_inputs) > 0
            and isinstance(video_inputs[0], (list, tuple))
            and len(video_inputs[0]) == 2
        ):
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata = list(video_metadata)
        elif isinstance(video_kwargs, dict) and "video_metadata" in video_kwargs:
            video_metadata = video_kwargs["video_metadata"]

        new_videos = []
        x_videos = []
        for video in video_inputs:
            chunks = tensor_to_temporal_stack_list(video, temporal_patch_size)
            new_videos.extend(chunks)
            x_videos.append(chunks)


        if video_metadata is not None:
            video_metadata = split_video_metadata(
                video_metadata, temporal_patch_size
            )

        print("PROMPT:", repr(prompt))
        prompt = expand_video_prompt_blocks(prompt, x_videos, video_metadata)
        # if isinstance(video_kwargs, dict) and "video_metadata" in video_kwargs:
        #     video_kwargs = dict(video_kwargs)
        #     video_kwargs["video_metadata"] = video_metadata
        if video_metadata is not None and len(video_metadata) == len(new_videos):
            video_inputs = list(zip(new_videos, video_metadata))
        else:
            video_inputs = new_videos
    # breakpoint()
    # prompt = expand_video_prompt(prompt, x_videos, 2)

    print("PROMPT:", repr(prompt))  # ✅ 先检查里面是否有视觉占位符

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,      # ✅ 官方示例里有
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        max_model_len=16384,
        dtype="bfloat16",
        mm_processor_cache_gb=0,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=2048,   # 先缩短，便于调试
        stop_token_ids=[],
    )

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    for o in outputs:
        print(o.outputs[0].text)
