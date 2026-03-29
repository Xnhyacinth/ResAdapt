import os
import numpy as np

import vllm.inputs.preprocess
import vllm.model_executor.models.qwen2_5_vl

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import torch
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
)

MAX_PIXELS = 14 * 14 * 4 * 16384

# from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
# def _get_mm_fields_config(
#     self,
#     hf_inputs: BatchFeature,
#     hf_processor_mm_kwargs: Mapping[str, object],
# ) -> Mapping[str, MultiModalFieldConfig]:
#     if "image_timestamps" in hf_processor_mm_kwargs:
#         hf_inputs["image_timestamps"] = hf_processor_mm_kwargs["image_timestamps"]
#     if "video_timestamps" in hf_processor_mm_kwargs:
#         hf_inputs["video_timestamps"] = hf_processor_mm_kwargs["video_timestamps"]
#     base_config = Qwen2VLMultiModalProcessor._get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs)
#     # print("hf_inputs", hf_inputs)
#     # print("hf_processor_mm_kwargs", hf_processor_mm_kwargs)
#     return dict(
#         **base_config,
#         second_per_grid_ts=MultiModalFieldConfig.batched("video"),
#         image_timestamps=MultiModalFieldConfig.batched("image"),
#         video_timestamps=MultiModalFieldConfig.batched("video"),
#         )


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

    # if os.environ.get("DEBUG_MODE") == "debug":
    # print(f"[DEBUG M-RoPE] Total Images: {image_nums}, Videos: {video_nums}")

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
    # print("llm_positions[0]", llm_positions[0].tolist())
    # print("llm_positions[1]", llm_positions[1].tolist())
    # print("llm_positions[2]", llm_positions[2].tolist())
    # print("mrope_position_delta", mrope_position_delta)
    return llm_positions, mrope_position_delta

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


def get_mrope_input_positions11(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    llm_pos_ids_list: list = []
    st = 0

    for (
        offset,
        llm_grid_t,
        llm_grid_h,
        llm_grid_w,
        t_factor,
    ) in self.iter_mm_grid_thw(mm_features):
        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )

        grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w))
        if t_factor != 1.0:
            grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
        llm_pos_ids_list.append(grid_indices.reshape(3, -1) + text_len + st_idx)
        st = offset + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)
    return torch.from_numpy(llm_positions), mrope_position_delta

def iter_mm_grid_thw22(
    self, mm_features: list[MultiModalFeatureSpec]
) -> Iterator[tuple[int, int, int, int, object]]:
    """
    Iterate over multimodal features and yield grid information.
    Updated to yield raw timestamps for Temporal Merge support.
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    temporal_patch_size = int(getattr(self.config.vision_config, "temporal_patch_size", 1))
    video_time_cursor = 0
    prev_end_offset: int | None = None
    prev_modality: str | None = None

    def _unwrap_mm_field(value, default):
        if value is None:
            return default
        if hasattr(value, "data"):
            value = value.data
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "item"):
            value = value.item()
        return value

    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        offset = mm_feature.mm_position.offset

        if mm_feature.modality == "image":
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            assert t == 1, f"Image must have 1 frame, got {t}"

            ts_elem = mm_feature.data.get("image_timestamps")
            raw_ts = _unwrap_mm_field(ts_elem, 0.0)
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            token_count = llm_h * llm_w
            prev_end_offset = offset + token_count
            prev_modality = "image"
            yield offset, 1, llm_h, llm_w, raw_ts

        elif mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            token_count = int(t) * llm_h * llm_w

            is_contiguous_video = prev_modality == "video" and prev_end_offset is not None and (
                offset == prev_end_offset or offset == prev_end_offset + 2
            )
            if not is_contiguous_video:
                video_time_cursor = 0

            ts_elem = mm_feature.data.get("video_timestamps")
            raw_ts = _unwrap_mm_field(ts_elem, None)
            if not isinstance(raw_ts, (list, tuple)) or len(raw_ts) != int(t):
                raw_ts = [video_time_cursor + i * temporal_patch_size for i in range(int(t))]
            video_time_cursor += int(t) * temporal_patch_size
            prev_end_offset = offset + token_count
            prev_modality = "video"

            yield offset, t, llm_h, llm_w, raw_ts

        else:
            raise ValueError(f"Unsupported modality: {mm_feature.modality}")

def get_mrope_input_positions22(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    llm_pos_ids_list: list = []
    st = 0

    def _next_base_pos() -> int:
        return int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0

    # Anchor position for temporal merge mode (video chunks / leading image).
    current_video_anchor_pos = None

    for (
        offset,
        llm_grid_t,
        llm_grid_h,
        llm_grid_w,
        raw_ts,
    ) in self.iter_mm_grid_thw(mm_features):
        mm_start = offset
        vision_start_token_id = getattr(self.config, "vision_start_token_id", None)
        if (
            vision_start_token_id is not None
            and mm_start < len(input_tokens)
            and input_tokens[mm_start] == int(vision_start_token_id)
        ):
            mm_start = mm_start + 1

        text_len = mm_start - st
        st_idx = _next_base_pos()
        if text_len > 0:
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        base_pos_default = st_idx + text_len - 1
        if base_pos_default < 0:
            base_pos_default = 0

        # Temporal merge is enabled when timestamps are per-frame (list/tuple) or a leading image (ts=0).
        is_temporal_list = isinstance(raw_ts, (list, tuple))
        is_image_start = isinstance(raw_ts, (int, float)) and abs(raw_ts) < 1e-6 and llm_grid_t == 1
        start_time = raw_ts[0] if is_temporal_list else raw_ts

        if abs(start_time) < 1e-6:
            current_video_anchor_pos = base_pos_default
        elif not is_temporal_list and current_video_anchor_pos is not None:
            current_video_anchor_pos = None

        grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w))

        if current_video_anchor_pos is not None and (is_temporal_list or is_image_start):
            base_pos = current_video_anchor_pos
            if is_temporal_list:
                ts_array = np.array(raw_ts, dtype=np.int64)
                ts_array = ts_array[:, None, None]
                grid_indices[0] = ts_array
            else:
                grid_indices[0] = int(start_time)

            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + base_pos)

        else:
            tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 1.0)
            t_factor = raw_ts * tokens_per_second if isinstance(raw_ts, (int, float)) else 1.0

            if t_factor != 1.0:
                grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)

            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + base_pos_default)

        st = mm_start + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = _next_base_pos()
        text_len = len(input_tokens) - st
        if text_len > 0:
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    print("llm_positions[0]", llm_positions[0].tolist())
    print("llm_positions[1]", llm_positions[1].tolist())
    print("llm_positions[2]", llm_positions[2].tolist())
    print("mrope_position_delta", mrope_position_delta)
    return torch.from_numpy(llm_positions), mrope_position_delta


merge_len = 4
def _grid_token_count(grid):
    try:
        import torch
        if torch.is_tensor(grid):
            if grid.numel() == 0:
                return 0
            prod = grid.long().prod(dim=-1)
            return int((prod // merge_len).sum().item())
    except Exception:
        pass
    if isinstance(grid, (list, tuple)):
        total = 0
        for item in grid:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    t, h, w = item[:3]
                    total += (int(t) * int(h) * int(w)) // merge_len
                except Exception:
                    continue
        return total
    return 0


if __name__ == "__main__":
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions22
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw22


    # from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions00
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw22
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration._parse_and_validate_image_input = _parse_and_validate_image_input
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates

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
    images = []
    video_inputs = [video[0] for video in videos]
    video_metadata = [video[1] for video in videos]

    # outputs = processor(text=[prompt], videos=video_inputs)['input_ids']
    # breakpoint()
    from visionthink.adaptive.utils import tensor_to_pil_list, expand_video_prompt, tensor_to_tensor_list, expand_image_prompt, tensor_to_temporal_stack_list, split_video_metadata
    
    
    # for video in video_inputs:
    #     images.extend(tensor_to_tensor_list(video))
    # prompt = expand_image_prompt(prompt, video_inputs)
    # video_inputs = None


    # video_metadata = split_video_metadata(video_metadata, 2)
    # # video_timestamps = [video['video_timestamps'] for video in video_metadata]

    # new_videos = []
    # x_videos = []
    # for video in video_inputs:
    #     new_videos.extend(tensor_to_temporal_stack_list(video, 2))
    #     x_videos.append(tensor_to_temporal_stack_list(video, 2))
    # prompt = expand_video_prompt(prompt, x_videos, 2)
    # video_inputs = new_videos
    # video_inputs = [(video, metadata) for video, metadata in zip(new_videos, video_metadata)]

    # breakpoint()
    print(video_inputs[0][0].shape)
    print(prompt)
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
