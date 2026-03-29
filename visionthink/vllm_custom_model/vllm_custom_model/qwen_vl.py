import torch.nn as nn
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config import VllmConfig

from transformers import AutoProcessor, AutoTokenizer


class MyQwen2_5_VLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        if "image_timestamps" in hf_processor_mm_kwargs:
            hf_inputs["image_timestamps"] = hf_processor_mm_kwargs["image_timestamps"]
        if "video_timestamps" in hf_processor_mm_kwargs:
            hf_inputs["video_timestamps"] = hf_processor_mm_kwargs["video_timestamps"]
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
            image_timestamps=MultiModalFieldConfig.batched("image"),
            video_timestamps=MultiModalFieldConfig.batched("video"),
            )
@MULTIMODAL_REGISTRY.register_processor(
    MyQwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)

class MyQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    vLLM implementation of UniQwen with adaptive multi-modal scaling.
    """
    
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
        image_timestamps = kwargs.get("image_timestamps", None)
        video_timestamps = kwargs.get("video_timestamps", None)

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

        if image_timestamps is not None:
            image_nums = len(image_grid_thw)
            # print("image_grid_thw", image_grid_thw)

        if video_timestamps is not None:
            video_nums = len(video_grid_thw)
            print("video_grid_thw", video_grid_thw)
            print("video_timestamps", video_timestamps)

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
                if video_timestamps and len(video_timestamps) > video_index:
                    ts_data = video_timestamps[video_index]
                    if not isinstance(ts_data, torch.Tensor):
                        ts_data = torch.tensor(ts_data)
                    if ts_data.ndim == 0:
                        ts_data = ts_data.view(1)

                    current_video_ts_list = ts_data

                    if len(current_video_ts_list) > 0:
                        start_time = float(current_video_ts_list.min())
                        
                        # [Anchor Check] Video Chunk Start
                        if abs(start_time) < 1e-6:
                            current_st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            text_len_temp = ed_video - st
                            current_video_anchor_pos = current_st_idx + text_len_temp
                            # print(f"[DEBUG] Video Anchor Set: {current_video_anchor_pos}")
                        elif current_video_anchor_pos is None:
                            current_video_anchor_pos = None

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

                print(f"[DEBUG] Using Anchor {base_pos}. Type: {'IMG' if is_image_mode else 'VID'}")
                
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
                print(f"[DEBUG] Frame {image_index-1} (Standard): St_Idx {debug_st_idx}. (Why? Mode={is_image_mode}, TS={current_image_ts}, Anchor={current_video_anchor_pos})")
                
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