import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/"

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info

from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)

from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2RMSNorm,
)

from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor


from visionthink.adaptive.utils import scale_messages_from_mmdata, apply_adaptive_scaling
from visionthink.predictor.configuration_predictor import PredictorConfig

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

class Qwen2_5_VLPatchReshape(nn.Module):
    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).contiguous().view(-1, self.hidden_size)
        return x
    
class UniQwenVisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config: PredictorConfig):
        config.depth = config.tower_depth
        super().__init__(config)
        self.merger = Qwen2_5_VLPatchReshape(config.hidden_size, config.spatial_merge_size)
        self.hidden_size = config.hidden_size

@dataclass
class UniQwenOutputWithScale(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Extends Qwen2_5_VLCausalLMOutputWithPast to include image loss and transformed features.

    Args:
        image_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed for the image feature regression task.
        transformed_features (`torch.FloatTensor` of shape `(batch_size, vision_hidden_size)`, *optional*):
            Features transformed by the regression head.
    """
    scales: Optional[torch.FloatTensor] = None
    log_probs: Optional[torch.FloatTensor] = None
    
class PredictorForConditionalGeneration(PreTrainedModel):
    """
    
    Attributes:
        predictor (nn.Linear): A linear layer that projects features from text hidden dimension
                                   to vision hidden dimension, enabling cross-modal transformation.
    """

    config: PredictorConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLVisionBlock"] # "CrossModalMatcher", "DualPathEncoder", "FrameInformationEncoder", "ContrastiveDifferentiator", 
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config: PredictorConfig):
        """
        Initialize UniQwen with a vision tower and a scale predictor.
        Ensures consistent defaults and dtype alignment across submodules.
        """
        super().__init__(config)

        # Vision config defaults
        # vc = config.vision_config
        vc = config
        vc.min_scale = getattr(vc, "min_scale", 0.25)
        vc.max_scale = getattr(vc, "max_scale", 3.0)
        vc.cross_depth = getattr(vc, "cross_depth", 8)
        vc.self_depth = getattr(vc, "self_depth", 8)
        vc.tower_depth = getattr(vc, "tower_depth", 8)
        vc.use_text_encoder = getattr(vc, "use_text_encoder", False)
        vc.text_encoder_depth = getattr(vc, "text_encoder_depth", 0)
        vc.text_encoder_heads = getattr(vc, "text_encoder_heads", None)
        vc.text_encoder_dropout = getattr(vc, "text_encoder_dropout", 0.0)
        vc.use_cross_attn_gate = getattr(vc, "use_cross_attn_gate", False)
        vc._attn_implementation = "flash_attention_2"

        print(f"set predictor min_scale: {vc.min_scale}")
        print(f"set predictor max_scale: {vc.max_scale}")

        # Initialize vision tower and predictor
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.llm_hidden_size, padding_idx=config.pad_token_id)
        self.vision_tower = UniQwenVisionTransformer(vc)
        
        # Select predictor based on version config
        predictor_version = getattr(vc, "predictor_version", "v1")
        if predictor_version == "v2":
            from visionthink.predictor.AZNetv2 import RegressionHeadPredictorV2
            self.predictor = RegressionHeadPredictorV2(vc)
            print(f"Using V2 predictor (DifferentiableImportancePredictor)")
        else:
            from visionthink.predictor.AZNetv2 import RegressionHeadPredictor
            self.predictor = RegressionHeadPredictor(vc)
            print(f"Using V1 predictor (FrameWiseScalePredictor)")

        # Processor/tokenizer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # Token indices
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

        # Align submodules dtype to base model dtype
        model_dtype = next(self.parameters()).dtype
        self.vision_tower.to(dtype=model_dtype)
        self.predictor.to(dtype=model_dtype)

        self.post_init()

    def _init_weights(self, module: nn.Module):
        """Initialize module weights following Transformer convention."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def get_vit_features(self, pixel_values, image_grid_thw, normalize=True):
        vision_outputs = self.vision_tower(pixel_values, image_grid_thw)
        
        if pixel_values.ndim == 2:
            vision_outputs = vision_outputs.reshape(-1, self.config.vision_config.hidden_size)
        elif pixel_values.ndim == 3:
            batch_size = pixel_values.shape[0]
            vision_outputs = vision_outputs.reshape(batch_size, -1, self.config.vision_config.hidden_size)
        return vision_outputs


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Get default dtype (bfloat16) and device map from kwargs
        dtype = kwargs.get("dtype", torch.bfloat16)
        device_map = kwargs.get("device_map", None)

        # Determine if we need to clone weights from official full Qwen2.5-VL models
        need_clone = any(x in str(pretrained_model_name_or_path) for x in ["Qwen2.5-VL-7B", "Qwen2.5-VL-3B"])

        if need_clone:
            print(f"[Clone] Extracting core weights from official model: {pretrained_model_name_or_path}")
            
            # 1. Load the full official Qwen2.5-VL model
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path
            )
            base_sd = base_model.state_dict()
            
            # 2. Build new state dict with key mapping for target model architecture
            new_sd = OrderedDict()
            
            # Define weight key mapping rules
            # - Map ViT components from official model to vision tower in custom model
            # - Map embedding layer with consistent naming
            MAPPING = {
                "model.visual.": "vision_tower.",      # ViT backbone components
                "model.language_model.embed_tokens.": "predictor.embed_tokens." # Token embedding layer
            }
            # Exclude the unused projector MLP in visual merger module
            EXCLUDE = "model.visual.merger.mlp"        

            # Iterate through all weights in the official model
            for k, v in base_sd.items():
                matched = False
                # Apply mapping rules and skip excluded layers
                for old_p, new_p in MAPPING.items():
                    if k.startswith(old_p) and not k.startswith(EXCLUDE):
                        new_key = k.replace(old_p, new_p)
                        new_sd[new_key] = v
                        matched = True
                        break
            
            # 3. Instantiate custom model with config from official model
            # Note: cls(base_model.config) initializes only the 3 target layers (ViT/Embedding/Predictor)
            # config = base_model.config
            # breakpoint()
            config = kwargs['config']
            model = cls(config).to(dtype)
            
            # 4. Load filtered weights into custom model
            # strict=False: Predictor layer weights are not present in official model (expected missing)
            msg = model.load_state_dict(new_sd, strict=False)
            print(f"[Init] Weight cloning completed. Missing keys (expected for Predictor): {len(msg.missing_keys)}, Unexpected keys (should be 0): {len(msg.unexpected_keys)}")
            # breakpoint()
            # Release GPU memory by deleting base model
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return model

        # If loading from a pre-cloned UniQwen format checkpoint, use parent class implementation
        print(f"[Normal] Loading model from '{pretrained_model_name_or_path}' ...")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size, return_video_metadata=True)
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            # prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()

            # if remove_system_prompt:
            #     prompt_ids = prompt_ids[len(self.system_prompt) :]

            return model_inputs
        else:
            prompt_ids =  self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),

            if remove_system_prompt:
                prompt_ids = prompt_ids[len(self.system_prompt) :]

            return prompt_ids

    def scale_multi_modal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        multi_modal_data=None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        messages: list[dict] = None,
        actions: Optional[torch.Tensor] = None,
        scale_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Multi-modal adaptive scaling with RL (e.g., GRPO).
        Now supports both Images and Videos.
        """
        eval_mode = kwargs.get("eval_mode", None)

        # Defaults when images/videos are absent
        scaled_multi_modal_data = multi_modal_data
        new_actions, scales, log_probs, new_scale_mask = None, None, None, None

        patch_size = self.config.patch_size
        temporal_patch_size = self.config.temporal_patch_size
        merge_size = self.config.spatial_merge_size
        image_factor = patch_size * merge_size
        merge_len = merge_size ** 2
        vision_start_token_id = self.config.vision_start_token_id

        if input_ids is None and messages is None:
            raise ValueError("Either input_ids or messages must be provided.")

        if messages is not None:
            def _validate_tensor_list(tensors):
                if not tensors:
                    return False
                ref = tensors[0]
                for t in tensors[1:]:
                    if t.shape[1:] != ref.shape[1:] or t.dtype != ref.dtype:
                        return False
                return True

            def _validate_grid_list(grids):
                if not grids:
                    return False
                for g in grids:
                    if g.ndim != 2 or g.shape[1] != 3:
                        return False
                    if (g <= 0).any():
                        return False
                return True

            def _process_wrapper(msg):
                if isinstance(msg, tuple):
                    mm_data = msg[1]
                    msg = msg[0]
                else:
                    mm_data = self.process_vision_info(msg)
                imgs = mm_data.get("images")
                vids = mm_data.get("videos")

                if kwargs.get("video2image", False) and vids is not None:
                    from visionthink.adaptive.utils import video2images
                    msg, imgs = video2images(msg, vids, imgs)
                    vids = None
                
                model_inputs = self.apply_chat_template(
                    msg,
                    images=imgs,
                    videos=vids,
                )

                return {
                    "model_inputs": model_inputs,
                    "images": imgs,
                    "videos": vids,
                    "message": msg
                }

            # Execute in parallel
            with ThreadPoolExecutor(max_workers=min(16, len(messages))) as executor:
                results = list(executor.map(_process_wrapper, messages))

            messages = []
            multi_modal_data = [] 
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []
            pixel_values_videos_list = []
            video_grid_thw_list = []
            
            for res in results:
                inputs = res["model_inputs"]
                
                input_ids_list.append(inputs["input_ids"].squeeze(0))
                attention_mask_list.append(inputs["attention_mask"].squeeze(0))
                
                if "pixel_values" in inputs:
                    pixel_values_list.append(inputs["pixel_values"])
                if "image_grid_thw" in inputs:
                    image_grid_thw_list.append(inputs["image_grid_thw"])
                    
                if "pixel_values_videos" in inputs:
                    pixel_values_videos_list.append(inputs["pixel_values_videos"])
                if "video_grid_thw" in inputs:
                    video_grid_thw_list.append(inputs["video_grid_thw"])
                
                multi_modal_data.append({
                    "images": res["images"],
                    "videos": res["videos"]
                })

                messages.append(res["message"])

            drop_images = False
            drop_videos = False
            if pixel_values_list:
                if not _validate_tensor_list(pixel_values_list):
                    drop_images = True
                if not _validate_grid_list(image_grid_thw_list):
                    drop_images = True
            if pixel_values_videos_list:
                if len(pixel_values_videos_list) != len(video_grid_thw_list):
                    drop_videos = True
                if not _validate_tensor_list(pixel_values_videos_list):
                    drop_videos = True
                if not _validate_grid_list(video_grid_thw_list):
                    drop_videos = True

            if drop_images:
                pixel_values_list = []
                image_grid_thw_list = []
            if drop_videos:
                pixel_values_videos_list = []
                video_grid_thw_list = []

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(self.device)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(self.device)
            
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0).to(self.device)
                image_grid_thw = torch.cat(image_grid_thw_list, dim=0).to(self.device)
            else:
                pixel_values = None
                image_grid_thw = None
                
            if pixel_values_videos_list:
                pixel_values_videos = torch.cat(pixel_values_videos_list, dim=0).to(self.device)
                video_grid_thw = torch.cat(video_grid_thw_list, dim=0).to(self.device)
            else:
                pixel_values_videos = None
                video_grid_thw = None
        else:
            drop_images = False
            drop_videos = False
            if pixel_values is not None:
                if image_grid_thw is None or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
                    drop_images = True
                elif (image_grid_thw <= 0).any():
                    drop_images = True
                elif pixel_values.numel() == 0:
                    drop_images = True
            if pixel_values_videos is not None:
                if video_grid_thw is None or video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
                    drop_videos = True
                elif (video_grid_thw <= 0).any():
                    drop_videos = True
                elif pixel_values_videos.numel() == 0:
                    drop_videos = True
            if drop_images:
                pixel_values = None
                image_grid_thw = None
            if drop_videos:
                pixel_values_videos = None
                video_grid_thw = None
        if hasattr(multi_modal_data, "tolist"):
            multi_modal_data = multi_modal_data.tolist()


        if messages is not None:
            if drop_images:
                for mm_data in multi_modal_data:
                    if isinstance(mm_data, dict) and "images" in mm_data:
                        mm_data["images"] = None
            if drop_videos:
                for mm_data in multi_modal_data:
                    if isinstance(mm_data, dict) and "videos" in mm_data:
                        mm_data["videos"] = None
        else:
            if drop_images or drop_videos:
                if isinstance(multi_modal_data, list):
                    for mm_data in multi_modal_data:
                        if isinstance(mm_data, dict):
                            if drop_images and "images" in mm_data:
                                mm_data["images"] = None
                            if drop_videos and "videos" in mm_data:
                                mm_data["videos"] = None
        all_image_embeds = [] # Queue for image features
        all_image_grids = []  # Queue for image grids
        
        
        all_video_embeds = [] # Queue for video features
        all_video_grids = []  # Queue for video grids

        if pixel_values is not None:
            image_feat_raw = self.vision_tower(pixel_values, image_grid_thw)
            # Split into individual images (image_grid_thw is [Total_Images, 3])
            split_sizes = (image_grid_thw.prod(-1) // merge_len).tolist()
            all_image_embeds = list(torch.split(image_feat_raw, split_sizes))
            all_image_grids = list(image_grid_thw)

        if pixel_values_videos is not None:
            video_feat_raw = self.vision_tower(pixel_values_videos, video_grid_thw)
            # Split into individual videos
            split_sizes = (video_grid_thw.prod(-1) // merge_len).tolist()
            all_video_embeds = list(torch.split(video_feat_raw, split_sizes))
            all_video_grids = list(video_grid_thw)

        unified_visual_embeds = [] # List of list of tensors (one list per sample)
        unified_visual_grids = []
        visual_per_sample = [] # How many visual objects in this sample
        
        # Pointers to pop from the queues
        img_ptr = 0
        vid_ptr = 0
        
        for mm_data in multi_modal_data:
            sample_embeds = []
            sample_grids = []
            count = 0
            
            if "images" in mm_data and mm_data["images"] is not None:
                num_imgs = len(mm_data["images"])
                # Extract the next N images from the global queue
                sample_embeds.extend(all_image_embeds[img_ptr : img_ptr + num_imgs])
                sample_grids.extend(all_image_grids[img_ptr : img_ptr + num_imgs])
                img_ptr += num_imgs
                count += num_imgs

            if "videos" in mm_data and mm_data["videos"] is not None:
                num_vids = len(mm_data["videos"])
                # Extract the next N videos from the global queue
                # if num_vids > 0 and vid_ptr < len(all_video_embeds):
                #     v = all_video_embeds[vid_ptr]
                #     if v is not None and v.shape[0] > 1:
                #         t = int(video_grid_thw[vid_ptr][0].item()) if video_grid_thw is not None else 0
                #         if t > 1:
                #             frame_tokens = v[: t * (v.shape[0] // t)].reshape(t, -1, v.shape[-1]).mean(dim=1)
                #             sim = torch.nn.functional.cosine_similarity(frame_tokens[1:], frame_tokens[:-1], dim=-1).mean().item()
                #             print(f"[init_frame_sim] mean={sim:.6f}")
                sample_embeds.extend(all_video_embeds[vid_ptr : vid_ptr + num_vids])
                sample_grids.extend(all_video_grids[vid_ptr : vid_ptr + num_vids])
                vid_ptr += num_vids
                count += num_vids
            
            # If a sample has both, they are now concatenated in sample_embeds
            unified_visual_embeds.extend(sample_embeds) # Flattened list for predictor

            visual_per_sample.append(count)
            unified_visual_grids.extend(sample_grids)

        # Validation: Ensure we consumed all inputs
        if pixel_values is not None:
            assert img_ptr == len(all_image_embeds), "Mismatch in image count processing"
        if pixel_values_videos is not None:
            assert vid_ptr == len(all_video_embeds), "Mismatch in video count processing"

        if len(unified_visual_embeds) > 0:
            # Convert grids to tensor if needed by predictor
            if len(unified_visual_grids) > 0:
                unified_grid_thw_tensor = torch.stack(unified_visual_grids)
            else:
                unified_grid_thw_tensor = None

            if hasattr(self.config, "use_text") and self.config.use_text:
                _, L = input_ids.shape
                visual_ids = [self.config.vision_end_token_id]
                
                is_visual_global = torch.zeros_like(input_ids, dtype=torch.bool)
                for vid in visual_ids:
                    if vid != -1:
                        is_visual_global |= (input_ids == vid)
                
                cutoff = 0
                if is_visual_global.any():
                    positions_global = torch.arange(L, device=self.device).unsqueeze(0)
                    visual_indices = torch.where(is_visual_global, positions_global, -1)
                    last_visual_idx_global = visual_indices.max(dim=1).values # (B,)
                    
                    min_last_idx = last_visual_idx_global.min().item()
                    cutoff = max(0, min_last_idx + 1)
                
                if cutoff > 0 and cutoff < L:
                    input_ids = input_ids[:, cutoff:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, cutoff:]
                
                text_embeds = self.predictor.embed_tokens(input_ids)
                B, L_new = input_ids.shape
                positions = torch.arange(L_new, device=self.device).unsqueeze(0) # (1, L_new)
                
                is_visual = torch.zeros_like(input_ids, dtype=torch.bool)
                for vid in visual_ids:
                    if vid != -1:
                        is_visual |= (input_ids == vid)
                
                if is_visual.any():
                    visual_indices = torch.where(is_visual, positions, -1)
                    last_visual_idx = visual_indices.max(dim=1, keepdim=True).values 
                else:
                    last_visual_idx = torch.full((B, 1), -1, device=self.device)

                target_suffix = torch.tensor(
                    [1446, 34813, 1744, 911, 279, 32711, 1882, 438, 458], 
                    device=self.device
                )
                len_suffix = len(target_suffix)
                suffix_start_idx = torch.full((B, 1), L_new, device=self.device)
                
                # unfold shape: (B, Windows, Window_Size)
                if L_new >= len_suffix:
                    windows = input_ids.unfold(dimension=1, size=len_suffix, step=1)
                    matches = (windows == target_suffix).all(dim=2) # (B, Windows) -> bool
                    
                    has_match, match_indices = matches.max(dim=1) # max on bool returns first True
                    found_mask = has_match.unsqueeze(1)
                    match_indices = match_indices.unsqueeze(1)
                    
                    suffix_start_idx = torch.where(found_mask, match_indices, suffix_start_idx)

                mask_after_visual = positions > last_visual_idx
                mask_before_suffix = positions < suffix_start_idx
                relevant_mask = mask_after_visual & mask_before_suffix

                if attention_mask is not None:
                    text_mask = attention_mask.bool() & relevant_mask
                else:
                    pad_id = self.tokenizer.pad_token_id
                    text_mask = (input_ids != pad_id) & relevant_mask

                if text_mask is not None:
                    if attention_mask is not None:
                        fallback_mask = attention_mask.bool()
                    else:
                        pad_id = self.tokenizer.pad_token_id
                        fallback_mask = (input_ids != pad_id)
                    empty_rows = text_mask.sum(dim=1) == 0
                    if empty_rows.any():
                        text_mask = text_mask.clone()
                        text_mask[empty_rows] = fallback_mask[empty_rows]

            else:
                text_embeds = None
                text_mask = None
                
            new_actions, scales, log_probs, new_scale_mask = self.predictor(
                visual_features=unified_visual_embeds, # List of Tensors
                visual_grid_thw=unified_grid_thw_tensor,
                text_features=text_embeds,
                text_mask=text_mask,
                actions=actions,
                visual_per_sample=visual_per_sample, # [2, 1, 0, ...]
                eval_mode=eval_mode,
                scale_mask=scale_mask,
                compute_frame_metrics=kwargs.get("compute_frame_metrics", False),
            )
        
        elif sum(visual_per_sample) == 0:
            B = len(visual_per_sample)
            nframes = self.config.max_frames
            device = self.device if hasattr(self, "device") else (input_ids.device if input_ids is not None else None)
            if device is None:
                device = torch.device("cpu")

            if actions is not None:
                log_probs = torch.zeros_like(actions, dtype=torch.float32, device=actions.device)
                return {"log_probs": log_probs}

            use_discrete_action = bool(getattr(self.config, "use_discrete_action", False))
            actions_dtype = torch.long if use_discrete_action else torch.float32

            scales = torch.ones((B, nframes), dtype=torch.float32, device=device)
            new_actions = torch.zeros((B, nframes), dtype=actions_dtype, device=device)
            new_scale_mask = torch.zeros((B, nframes), dtype=torch.bool, device=device)
            log_probs = torch.zeros((B, nframes), dtype=torch.float32, device=device)

            if not kwargs.get("return_mm_data", True):
                return {
                    "scales": scales,
                    "actions": new_actions,
                    "scale_mask": new_scale_mask,
                    "log_probs": log_probs,
                }

            return {
                "multi_modal_data": multi_modal_data,
                "actions": new_actions,
                "log_probs": log_probs,
                "scale_mask": new_scale_mask,
                "scaled_messages": messages,
            }

        if actions is not None:
            # frame_features = getattr(self.predictor, "last_frame_features", None)
            # Get losses for both V1 and V2 predictors (using proxy properties)
            # Both RegressionHeadPredictor and RegressionHeadPredictorV2 now have these properties
            contrastive_loss = getattr(self.predictor, "_last_contrastive_loss", None)
            sim_scale_loss = getattr(self.predictor, "_last_sim_scale_loss", None)
            
            # Fallback to scorer for V1 (which doesn't have proxy properties yet)
            if contrastive_loss is None and hasattr(self.predictor, "scorer"):
                scorer = self.predictor.scorer
                contrastive_loss = getattr(scorer, "_last_contrastive_loss", None)
                sim_scale_loss = getattr(scorer, "_last_sim_scale_loss", None)
            
            # Get frame metrics for frame-aware advantage computation
            frame_metrics = None
            if hasattr(self.predictor, "get_frame_metrics"):
                frame_metrics = self.predictor.get_frame_metrics()
            elif hasattr(self.predictor, "scorer") and hasattr(self.predictor.scorer, "get_frame_metrics"):
                frame_metrics = self.predictor.scorer.get_frame_metrics()
            
            return {
                "log_probs": log_probs,
                # "frame_features": frame_features,
                "contrastive_loss": contrastive_loss,
                "sim_scale_loss": sim_scale_loss,
                "frame_metrics": frame_metrics,
            }

        if not kwargs.get("return_mm_data", True):
            return {
                "scales": scales,
                "actions": new_actions,
                "scale_mask": new_scale_mask,
                "log_probs": log_probs,
            }

        scaled_messages = None
        if scales is not None:
            scaled_multi_modal_data = apply_adaptive_scaling(
                multi_modal_data=multi_modal_data,
                scales=scales,
                new_scale_mask=new_scale_mask,
                processor=self.processor,
                patch_size=patch_size,
                image_factor=image_factor,
                temporal_patch_size=self.processor.video_processor.temporal_patch_size,
                **kwargs
            )

            scaled_messages = scale_messages_from_mmdata(messages, scaled_multi_modal_data, kwargs.get("video2image", False))
            # breakpoint()
            # scaled_messages[0][0]['content'][0]['image'].shape

        # Retrieve frame metrics if computed (Critical for Advantage Calculation)
        frame_metrics = None
        if hasattr(self.predictor, "get_frame_metrics"):
            frame_metrics = self.predictor.get_frame_metrics()
        elif hasattr(self.predictor, "scorer") and hasattr(self.predictor.scorer, "get_frame_metrics"):
            frame_metrics = self.predictor.scorer.get_frame_metrics()

        return {
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "position_ids": position_ids,
            "multi_modal_data": scaled_multi_modal_data,
            "actions": new_actions,
            "log_probs": log_probs,
            # "scales": scales,
            "scale_mask": new_scale_mask,
            "scaled_messages": scaled_messages if scaled_messages is not None else messages,
            "frame_metrics": frame_metrics,
        }
    
    def _debug_nan(self, large_threshold=1e20, small_threshold=1e-20, optimizer=None):
        flag = 0
        for name, param in self.named_parameters():
            t = param.detach()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(torch.float32)
            report_lines = []
            # param checks
            nan_mask = torch.isnan(t)
            inf_mask = torch.isinf(t)
            abs_t = torch.abs(t)
            large_mask = abs_t > large_threshold
            small_nonzero_mask = (abs_t < small_threshold) & (t != 0)
            if nan_mask.any() or inf_mask.any() or large_mask.any() or small_nonzero_mask.any():
                report_lines.append(
                    f"param: nan={nan_mask.sum().item()}, inf={inf_mask.sum().item()}, "
                    f"> {large_threshold:.0e}={large_mask.sum().item()}, "
                    f"< {small_threshold:.0e}(nz)={small_nonzero_mask.sum().item()}, "
                    f"min={t.min().item():.3e}, max={t.max().item():.3e}"
                )
            # grad checks
            if param.grad is not None:
                g = param.grad.detach()
                if g.dtype in (torch.bfloat16, torch.float16):
                    g = g.to(torch.float32)
                g_nan = torch.isnan(g).sum().item()
                g_inf = torch.isinf(g).sum().item()
                g_abs = torch.abs(g)
                g_large = (g_abs > large_threshold).sum().item()
                g_small_nz = ((g_abs < small_threshold) & (g != 0)).sum().item()
                if g_nan or g_inf or g_large or g_small_nz:
                    report_lines.append(
                        f"grad: nan={g_nan}, inf={g_inf}, > {large_threshold:.0e}={g_large}, "
                        f"< {small_threshold:.0e}(nz)={g_small_nz}, "
                        f"min={g.min().item():.3e}, max={g.max().item():.3e}"
                    )
            # optimizer state checks
            if optimizer is not None:
                st = optimizer.state.get(param, None)
                if st is not None:
                    for k in ('exp_avg', 'exp_avg_sq'):
                        if k in st:
                            s = st[k]
                            s_t = s.detach().to(torch.float32) if s.dtype in (torch.bfloat16, torch.float16) else s.detach()
                            s_nan = torch.isnan(s_t).sum().item()
                            s_inf = torch.isinf(s_t).sum().item()
                            if s_nan or s_inf:
                                report_lines.append(
                                    f"state {k}: nan={s_nan}, inf={s_inf}, min={s_t.min().item():.3e}, max={s_t.max().item():.3e}"
                                )
            if report_lines:
                flag = 1
                print(f"[{name}] dtype={param.dtype}, device={param.device} :: " + " | ".join(report_lines))
        return flag

    def get_weighted_contrastive_loss(self):
        """Get weighted contrastive loss from predictor for training (V2 only)."""
        if hasattr(self.predictor, 'get_weighted_contrastive_loss'):
            return self.predictor.get_weighted_contrastive_loss()
        return None

    def get_contrastive_loss(self):
        """Get raw contrastive loss from predictor for training (V2 only)."""
        if hasattr(self.predictor, 'get_contrastive_loss'):
            return self.predictor.get_contrastive_loss()
        return None


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        multi_modal_data=None,
        messages: list[dict] = None,
        actions: Optional[torch.Tensor] = None,
        scale_mask: Optional[torch.Tensor] = None,
        cal_pred_log_prob: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, UniQwenOutputWithScale]:
        """
        Forward pass with optional multi-modal scaling.
        - If `multi_modal_data` is present (and not computing actions), or `cal_pred_log_prob=True`,
          route to `scale_multi_modal00` for predictor log-probs or scaled inputs.
        - Otherwise, run the base language model forward and optionally compute predictor log-probs when actions are provided.
        """
        # Route to scaling path when computing predictor log-probs or preparing scaled inputs
        # if cal_pred_log_prob or (multi_modal_data is not None and actions is None):
        return self.scale_multi_modal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            multi_modal_data=multi_modal_data,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            messages=messages,
            actions=actions,
            scale_mask=scale_mask,
            **kwargs,
        )

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    from verl.utils import hf_processor, hf_tokenizer
    local_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/my_qwen2.5_vl-3b"
    local_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_discrete"
    model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_flash_new"
    model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_gate"
    model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv2"
    model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictorv1"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify/scale_disc_filter_mul_cost_enc0.5-fsdp2-s16-7B-Single-bsz128-mini32-n1-min0.2-max2.0-len4-resp8/global_step_10/pred"
    # local_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
    tokenizer = hf_tokenizer(local_path)
        # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, use_fast=True)
    import datasets
    parquet_file = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Val/train.parquet"
    parquet_file = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet"
    # parquet_file = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-Mixedpath/val.parquet"
    # dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    from visionthink.main_ppo import create_rl_dataset
    config = {
        # 基础配置
        "tokenizer": None,
        "use_shm": False,
        "train_files": "~/data/rlhf/gsm8k/train.parquet",
        "val_files": "~/data/rlhf/gsm8k/test.parquet",
        "train_max_samples": -1,
        "val_max_samples": 20,
        "prompt_key": "prompt",
        "reward_fn_key": "data_source",
        "max_prompt_length": 8192,
        "max_response_length": 512,
        "train_batch_size": 1024,
        "val_batch_size": None,
        "return_raw_input_ids": False,
        "return_raw_chat": True,
        "return_full_prompt": False,
        "shuffle": True,
        "seed": None,
        "dataloader_num_workers": 8,
        "image_patch_size": 14,
        "validation_shuffle": False,
        "filter_overlong_prompts": False,
        "filter_overlong_prompts_workers": 1,
        "truncation": "error",
        "image_key": "images",
        "video_key": "videos",
        "trust_remote_code": False,
        "return_multi_modal_inputs": True,
        
        # 嵌套配置（custom_cls、sampler、datagen）
        "custom_cls": {
            "path": None,
            "name": None
        },
        "sampler": {
            "class_path": None,
            "class_name": None
        },
        "datagen": {
            "path": None,
            "name": None
        },
        
        # 额外参数
        "apply_chat_template_kwargs": {}
    }
    from types import SimpleNamespace
    # config = SimpleNamespace(**config)
    # from visionthink.adaptive.rl_dataset import RLHFDataset
    # from visionthink.predictor.rl_dataset import CustomRLHFDataset
    from verl.utils.dataset import RLHFDataset
    dataset = RLHFDataset(
        data_files=parquet_file,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        max_samples=20,
    )
    # breakpoint()
    config_ori = AutoConfig.from_pretrained(local_path)
    config = PredictorConfig.from_pretrained(model_path)
    # config.scale_bins = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # config.continuous_dist = "logistic_normal"
    # config.continuous_eval_quantile = 0.7
    # config.logistic_normal_init_sigma = 0.7

    # ===== Base Config =====
    config.continuous_dist = "beta"
    config.beta_param_scale = 0.5
    config.beta_add_one = True

    config.use_discrete_action = False
    config.use_text = True
    config.use_text_encoder = True
    config.text_encoder_depth = 2
    config.text_encoder_ff_mult = 4
    config.text_encoder_heads = 8
    config.text_encoder_dropout = 0.0
    config.regression_head_mode = "mlp"
    config.use_cross_attn_gate = True
    config.min_scale = 0.25
    config.max_scale = 2.0
    config.max_frames = 4
    config.output_dim = 1024
    config.dim_head = 64
    # Note: cross_depth/self_depth only affect V1 (PredictorDecoderLayer), not V2
    config.cross_depth = 2
    config.self_depth = 2
    config.tower_depth = 16
    config.vocab_size = config_ori.vocab_size
    config.llm_hidden_size = config_ori.hidden_size
    config.pad_token_id = config_ori.pad_token_id
    config.image_token_id = config_ori.image_token_id
    config.video_token_id = config_ori.video_token_id
    config.vision_start_token_id = config_ori.vision_start_token_id
    config.vision_end_token_id = config_ori.vision_end_token_id
    config.vision_token_id = config_ori.vision_token_id

    # ===== V2 Predictor Config (DifferentiableImportancePredictor) =====
    config.predictor_version = "v2"  # "v1" for original, "v2" for new architecture

    # Sparse gating
    config.gate_mode = "gumbel_softmax"  # Differentiable selection
    config.gate_temperature = 0.5  # Starting temperature (removed duplicate config)
    config.gate_k_ratio = 0.25  # Top-k ratio for sparse selection

    # Contrastive loss (Feature Diversity) - increased weight for better frame differentiation
    config.contrastive_weight = 0.2  # Increased from 0.1 to 0.2 for stronger diversity
    config.contrastive_temperature = 0.1  # Standard InfoNCE temp

    # Sim Scale Loss (Redundancy Compression) - increased weight for better redundancy handling
    # Penalize scale[t] > scale[t-1] when sim(frame[t], frame[t-1]) is high
    config.sim_scale_weight = 0.2  # Increased from 0.1 to 0.2 for stronger compression
    config.sim_tau = 0.5  # Similarity threshold (redundancy threshold)
    config.sim_temp = 0.1  # Sigmoid temperature (sharpness of transition)
    config.sim_gamma = 0.05  # Scale decrease magnitude
    
    model = PredictorForConditionalGeneration.from_pretrained(model_path, config=config).cuda()

    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # model = PredictorForConditionalGeneration.from_pretrained(
    #     model_path,
    #     config=config,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    total, trainable = count_params(model)

    print(f"Total params: {total/1e9:.3f} B")
    print(f"Trainable params: {trainable/1e9:.3f} B")

    # image = dataset[0]['multi_modal_data']['image']
    image = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/VisionThink/scissor.png")
    image1 = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Pretrain/images/00000/000000010.jpg")
    messages = [
            [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        # "video": [
                        #     "file:///path/to/frame1.jpg",
                        #     "file:///path/to/frame2.jpg",
                        #     "file:///path/to/frame3.jpg",
                        #     "file:///path/to/frame4.jpg",
                        # ],
                        "video": "file:///mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4",
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
    ]

    messages_text = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "video",
                #     # "video": [
                #     #     "file:///path/to/frame1.jpg",
                #     #     "file:///path/to/frame2.jpg",
                #     #     "file:///path/to/frame3.jpg",
                #     #     "file:///path/to/frame4.jpg",
                #     # ],
                #     "video": "file:///mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"
                # },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # texts = [
    #     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    #     for msg in messages
    # ]
    # # image_inputs, video_inputs = process_vision_info(messages)
    # image_inputs, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    # imgs_np = [to_numpy_array(img) for img in image_inputs]
    # video_inputs = [video[0] for video in videos]
    # video_metadata = [video[1] for video in videos]
    # new_videos = [
    #     (video, metadata) for video, metadata in zip(video_inputs, video_metadata, strict=True)
    # ]
    # inputs = processor(
    #     text=texts,
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs['input_ids'], inputs['attention_mask'] = verl_F.postprocess_data(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask'],
    #     max_length=10000,
    #     pad_token_id=tokenizer.pad_token_id,
    #     left_pad=True,
    #     truncation='left',
    # )
    # inputs_scale = inputs.to("cuda")
    # # breakpoint()processor.tokenizer.vision_end_token_id
    # inputs_scale.update({"multi_modal_data": [{"image": [image[0], dataset[1]['multi_modal_data']['image'][0]]}, {"image": image}], "text": texts, "eval_mode": False})
    # inputs_scale.update({"multi_modal_data": [{"image": [image[0], dataset[1]['multi_modal_data']['image'][0]]}, {"image": image}, {"video": new_videos}], "text": texts, "eval_mode": True})
    # inputs_scale['cal_pred_log_prob']=True
    # save_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor-disc-new"
    # model.save_pretrained(save_path, safe_serialization=True, dtype=model.dtype)
    # tokenizer.save_pretrained(save_path)
    # processor.save_pretrained(save_path)
    # breakpoint()
    # model.predictor.embed_tokens.weight
    # inputs_scale['input_ids'].shape
    # new_inputs['input_ids'].shape
    # inputs_scale['attention_mask'].shape
    # new_inputs['attention_mask'].shape
    # inputs_scale['position_ids'].shape
    # new_inputs['position_ids'].shape
    # new_inputs['multi_modal_inputs']['pixel_values']  new_inputs['multi_modal_inputs']['image_grid_thw']  new_inputs['multi_modal_inputs']['video_grid_thw']
    # inputs = {"input_ids": dataset[0]['input_ids'].unsqueeze(0), "attention_mask":dataset[0]['attention_mask'], "position_ids": dataset[0]['position_ids'], "multi_modal_data": dataset[0]['multi_modal_data'], **dataset[0]['multi_modal_inputs']}
    # new_inputs = model(**inputs_scale)
    # breakpoint()
    new_inputs = model(**{"messages": messages, "return_mm_data": False, "eval_mode": True})
    new_inputs = model(**{"messages": [messages_text, dataset[0]['raw_prompt'], dataset[-1]['raw_prompt']], "return_mm_data": False})
    breakpoint()
    new_inputs = model(**{"messages": [dataset[0]['raw_prompt'], dataset[1]['raw_prompt']], "video2image": True})
    new_inputs_video = model(**{"messages": [dataset[0]['raw_prompt']]})
    # breakpoint()
    # new_inputs['multi_modal_data'][0]['images']
    # new_inputs.pop('position_ids')
    # new_inputs_video['multi_modal_data'][0]['videos']
    # model(**new_inputs)

if __name__ == "__main__":
    main()
