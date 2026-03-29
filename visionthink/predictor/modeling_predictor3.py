import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math

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
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPreTrainedModel,
    Qwen2VLVideoProcessor
)
from torchvision.transforms import InterpolationMode
from transformers.cache_utils import Cache
from transformers.image_transforms import resize
from transformers.image_utils import (
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
    ChannelDimension,
    SizeDict
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2RMSNorm,
)

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from qwen_vl_utils import process_vision_info

from visionthink.adaptive.utils import get_images_per_sample, tensor_to_pil_list, get_visual_objects_per_sample, split_video_metadata
from visionthink.predictor.AZNet import RegressionHeadPredictor
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
    _no_split_modules = ["PredictorDecoderLayer", "ProjectorBlock"]
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
        vc._attn_implementation = "flash_attention_2"

        # Initialize vision tower and predictor
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.llm_hidden_size, padding_idx=config.pad_token_id)
        self.vision_tower = UniQwenVisionTransformer(vc)
        self.predictor = RegressionHeadPredictor(vc)

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
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
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
            model = cls(config).to(torch_dtype)
            
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


    def scale_multi_modal(
        self,
        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # multi_modal_data=None,
        # pixel_values: Optional[torch.Tensor] = None,
        # pixel_values_videos: Optional[torch.FloatTensor] = None,
        # image_grid_thw: Optional[torch.LongTensor] = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
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

        if hasattr(multi_modal_data, "tolist"):
            multi_modal_data = multi_modal_data.tolist()

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
            
            if "image" in mm_data and mm_data["image"] is not None:
                num_imgs = len(mm_data["image"])
                # Extract the next N images from the global queue
                sample_embeds.extend(all_image_embeds[img_ptr : img_ptr + num_imgs])
                sample_grids.extend(all_image_grids[img_ptr : img_ptr + num_imgs])
                img_ptr += num_imgs
                count += num_imgs

            if "video" in mm_data and mm_data["video"] is not None:
                num_vids = len(mm_data["video"])
                # Extract the next N videos from the global queue
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
                device = input_ids.device
                visual_ids = [self.config.vision_end_token_id]
                
                is_visual_global = torch.zeros_like(input_ids, dtype=torch.bool)
                for vid in visual_ids:
                    if vid != -1:
                        is_visual_global |= (input_ids == vid)
                
                cutoff = 0
                if is_visual_global.any():
                    positions_global = torch.arange(L, device=device).unsqueeze(0)
                    visual_indices = torch.where(is_visual_global, positions_global, -1)
                    last_visual_idx_global = visual_indices.max(dim=1).values # (B,)
                    
                    min_last_idx = last_visual_idx_global.min().item()
                    cutoff = max(0, min_last_idx + 1)
                
                if cutoff > 0 and cutoff < L:
                    input_ids = input_ids[:, cutoff:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, cutoff:]
                
                text_embeds = self.predictor.embed_tokens(input_ids)
                
                _, L_new = input_ids.shape
                positions = torch.arange(L_new, device=device).unsqueeze(0) # (1, L_new)
                
                is_visual = torch.zeros_like(input_ids, dtype=torch.bool)
                for vid in visual_ids:
                    if vid != -1:
                        is_visual |= (input_ids == vid)
                
                if is_visual.any():
                    visual_indices = torch.where(is_visual, positions, -1)
                    last_visual_idx = visual_indices.max(dim=1, keepdim=True).values # (B, 1)
                    relevant_mask = positions > last_visual_idx
                else:
                    relevant_mask = torch.ones_like(input_ids, dtype=torch.bool)

                if attention_mask is not None:
                    text_mask = attention_mask.bool() & relevant_mask
                else:
                    pad_id = self.tokenizer.pad_token_id
                    text_mask = (input_ids != pad_id) & relevant_mask
                
            else:
                input_ids = None
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
            )
        
        if actions is not None:
            return {"log_probs": log_probs}

        if scales is not None:
            scales_cpu = scales.cpu()
            scale_mask_cpu = new_scale_mask.cpu() if new_scale_mask is not None else None
            
            scaled_multi_modal_data = []
            
            # We need to iterate again and maintain a global index for the 'scales' tensor
            # scales tensor is usually [Batch_Size, Max_Objects_Per_Sample] 
            # OR [Total_Objects] depending on your implementation. 
            # Assuming [Batch, Max_Objects] structure common in RL.
            
            for i, mm_data in enumerate(multi_modal_data):
                new_mm_data = mm_data.copy()
                object_idx_in_sample = 0 # Track index within this sample (0, 1, 2...)
                
                if "image" in mm_data and mm_data["image"] is not None:
                    imgs = mm_data["image"]
                    imgs_np = [to_numpy_array(img) for img in imgs]
                    scaled_images = []
                    
                    for j, img_np in enumerate(imgs_np):
                        scale_factor = scales_cpu[i][object_idx_in_sample].item()
                        # Check mask
                        if scale_mask_cpu is not None:
                            assert scale_mask_cpu[i][object_idx_in_sample], f"Padding scale used for real image!"

                        input_data_format = infer_channel_dimension_format(img_np)
                        height, width = get_image_size(img_np, channel_dim=input_data_format)

                        new_h, new_w = int(height * scale_factor), int(width * scale_factor)
                        new_h, new_w = max(patch_size, new_h), max(patch_size, new_w)
                        
                        resized_h, resized_w = smart_resize(new_h, new_w, factor=image_factor, max_pixels=IMAGE_MAX_TOKEN_NUM * image_factor ** 2)
                        
                        image_resized = resize(img_np, size=(resized_h, resized_w), resample=PILImageResampling.BICUBIC, input_data_format=input_data_format)
                        # scaled_images.append(Image.fromarray(image_resized))
                        scaled_images.append(torch.from_numpy(image_resized).contiguous().cpu())
                        
                        object_idx_in_sample += 1 # Increment local counter
                    
                    new_mm_data["image"] = scaled_images

                if "video" in mm_data and mm_data["video"] is not None:
                    videos = mm_data["video"]
                    scaled_videos = []
                    for j, video in enumerate(videos):
                        if isinstance(video, tuple):
                            video_frames, video_metadata = video
                            video_metadata = split_video_metadata(video_metadata, temporal_patch_size)
                            for idx in range(len(video_metadata)):
                                video_metadata[idx]["video_timestamps"] = idx * temporal_patch_size
                        else:
                            video_frames = video
                            video_metadata = [{"video_timestamps": idx} for idx in range(0, video_frames.shape[0], temporal_patch_size)]
                        num_frames = video_frames.shape[0]
                        height, width = get_image_size(video_frames[0], channel_dim=ChannelDimension.FIRST)

                        new_video_chunks = []
                        for k in range(0, num_frames, temporal_patch_size):
                            idx = k // temporal_patch_size
                            chunk_frames = video_frames[k : k + temporal_patch_size]
                            current_scale_idx = object_idx_in_sample + idx
                            scale_factor = scales_cpu[i][current_scale_idx].item()
                            
                            if scale_mask_cpu is not None:
                                assert scale_mask_cpu[i][current_scale_idx], \
                                    f"Padding scale used for real video frame chunk starting at frame {k} in Sample {i}!"

                            new_h, new_w = int(height * scale_factor), int(width * scale_factor)
                            new_h, new_w = max(patch_size, new_h), max(patch_size, new_w)
                            
                            resized_h, resized_w = smart_resize(
                                new_h, new_w, 
                                factor=image_factor, 
                                max_pixels=VIDEO_MAX_TOKEN_NUM * image_factor ** 2
                            )
                            
                            chunk_processed_frames = self.processor.video_processor.resize(
                                image=chunk_frames,
                                size=SizeDict(height=resized_h, width=resized_w),
                                interpolation=InterpolationMode.BICUBIC,
                            )
                            
                            new_video_chunks.append((chunk_processed_frames.cpu(), video_metadata[idx]))

                        scaled_videos.extend(new_video_chunks)
                        # scaled_video_metadata.extend(video_metadata)
                        
                        num_chunks = math.ceil(num_frames / temporal_patch_size)
                        object_idx_in_sample += num_chunks
                        
                    new_mm_data["video"] = scaled_videos
                    # print(new_mm_data["video"])
                scaled_multi_modal_data.append(new_mm_data)

        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "position_ids": position_ids,
            "multi_modal_data": scaled_multi_modal_data,
            "actions": new_actions,
            "log_probs": log_probs,
            # "scales": scales,
            "scale_mask": new_scale_mask
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
        # print("@" * 50)
        # # print(f"input_ids.shape: {input_ids.shape}")
        # # print(f"pixel_values.shape: {pixel_values.shape}")
        # # if position_ids is not None:
        # #     print(f"position_ids.shape: {position_ids.shape}")
        # # print(f"multi_modal_data: {multi_modal_data}")
        # # print(f"text_embeds.shape: {text_embeds[:, last_image_pos + 1:].shape}")
        # print(f"input_ids.dtype: {input_ids.dtype}")
        # if position_ids is not None:
        #     print(f"position_ids.dtype: {position_ids.dtype}")
        # print(f"image_grid_thw.dtype: {image_grid_thw.dtype}")
        # print(f"pixel_values.dtype: {pixel_values.dtype}")
        # print("@" * 50)

        ###
        # print("%" * 50, "forward")
        # print("self.dtype", self.dtype)
        # print("self.model.dtype", self.model.dtype)
        # print("self.predictor.dtype", self.predictor.dtype)
        # print("self.vision_tower.dtype", self.vision_tower.dtype)
        # print("self.visual.dtype", self.visual.dtype)

        # print("self.predictor.weight.dtype", self.predictor.mlp[0].weight.dtype)
        # print("self.predictor.scorer.weight.dtype", self.predictor.scorer.layers[0].cross_attn.to_q.weight.dtype)
        # print("self.vision_tower.weight.dtype", self.vision_tower.blocks[0].mlp.gate_proj.weight.dtype)
        # print("self.visual.weight.dtype", self.visual.blocks[0].mlp.gate_proj.weight.dtype)
        # print("self.model.weight.dtype", self.language_model.layers[0].mlp.gate_proj.weight.dtype)
        # print("%" * 50, "forward")
        ###

        # self._debug_nan()

        # Route to scaling path when computing predictor log-probs or preparing scaled inputs
        if cal_pred_log_prob or (multi_modal_data is not None and actions is None):
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

        print("predictor forward error!!!")

        # Optionally compute predictor log-probs with original (pre-scaled) inputs
        log_probs = None
        if actions is not None:
            ori_input_ids = kwargs.pop("ori_input_ids")
            ori_position_ids = kwargs.pop("ori_position_ids")
            ori_attention_mask = kwargs.pop("ori_attention_mask")
            ori_multi_modal_inputs = kwargs.pop("ori_multi_modal_inputs")
            log_probs = self.scale_multi_modal(
                input_ids=ori_input_ids,
                attention_mask=ori_attention_mask,
                position_ids=ori_position_ids,
                multi_modal_data=multi_modal_data,
                actions=actions,
                **ori_multi_modal_inputs,
            )["log_probs"]

        # Standard LM forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            second_per_grid_ts=second_per_grid_ts,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return UniQwenOutputWithScale(
            loss=outputs.loss,
            logits=outputs.logits,
            log_probs=log_probs,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


        # if multi_modal_data is None:
        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask, 
        #         position_ids=position_ids, 
        #         past_key_values=past_key_values, 
        #         inputs_embeds=inputs_embeds,
        #         labels=labels, 
        #         use_cache=use_cache, 
        #         output_attentions=output_attentions, 
        #         output_hidden_states=output_hidden_states, 
        #         pixel_values=pixel_values, 
        #         pixel_values_videos=pixel_values_videos,
        #         image_grid_thw=image_grid_thw,
        #         video_grid_thw=video_grid_thw,
        #         rope_deltas=rope_deltas,
        #         second_per_grid_ts=second_per_grid_ts,
        #         cache_position=cache_position,
        #         logits_to_keep=logits_to_keep,
        #         **kwargs 
        #     )
        # else:
        #     # print(f"scale multi_modal_data: {multi_modal_data}")
        #     return self.scale_multi_modal(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         multi_modal_data=multi_modal_data,
        #         pixel_values=pixel_values, 
        #         pixel_values_videos=pixel_values_videos,
        #         image_grid_thw=image_grid_thw,
        #         video_grid_thw=video_grid_thw,
        #         actions=actions,
        #         **kwargs
        #     )


def main():
    from verl.utils import hf_processor, hf_tokenizer
    local_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/my_qwen2.5_vl-3b"
    local_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_discrete"
    model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_flash_new"
    # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/ckpts/GeneralQA_Qwen_Verify/scale_disc_filter_mul_cost_enc0.5-fsdp2-s16-7B-Single-bsz128-mini32-n1-min0.2-max2.0-len4-resp8/global_step_10/pred"
    # local_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
    tokenizer = hf_tokenizer(local_path)
        # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, use_fast=True)
    import datasets
    parquet_file = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Val/train.parquet"
    # parquet_file = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/val.parquet"
    # dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    from visionthink.main_ppo import create_rl_dataset
    config = {}
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
    from visionthink.adaptive.rl_dataset import RLHFDataset
    from visionthink.data import CustomRLHFDataset
    dataset = RLHFDataset(
        data_files=parquet_file,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        max_samples=20,
    )
    # breakpoint()
    # config_ori = AutoConfig.from_pretrained(local_path)
    # config = PredictorConfig.from_pretrained(model_path)
    # # config.scale_bins = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # config.use_discrete_action = True
    # config.use_text = True
    # config.min_scale = 0.25
    # config.max_scale = 2.0
    # config.max_frames = 4
    # config.output_dim = 1024
    # config.dim_head = 64
    # config.cross_depth = 2
    # config.self_depth = 2
    # config.tower_depth = 8
    # config.vocab_size = config_ori.vocab_size
    # config.llm_hidden_size = config_ori.hidden_size
    # config.pad_token_id = config_ori.pad_token_id
    # config.image_token_id = config_ori.image_token_id
    # config.video_token_id = config_ori.video_token_id
    # config.vision_start_token_id = config_ori.vision_start_token_id,
    # config.vision_end_token_id = config_ori.vision_end_token_id,
    # config.vision_token_id = config_ori.vision_token_id,
    # model = PredictorForConditionalGeneration.from_pretrained(model_path, config=config).cuda()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = PredictorForConditionalGeneration.from_pretrained(model_path, config=config, trust_remote_code=True, dtype="auto", ignore_mismatched_sizes=True).cuda()

    image = dataset[0]['multi_modal_data']['image']
    messages = [
            [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image[0],
                    },
                    {
                        "type": "image",
                        "image": dataset[1]['multi_modal_data']['image'][0],
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
                        "image": image[0],
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

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 # "video": [
    #                 #     "file:///path/to/frame1.jpg",
    #                 #     "file:///path/to/frame2.jpg",
    #                 #     "file:///path/to/frame3.jpg",
    #                 #     "file:///path/to/frame4.jpg",
    #                 # ],
    #                 "video": "file:///mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]

    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    # image_inputs, video_inputs = process_vision_info(messages)
    image_inputs, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    imgs_np = [to_numpy_array(img) for img in image_inputs]
    video_inputs = [video[0] for video in videos]
    video_metadata = [video[1] for video in videos]
    new_videos = [
        (video, metadata) for video, metadata in zip(video_inputs, video_metadata, strict=True)
    ]
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs['input_ids'], inputs['attention_mask'] = verl_F.postprocess_data(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=10000,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation='left',
    )
    inputs_scale = inputs.to("cuda")
    # breakpoint()processor.tokenizer.vision_end_token_id
    inputs_scale.update({"multi_modal_data": [{"image": [image[0], dataset[1]['multi_modal_data']['image'][0]]}, {"image": image}], "text": texts, "eval_mode": False})
    inputs_scale.update({"multi_modal_data": [{"image": [image[0], dataset[1]['multi_modal_data']['image'][0]]}, {"image": image}, {"video": new_videos}], "text": texts, "eval_mode": True})
    inputs_scale['cal_pred_log_prob']=True
    # save_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor-disc-new"
    # model.save_pretrained(save_path, safe_serialization=True, torch_dtype=model.dtype)
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
    new_inputs = model(**inputs_scale)
    breakpoint()
    # new_inputs['multi_modal_data'][2]['video']
    # new_inputs.pop('position_ids')
    # multi_modal_inputs =
    model(**new_inputs)

if __name__ == "__main__":
    main()