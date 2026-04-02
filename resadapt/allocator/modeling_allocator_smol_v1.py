import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, AutoModelForImageTextToText, PreTrainedModel
from concurrent.futures import ThreadPoolExecutor

from resadapt.allocator.attention_utils import (
    forward_hf_text_model_safe,
    resolve_pretrained_attn_implementation,
    torch_dtype_for_hf_pretrained,
)
from resadapt.allocator.smol_config import SmolAllocatorConfig
from resadapt.allocator.aznet_smol_v1 import RegressionHeadAllocatorSmol
from resadapt.allocator.video_decode_utils import patch_video_processor_fetch_videos


class SmolAllocatorForConditionalGeneration(PreTrainedModel):
    config_class = SmolAllocatorConfig

    supports_gradient_checkpointing = True
    # _no_split_modules = ["SmolVLMVisionAttention", "SmolVLMDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config: SmolAllocatorConfig):
        super().__init__(config)
        self.config = config
        model_name = getattr(config, "smol_model_name", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_name)

        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            self.processor.video_processor.num_frames = int(getattr(config, "max_frames", 16))
            self.processor.video_processor.fps = int(getattr(config, "fps", 2.0))
            patch_video_processor_fetch_videos(self.processor.video_processor)
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            self.processor.image_processor.do_image_splitting = False
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        weight_dtype = torch_dtype_for_hf_pretrained(config)
        attn_impl = resolve_pretrained_attn_implementation(
            getattr(config, "_attn_implementation", None),
            weight_dtype=weight_dtype,
        )
        load_kwargs = {
            "attn_implementation": attn_impl,
        }
        if weight_dtype is not None:
            load_kwargs["dtype"] = weight_dtype
        self.smol_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.allocator = RegressionHeadAllocatorSmol(config)

    def _resolve_text_model(self):
        for attr in ["text_model", "language_model", "model"]:
            m = getattr(self.smol_model.model, attr, None)
            if m is not None:
                return m
        m = getattr(self.smol_model, "get_text_model", None)
        if callable(m):
            return m()
        return None

    def _resolve_vision_model(self):
        for attr in ["vision_model", "vision_tower", "vision_encoder", "vision_module"]:
            m = getattr(self.smol_model.model, attr, None)
            if m is not None:
                return m
        m = getattr(self.smol_model.model, "get_vision_model", None)
        if callable(m):
            return m()
        return None

    def _encode_text(self, input_ids, attention_mask):
        text_model = self._resolve_text_model()
        if text_model is None:
            emb = self.smol_model.get_input_embeddings()
            return emb(input_ids)
        return forward_hf_text_model_safe(
            text_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            smol_parent=self.smol_model,
        )

    def _encode_vision(self, pixel_values):
        vision_model = self._resolve_vision_model()
        if vision_model is None:
            return None
        outputs = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    def _sample_has_video(self, sample):
        if isinstance(sample, list):
            for msg in sample:
                if self._sample_has_video(msg):
                    return True
            return False
        content = sample.get("content", "")
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    return True
        return False

    def _sample_hw(self, pixel_values, sample_index):
        if pixel_values is None:
            return None
        h, w = pixel_values[sample_index].shape[-2:]
        return h, w

    def _get_image_features00(self, pixel_values, pixel_attention_mask):
        if hasattr(self.smol_model, "get_image_features"):
            return self.smol_model.get_image_features(pixel_values, pixel_attention_mask)
        return self._encode_vision(pixel_values)

    def _get_image_features(self, pixel_values, pixel_attention_mask=None):
        if hasattr(self.smol_model, "get_image_features"):
            return self.smol_model.get_image_features(pixel_values, pixel_attention_mask)
        return self._encode_vision(pixel_values)

    def _infer_grid_thw(self, token_count, num_frames=None):
        merge_len = self.config.spatial_merge_size ** 2
        t = num_frames if num_frames else 1
        l = max(1, token_count // t) if num_frames else token_count
        return torch.tensor([t, l * merge_len, 1], device=self.device, dtype=torch.long)

    @staticmethod
    def _normalize_content(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            content = [content]
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "video" and "video" not in item and "path" in item:
                    item["video"] = item["path"]
                elif item_type == "image" and "image" not in item and "path" in item:
                    item["image"] = item["path"]
        return content

    def _get_video_total_frames(self, video_path):
        if not isinstance(video_path, str):
            return None
        if not os.path.exists(video_path):
            return None
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(video_path)
            return int(decoder.num_frames)
        except Exception:
            pass
        try:
            import decord
            vr = decord.VideoReader(video_path)
            return int(len(vr))
        except Exception:
            pass
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return length if length > 0 else None
        except Exception:
            return None

    def _normalize_message(self, msg):
        if isinstance(msg, tuple):
            msg = msg[0]
        if isinstance(msg, list):
            return [self._normalize_message(m) for m in msg]
        return {**msg, "content": self._normalize_content(msg.get("content", ""))}

    @staticmethod
    def _extract_plain_text(msg):
        if isinstance(msg, list):
            return "\n".join(filter(None, (SmolAllocatorForConditionalGeneration._extract_plain_text(m) for m in msg)))
        if msg.get("role") == "system":
            return ""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        texts = []
        for item in content if isinstance(content, list) else [content]:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text"):
                texts.append(item.get("text", ""))
        return "\n".join(filter(None, texts))

    @staticmethod
    def _ensure_pixel_values_5d(pixel_values):
        if pixel_values is None:
            return None
        if pixel_values.dim() == 4:
            return pixel_values.unsqueeze(1)
        return pixel_values

    @staticmethod
    def _ensure_pixel_attention_mask_4d(pixel_attention_mask):
        if pixel_attention_mask is None:
            return None
        if pixel_attention_mask.dim() == 3:
            return pixel_attention_mask.unsqueeze(1)
        return pixel_attention_mask

    @staticmethod
    def _pad_pixel_attention_mask(pixel_attention_mask_list):
        if not pixel_attention_mask_list:
            return None
        if pixel_attention_mask_list[0].dim() == 3:
            return torch.cat(pixel_attention_mask_list, dim=0)
        max_n = max(mask.shape[1] for mask in pixel_attention_mask_list)
        padded = []
        for mask in pixel_attention_mask_list:
            if mask.shape[1] < max_n:
                pad_shape = (mask.shape[0], max_n - mask.shape[1], *mask.shape[2:])
                mask = torch.cat(
                    [mask, torch.zeros(pad_shape, device=mask.device, dtype=mask.dtype)],
                    dim=1,
                )
            padded.append(mask)
        return torch.cat(padded, dim=0)

    @staticmethod
    def _real_image_mask(pixel_values):
        batch_size, num_images = pixel_values.shape[:2]
        flat = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        nb_values = flat.shape[1:].numel()
        real_mask = (flat == 0.0).sum(dim=(-1, -2, -3)) != nb_values
        if not real_mask.any():
            real_mask[0] = True
        return real_mask.view(batch_size, num_images)

    @staticmethod
    def _flatten_grid(grid):
        if grid is None:
            return None
        if grid.dim() == 3:
            return grid.reshape(-1, grid.shape[-1])
        return grid

    def _process_single_message(self, msg):
        normalized = self._normalize_message(msg)
        # total_frames = None
        # if "content" in normalized and isinstance(normalized["content"], list):
        #     for item in normalized["content"]:
        #         if not isinstance(item, dict):
        #             continue
        #         if item.get("type") != "video":
        #             continue
        #         video_path = item.get("video") or item.get("path")
        #         frames = self._get_video_total_frames(video_path)
        #         if frames is None:
        #             continue
        #         item_max = item.get("max_frames", None)
        #         if item_max is not None:
        #             frames = min(frames, int(item_max))
        #         total_frames = int(frames) if total_frames is None else min(total_frames, int(frames))
        # max_frames = self.config.max_frames if total_frames is None else min(self.config.max_frames, total_frames)
        # try:
        inputs = self.processor.apply_chat_template(
            [normalized],
            num_frames=self.config.max_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # except Exception:
        #     retry_frames = max(1, int(max_frames) // 2)
        #     inputs = self.processor.apply_chat_template(
        #         [normalized],
        #         num_frames=retry_frames,
        #         add_generation_prompt=True,
        #         tokenize=True,
        #         return_dict=True,
        #         return_tensors="pt",
        #     )
        return {
            "normalized": normalized,
            "plain_text": self._extract_plain_text(normalized),
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "pixel_attention_mask": inputs.get("pixel_attention_mask"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
        }

    @staticmethod
    def _pad_pixel_values(pixel_values_list):
        if not pixel_values_list:
            return None
        if pixel_values_list[0].dim() != 5:
            return torch.cat(pixel_values_list, dim=0)
        max_t = max(pv.shape[1] for pv in pixel_values_list)
        padded = []
        for pv in pixel_values_list:
            if pv.shape[1] < max_t:
                pad_shape = (pv.shape[0], max_t - pv.shape[1], *pv.shape[2:])
                pv = torch.cat([pv, torch.zeros(pad_shape, device=pv.device, dtype=pv.dtype)], dim=1)
            padded.append(pv)
        return torch.cat(padded, dim=0)

    def _process_messages_parallel(self, messages, max_workers=None):
        if len(messages) <= 1 or max_workers == 1:
            return [self._process_single_message(m) for m in messages]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._process_single_message, messages))

    def scale_multi_modal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        messages=None,
        actions: torch.Tensor = None,
        scale_mask: torch.Tensor = None,
        **kwargs,
    ):
        eval_mode = kwargs.get("eval_mode", False)
        compute_aux = actions is not None and eval_mode is not True
        return_mm_data = kwargs.get("return_mm_data", False)
        preprocess_workers = kwargs.get("preprocess_workers", 16)

        if input_ids is None and messages is None:
            raise ValueError("Either input_ids or messages must be provided.")

        # Handle messages preprocessing
        if messages is not None:
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                messages = [messages]
            
            # Parallel processing of messages
            processed = self._process_messages_parallel(messages, max_workers=min(preprocess_workers, len(messages)))
            # Extract and pad
            pad_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id else 0
            
            input_ids_list = [p["input_ids"].squeeze(0) for p in processed if p["input_ids"] is not None]
            if input_ids_list:
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(self.device)
                attention_mask = pad_sequence(
                    [p["attention_mask"].squeeze(0) for p in processed if p["attention_mask"] is not None],
                    batch_first=True, padding_value=0
                ).to(self.device)
            
            pixel_values = self._pad_pixel_values([p["pixel_values"] for p in processed if p["pixel_values"] is not None])
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            pixel_attention_mask = self._pad_pixel_attention_mask(
                [p["pixel_attention_mask"] for p in processed if p["pixel_attention_mask"] is not None]
            )
            if pixel_attention_mask is not None:
                pixel_attention_mask = pixel_attention_mask.to(self.device)
            
            grids = [p["image_grid_thw"] for p in processed if p["image_grid_thw"] is not None]
            image_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None
            
            grids = [p["video_grid_thw"] for p in processed if p["video_grid_thw"] is not None]
            video_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None
            
            messages = [p["normalized"] for p in processed]
            
            # Tokenize plain texts
            plain_texts = [p["plain_text"] for p in processed]
            if self.tokenizer and any(t.strip() for t in plain_texts):
                text_inputs = self.tokenizer(plain_texts, add_special_tokens=False, padding=True, return_tensors="pt")
                text_input_ids = text_inputs["input_ids"].to(self.device)
                text_attention_mask = text_inputs["attention_mask"].to(self.device)
            else:
                text_input_ids = text_attention_mask = None
        else:
            text_input_ids = text_attention_mask = None

        # Encode text
        text_features = None
        text_mask = None
        if getattr(self.config, "use_text", True):
            if text_input_ids is not None:
                text_mask = text_attention_mask.bool()
                text_features = self._encode_text(text_input_ids, text_attention_mask)
            elif input_ids is not None:
                attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long)
                text_mask = attention_mask.bool()
                text_features = self._encode_text(input_ids, attention_mask)

        # Encode vision
        visual_features, visual_grids, visual_per_sample = [], [], []
        sample_count = len(messages) if messages is not None else (input_ids.shape[0] if input_ids is not None else 0)
        if sample_count > 0 and pixel_values is not None:
            pixel_values = self._ensure_pixel_values_5d(pixel_values)
            pixel_attention_mask = self._ensure_pixel_attention_mask_4d(pixel_attention_mask)
            image_hidden_states = self._get_image_features(pixel_values, pixel_attention_mask)

            if image_hidden_states is None:
                flat = pixel_values.view(-1, *pixel_values.shape[2:])
                image_hidden_states = self._encode_vision(flat)
            if image_hidden_states is not None:
                real_mask = self._real_image_mask(pixel_values)
                patch_size = getattr(self.config, "patch_size", None)
                if patch_size is None:
                    vision_config = getattr(getattr(self.smol_model, "config", None), "vision_config", None)
                    patch_size = getattr(vision_config, "patch_size", 16)
                offset = 0
                for i in range(sample_count):
                    count = int(real_mask[i].sum().item()) if i < real_mask.shape[0] else 0
                    h_w = self._sample_hw(pixel_values, i)
                    if count == 0 or h_w is None:
                        visual_per_sample.append(0)
                        continue
                    h, w = h_w
                    merge = int(getattr(self.config, "spatial_merge_size", 2))
                    grid_h = max(1, h // (patch_size * merge))
                    grid_w = max(1, w // (patch_size * merge))
                    sample_states = image_hidden_states[offset : offset + count]
                    is_video = messages is not None and self._sample_has_video(messages[i])
                    if is_video:
                        combined = sample_states.reshape(-1, sample_states.shape[-1])
                        visual_features.append(combined)
                        visual_grids.append(
                            torch.tensor([count, grid_h, grid_w], device=self.device, dtype=torch.long)
                        )
                        visual_per_sample.append(1)
                    else:
                        for k in range(count):
                            visual_features.append(sample_states[k])
                            visual_grids.append(
                                torch.tensor([1, grid_h, grid_w], device=self.device, dtype=torch.long)
                            )
                        visual_per_sample.append(count)
                    offset += count
        
        if not visual_features:
            return {
                "scales": None,
                "actions": None,
                "scale_mask": None,
                "log_probs": None,
                "multi_modal_data": [None] * sample_count,
                "scaled_messages": messages,
                "frame_metrics": None,
            }
        
        visual_grid_thw = torch.stack(visual_grids, dim=0).to(self.device)
        new_actions, scales, log_probs, new_scale_mask = self.allocator(
            visual_features=visual_features,
            visual_grid_thw=visual_grid_thw,
            text_features=text_features,
            text_mask=text_mask,
            actions=actions,
            visual_per_sample=visual_per_sample,
            eval_mode=eval_mode,
            scale_mask=scale_mask,
            compute_frame_metrics=kwargs.get("compute_frame_metrics", False) and compute_aux,
        )

        frame_metrics = {}

        if actions is not None:
            if compute_aux:
                contrastive_loss = self.allocator.get_contrastive_loss() if hasattr(self.allocator, "get_contrastive_loss") else getattr(self.allocator, "_last_contrastive_loss", None)
                sim_scale_loss = (
                    self.allocator.get_sim_scale_loss()
                    if hasattr(self.allocator, "get_sim_scale_loss")
                    else getattr(self.allocator, "_last_sim_scale_loss", None)
                )
                frame_metrics = self.allocator.get_frame_metrics()
                scale_var = getattr(self.allocator.scorer, "_last_scale_var", None)
                concentration_loss = self.allocator._last_concentration_loss
                entropy = getattr(self.allocator.scorer, "last_entropy", None)
            else:
                contrastive_loss = None
                sim_scale_loss = None
                frame_metrics = {}
                scale_var = None
                concentration_loss = None
                entropy = None
            return {
                "log_probs": log_probs,
                "contrastive_loss": contrastive_loss,
                "sim_scale_loss": sim_scale_loss,
                "frame_metrics": frame_metrics,
                "scale_var": scale_var,
                "concentration_loss": concentration_loss,
                "entropy": entropy,
            }

        if not return_mm_data:
            return {
                "scales": scales, 
                "actions": new_actions, 
                "scale_mask": new_scale_mask,
                "log_probs": log_probs, 
                # "frame_metrics": frame_metrics,
                # "entropy": getattr(self.allocator.scorer, "last_entropy", None),
            }

        return {
            "multi_modal_data": [None] * sample_count,
            "actions": new_actions,
            "log_probs": log_probs,
            "scale_mask": new_scale_mask,
            "scaled_messages": messages,
            "frame_metrics": frame_metrics,
        }

    def forward(self, *args, **kwargs):
        return self.scale_multi_modal(*args, **kwargs)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    # Matches aznet_smol_v1 + aznet_v1 / importance_allocator_v2 (see SmolAllocatorConfig).
    config = SmolAllocatorConfig(
        smol_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        vocab_size=49280,
        patch_size=16,
        spatial_merge_size=2,
        hidden_size=768,
        vision_hidden_size=576,
        llm_hidden_size=768,
        out_hidden_size=576,
        output_dim=1024,
        self_depth=2,
        cross_depth=1,
        dim_head=64,
        num_heads=16,
        max_frames=8,
        min_scale=0.2,
        max_scale=2.0,
        use_discrete_action=False,
        use_text=True,
        regression_head_mode="mlp",
        use_differentiable_importance=False,
        dropout=0.0,
        ff_mult=4,
        beta_add_one=False,
        beta_param_scale=0.5,
        beta_init_mode="uniform",
        gate_temperature=1.0,
        gate_query_scale=1.0,
        continuous_dist="beta",
        continuous_eval_quantile=0.5,
        logistic_normal_init_sigma=0.7,
        pool_gate_mode="no_ln",
        info_fuse_mode="pooled_ln",
        sim_scale_weight=0.0,
        sim_tau=0.6,
        sim_temp=0.15,
        sim_gamma=0.05,
    )
    model = SmolAllocatorForConditionalGeneration(config).to("cuda" if torch.cuda.is_available() else "cpu")
    # "YOUR_WORKSPACE_PATH/models/allocatorv2_sft",
    # model = SmolAllocatorForConditionalGeneration.from_pretrained(
    #     "YOUR_WORKSPACE_PATH/models/allocator_smol_init",
    #     config=config,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    count_params(model)

    from PIL import Image
    image = Image.open("YOUR_WORKSPACE_PATH/VisionThink/scissor.png")
    image1 = Image.open("YOUR_WORKSPACE_PATH/vlm_datasets/LLaVA-Pretrain/images/00000/000000010.jpg")
    
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "video", "video": "YOUR_WORKSPACE_PATH/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4", "max_frames": 8}, {"type": "text", "text": "Describe this video."}]}],
    ]

    messages_text = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    save_path = ""
    if save_path:
        model.save_pretrained(save_path, safe_serialization=True)
        import shutil
        import json
        shutil.copy(__file__, save_path)
        config_path = os.path.join(save_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            module_name = os.path.basename(__file__).replace(".py", "")
            auto_map_dict = {
                "AutoConfig": f"{module_name}.SmolAllocatorConfig",
                "AutoModel": f"{module_name}.SmolAllocatorForConditionalGeneration",
                "AutoModelForVision2Seq": f"{module_name}.SmolAllocatorForConditionalGeneration"
            }
            new_config = {"auto_map": auto_map_dict}
            new_config.update(config_data)
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)
    out = model.scale_multi_modal(messages=messages, return_mm_data=False, eval_mode=True)
    out_text = model.scale_multi_modal(messages=messages_text, return_mm_data=False, eval_mode=True)
    _ = (out, out_text)
