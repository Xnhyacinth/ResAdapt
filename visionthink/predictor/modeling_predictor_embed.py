import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, AutoModelForImageTextToText, PreTrainedModel
from concurrent.futures import ThreadPoolExecutor

from visionthink.predictor.configuration_predictor_smol import SmolPredictorConfig
from visionthink.predictor.AZNetv3 import RegressionHeadPredictor


class SmolEmbedPredictorForConditionalGeneration(PreTrainedModel):
    config_class = SmolPredictorConfig

    supports_gradient_checkpointing = True
    # _no_split_modules = ["SmolVLMVisionAttention", "SmolVLMDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config: SmolPredictorConfig):
        super().__init__(config)
        self.config = config
        model_name = getattr(config, "smol_model_name", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"set predictor min_scale: {config.min_scale}")
        print(f"set predictor max_scale: {config.max_scale}")
        print(f"set predictor max_frames: {config.max_frames}")

        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            self.processor.video_processor.num_frames = int(getattr(config, "max_frames", 16))
            self.processor.video_processor.fps = int(getattr(config, "fps", 2.0))
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            self.processor.image_processor.do_image_splitting = False
            
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        smol_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=getattr(config, "dtype", "auto"),
            _attn_implementation=getattr(config, "_attn_implementation", "flash_attention_2"),
        )
        self.text_embedding = smol_model.get_input_embeddings()
        self.vision_model = self._resolve_vision_model_from(smol_model.model)
        self._sync_config_dims()
        self.smol_model = None
        self.predictor = RegressionHeadPredictor(config)

    def _sync_config_dims(self):
        text_embed = self.text_embedding
        text_dim = int(getattr(text_embed, "embedding_dim", 0))
        if text_dim > 0:
            self.config.text_embed_dim = text_dim
            self.config.llm_hidden_size = text_dim
            self.config.out_hidden_size = getattr(self.config, "out_hidden_size", text_dim)
            self.config.output_dim = getattr(self.config, "output_dim", text_dim)
        vision_model = self._resolve_vision_model()
        if vision_model is not None:
            vision_embed = vision_model.get_input_embeddings()
            vision_dim = int(getattr(vision_embed, "embedding_dim", getattr(vision_model.config, "hidden_size", 0)))
            if vision_dim > 0:
                self.config.hidden_size = getattr(self.config, "hidden_size", vision_dim)

    def _resolve_vision_model(self):
        if getattr(self, "vision_model", None) is not None:
            return self.vision_model
        return None

    @staticmethod
    def _resolve_vision_model_from(model):
        for attr in ["vision_model", "vision_tower", "vision_encoder", "vision_module"]:
            m = getattr(model, attr, None)
            if m is not None:
                return m
        m = getattr(model, "get_vision_model", None)
        if callable(m):
            return m()
        return None

    def _normalize_content(self, content):
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

    def _normalize_message(self, msg):
        if isinstance(msg, tuple):
            msg = msg[0]
        if isinstance(msg, list):
            return [self._normalize_message(m) for m in msg]
        content = self._normalize_content(msg.get("content", ""))
        if isinstance(content, list):
            texts = [it for it in content if isinstance(it, dict) and it.get("type") in (None, "text")]
            medias = [it for it in content if isinstance(it, dict) and it.get("type") in {"image", "video"}]
            content = texts + medias
        return {**msg, "content": content}

    def _apply_chat_template(self, message):
        inputs = self.processor.apply_chat_template(
            [message],
            num_frames=self.config.max_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    def _process_single_message(self, msg):
        normalized = self._normalize_message(msg)
        inputs = self._apply_chat_template(normalized)
        return {
            "normalized": normalized,
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "pixel_attention_mask": inputs.get("pixel_attention_mask"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }

    def _process_messages_parallel(self, messages, max_workers=None):
        if len(messages) <= 1 or max_workers == 1:
            return [self._process_single_message(m) for m in messages]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._process_single_message, messages))

    def _pad_pixel_values(self, pixel_values_list):
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

    def _pad_pixel_attention_mask(self, pixel_attention_mask_list):
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

    def _build_patch_attention_mask(self, pixel_values, patch_size):
        bsz = pixel_values.size(0)
        h = pixel_values.size(2) // patch_size
        w = pixel_values.size(3) // patch_size
        return torch.ones((bsz, h, w), dtype=torch.bool, device=pixel_values.device)

    def _merge_patches(self, embeds, grid_thw):
        t = int(grid_thw[0].item())
        h = int(grid_thw[1].item())
        w = int(grid_thw[2].item())
        l = t * h * w
        if embeds.shape[0] < l:
            l = embeds.shape[0]
        embeds = embeds[:l]
        d = embeds.shape[-1]
        if t <= 0 or h <= 0 or w <= 0:
            return embeds, torch.tensor([1, 1, 1], device=embeds.device, dtype=torch.long)
        x = embeds.view(t, h, w, d)
        merge = int(getattr(self.config, "spatial_merge_size", 2))
        if merge <= 1:
            return x.view(t, h * w, d), torch.tensor([t, h, w], device=embeds.device, dtype=torch.long)
        h_trim = (h // merge) * merge
        w_trim = (w // merge) * merge
        if h_trim <= 0 or w_trim <= 0:
            return x.view(t, h * w, d), torch.tensor([t, h, w], device=embeds.device, dtype=torch.long)
        x = x[:, :h_trim, :w_trim, :]
        x = x.view(t, h_trim // merge, merge, w_trim // merge, merge, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(t, (h_trim // merge) * (w_trim // merge), d * merge * merge)
        return x, torch.tensor([t, h_trim // merge, w_trim // merge], device=embeds.device, dtype=torch.long)

    def _sample_hw(self, pixel_values, sample_index):
        if pixel_values is None:
            return None
        h, w = pixel_values[sample_index].shape[-2:]
        return h, w

    @staticmethod
    def _real_image_mask(pixel_values):
        if pixel_values is None:
            return None
        if pixel_values.dim() == 4:
            return torch.ones((pixel_values.shape[0], 1), device=pixel_values.device, dtype=torch.bool)
        b, t = pixel_values.shape[:2]
        flat = pixel_values.view(b * t, *pixel_values.shape[2:])
        nb_values = flat.shape[1:].numel()
        is_real = (flat == 0.0).sum(dim=(-1, -2, -3)) != nb_values
        return is_real.view(b, t)

    @staticmethod
    def _ensure_pixel_attention_mask_4d(pixel_attention_mask):
        if pixel_attention_mask is None:
            return None
        if pixel_attention_mask.dim() == 4:
            return pixel_attention_mask
        if pixel_attention_mask.dim() == 3:
            return pixel_attention_mask.unsqueeze(1)
        raise ValueError("pixel_attention_mask must be 3D or 4D.")

    def scale_multi_modal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.LongTensor = None,
        messages=None,
        actions: torch.Tensor = None,
        scale_mask: torch.Tensor = None,
        **kwargs,
    ):
        eval_mode = bool(kwargs.get("eval_mode", False))
        compute_aux = actions is not None and eval_mode is not True
        return_mm_data = kwargs.get("return_mm_data", False)
        preprocess_workers = kwargs.get("preprocess_workers", 16)

        if input_ids is None and messages is None:
            raise ValueError("Either input_ids or messages must be provided.")

        processed = None
        if messages is not None:
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                messages = [messages]
            processed = self._process_messages_parallel(messages, max_workers=min(preprocess_workers, len(messages)))
            pad_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id else 0
            input_ids_list = [p["input_ids"].squeeze(0) for p in processed if p["input_ids"] is not None]
            if input_ids_list:
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(self.device)
                attention_mask = pad_sequence(
                    [p["attention_mask"].squeeze(0) for p in processed if p["attention_mask"] is not None],
                    batch_first=True,
                    padding_value=0,
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
        else:
            pixel_attention_mask = None

        if input_ids is None:
            raise ValueError("Missing input_ids after preprocessing.")

        if self.text_embedding is None:
            raise ValueError("Text embedding module not initialized.")
        text_features = self.text_embedding(input_ids)
        text_mask = attention_mask.bool() if attention_mask is not None else None

        if self.vision_model is None:
            raise ValueError("SmolVLM vision model not found.")
        vision_config = getattr(self.vision_model, "config", None) if self.vision_model is not None else None
        patch_size = int(getattr(vision_config, "patch_size", getattr(self.config, "patch_size", 16)))
        vision_embeddings = self.vision_model.get_input_embeddings()

        all_image_embeds = []
        all_image_grids = []
        if pixel_values is not None:
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            if pixel_values.dim() != 5:
                raise ValueError("pixel_values must be 4D or 5D for SmolVLM image embeddings.")
            pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
            pixel_attention_mask = self._ensure_pixel_attention_mask_4d(pixel_attention_mask)
            patch_mask = self._build_patch_attention_mask(pixel_values.view(-1, *pixel_values.shape[2:]), patch_size)
            img_embeds = vision_embeddings(
                pixel_values=pixel_values.view(-1, *pixel_values.shape[2:]),
                patch_attention_mask=patch_mask,
            )
            img_embeds = img_embeds.view(pixel_values.shape[0], pixel_values.shape[1], img_embeds.shape[1], img_embeds.shape[2])
            real_mask = self._real_image_mask(pixel_values)
            patch_h = patch_mask.shape[1]
            patch_w = patch_mask.shape[2]
            for i in range(pixel_values.shape[0]):
                count = int(real_mask[i].sum().item()) if real_mask is not None else int(pixel_values.shape[1])
                h_w = self._sample_hw(pixel_values, i)
                if count == 0 or h_w is None:
                    continue
                h, w = h_w
                merge = int(getattr(self.config, "spatial_merge_size", 2))
                grid_h = max(1, h // (patch_size * merge))
                grid_w = max(1, w // (patch_size * merge))
                sample_states = img_embeds[i, :count]
                for k in range(count):
                    merged, grid = self._merge_patches(sample_states[k], torch.tensor([1, patch_h, patch_w], device=self.device, dtype=torch.long))
                    all_image_embeds.append(merged.reshape(-1, merged.shape[-1]))
                    all_image_grids.append(torch.tensor([1, grid_h, grid_w], device=self.device, dtype=torch.long))

        visual_features = []
        visual_grids = []
        visual_per_sample = []
        if processed is not None:
            offset = 0
            sample_count = len(processed)
            real_mask = self._real_image_mask(pixel_values)
            for i in range(sample_count):
                count = int(real_mask[i].sum().item()) if real_mask is not None else 0
                if count <= 0:
                    visual_per_sample.append(0)
                    continue
                visual_features.extend(all_image_embeds[offset:offset + count])
                visual_grids.extend(all_image_grids[offset:offset + count])
                visual_per_sample.append(count)
                offset += count
        else:
            visual_features.extend(all_image_embeds)
            visual_grids.extend(all_image_grids)
            visual_per_sample = [len(visual_features)]

        if not visual_features:
            return {
                "scales": None,
                "actions": None,
                "scale_mask": None,
                "log_probs": None,
                "multi_modal_data": [None] * (len(visual_per_sample)),
                "scaled_messages": messages,
                "frame_metrics": None,
            }

        visual_grid_thw = torch.stack(visual_grids, dim=0).to(self.device)

        new_actions, scales, log_probs, new_scale_mask = self.predictor(
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

        frame_metrics = self.predictor.get_frame_metrics()

        if actions is not None:
            if compute_aux:
                contrastive_loss = self.predictor.get_contrastive_loss() if hasattr(self.predictor, "get_contrastive_loss") else getattr(self.predictor, "_last_contrastive_loss", None)
                sim_scale_loss = (
                    self.predictor.get_sim_scale_loss()
                    if hasattr(self.predictor, "get_sim_scale_loss")
                    else getattr(self.predictor, "_last_sim_scale_loss", None)
                )
                frame_metrics = self.predictor.get_frame_metrics()
                scale_var = getattr(self.predictor.scorer, "_last_scale_var", None)
                concentration_loss = self.predictor._last_concentration_loss
                entropy = getattr(self.predictor.scorer, "last_entropy", None)
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
            }

        return {
            "multi_modal_data": [None] * (len(visual_per_sample)),
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
    from PIL import Image
    import numpy as np
    config = SmolPredictorConfig(
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
        continuous_dist="beta",
        beta_param_scale=0.5,
        beta_add_one=False,
        sim_scale_weight=0.2,
        sim_tau=0.5,
        sim_temp=0.1,
        sim_gamma=0.05,
        use_text=True,
        use_text_encoder=True,
        text_encoder_depth=2,
        text_encoder_ff_mult=2,
        text_encoder_heads=8,
        text_encoder_dropout=0.0,
        regression_head_mode="mlp",
        use_cross_attn_gate=True,
    )
    # model = SmolEmbedPredictorForConditionalGeneration(config).to("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_embed",
        # config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    total, trainable = count_params(model)
    print(f"Total params: {total/1e9:.3f} B")
    print(f"Trainable params: {trainable/1e9:.3f} B")
    image = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/VisionThink/scissor.png")
    image1 = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Pretrain/images/00000/000000010.jpg")
    
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "video", "video": "/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4", "max_frames": 8}, {"type": "text", "text": "Describe this video."}]}],
    ]

    messages_text = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    out = model.scale_multi_modal(messages=messages, return_mm_data=False, eval_mode=True)
    out_text = model.scale_multi_modal(messages=messages_text, return_mm_data=False, eval_mode=True)
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in out.items()})
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in out_text.items()})

    #  save_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_embed"
    # model.save_pretrained(save_path, safe_serialization=True)
    breakpoint()
