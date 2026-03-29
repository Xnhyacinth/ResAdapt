from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)

from functools import lru_cache, partial
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config import VllmConfig

from visionthink.adaptive.modeling_qwenvl import RegressionHeadPredictor, UniQwenVisionTransformer
from transformers import AutoProcessor, AutoTokenizer, AutoModel

from vllm.multimodal.evs import (compute_mrope_for_media,
                                 compute_retained_tokens_count,
                                 compute_retention_mask,
                                 recompute_mrope_positions)

import torch
from transformers.image_transforms import convert_to_rgb
import os
from collections.abc import (Callable, Generator, ItemsView, Iterable, Mapping, Sequence)
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                     MultiModalFieldConfig, MultiModalInputs,
                     MultiModalKwargsItem, MultiModalKwargsItems,
                     MultiModalKwargsOptionalItems, MultiModalUUIDDict,
                     PlaceholderRange)

class PredictorMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        predictor_path = os.getenv("PREDICTOR_PATH", None)

        if predictor_path is None:
            raise ValueError("PREDICTOR_PATH is not set")
        self.predictor = AutoModel.from_pretrained(
            predictor_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        self.predictor.eval()


    def _apply_hf_processor_main(
        self,
        prompt,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
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
        images = processor_data.get("images", None)
        videos = processor_data.get("videos", None)
        if images is not None:
            images = [convert_to_rgb(image) for image in images]
            inputs = self.predictor.processor(
                text=[prompt],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            ).to(self.predictor.device)
            inputs.update({"multi_modal_data": [{"image": images}], "text": [prompt], "eval_mode": True})
            scaled_images = self.predictor(**inputs)
            mm_items = self._to_mm_items({"image": scaled_images})

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

        return prompt_ids, mm_processed_data, False

    
@MULTIMODAL_REGISTRY.register_processor(
    PredictorMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)

class PredictorForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    vLLM implementation of UniQwen with adaptive multi-modal scaling.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # config = self.config

        # if not hasattr(config.vision_config, 'min_scale'):
        #     config.vision_config.min_scale = 0.25
        #     config.vision_config.max_scale = 3.0

        # if not hasattr(config.vision_config, "cross_depth"):
        #     print(f"init the cross_depth!!")
        #     config.vision_config.cross_depth = 8
        #     config.vision_config.self_depth = 8
        #     config.vision_config.tower_depth = 8

        # # # 2. Add Regression Head Predictor
        # self.predictor = RegressionHeadPredictor(config.vision_config)

        # # 3. Handle Vision Tower Alias
        # # self.vision_tower = self.visual 
        # self.vision_tower = UniQwenVisionTransformer(config.vision_config)

        # # 4. Initialize Processor & Tokenizer (Transformers versions)
        # try:
        #     self.processor = AutoProcessor.from_pretrained(
        #         config._name_or_path if hasattr(config, "_name_or_path") else "Qwen/Qwen2.5-VL-7B-Instruct",
        #         trust_remote_code=True
        #     )
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         config._name_or_path if hasattr(config, "_name_or_path") else "Qwen/Qwen2.5-VL-7B-Instruct",
        #         trust_remote_code=True
        #     )
        # except Exception as e:
        #     print(f"Warning: Failed to load processor inside model: {e}")
        #     self.processor = None
        #     self.tokenizer = None

        # self.image_token_index = getattr(config, "image_token_id", 151655) # Default Qwen2.5-VL image token
        # self.video_token_index = getattr(config, "video_token_id", 151656)



    def load_weights00(self, weights):
        base_weights_list = []
        predictor_weights_dict = {}
        vision_tower_weights_dict = {} 

        pred_prefix = "predictor."  # Prefix for predictor module weights
        vis_prefix = "vision_tower."  # Prefix for vision tower weights

        # print("[LoadWeights] Splitting incoming weights stream...")
        
        for name, tensor in weights:
            if pred_prefix in name:
                # predictor_weights_dict[name] = tensor
                rel_name = name.split(pred_prefix, 1)[1]
                predictor_weights_dict[rel_name] = tensor

            elif vis_prefix in name:
                # vision_tower_weights_dict[name] = tensor
                rel_name = name.split(vis_prefix, 1)[1]
                vision_tower_weights_dict[rel_name] = tensor
            else:
                base_weights_list.append((name, tensor))
        
        autoloaded_weights = super().load_weights(base_weights_list)

        if hasattr(self, 'vision_tower') and vision_tower_weights_dict:
            # print(f"[Load Vision] Loading {len(vision_tower_weights_dict)} tensors...")
            self.vision_tower.load_state_dict(vision_tower_weights_dict, strict=False)
            for rel_name in vision_tower_weights_dict.keys():
                full_name = f"vision_tower.{rel_name}" 
                autoloaded_weights.add(full_name)

        if hasattr(self, 'predictor') and predictor_weights_dict:
            # print(f"[Load Predictor] Loading {len(predictor_weights_dict)} tensors...")
            self.predictor.load_state_dict(predictor_weights_dict, strict=False)
            for rel_name in predictor_weights_dict.keys():
                full_name = f"predictor.{rel_name}"
                autoloaded_weights.add(full_name)

        return autoloaded_weights

        