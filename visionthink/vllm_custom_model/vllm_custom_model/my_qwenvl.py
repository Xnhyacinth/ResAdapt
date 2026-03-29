import torch.nn as nn
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config import VllmConfig

from visionthink.adaptive.modeling_qwenvl import RegressionHeadPredictor, UniQwenVisionTransformer
from transformers import AutoProcessor, AutoTokenizer


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)

class UniQwenForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    vLLM implementation of UniQwen with adaptive multi-modal scaling.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config

        # Vision config defaults
        config.vision_config.min_scale = getattr(config.vision_config, "min_scale", 0.25)
        config.vision_config.max_scale = getattr(config.vision_config, "max_scale", 3.0)
        config.vision_config.cross_depth = getattr(config.vision_config, "cross_depth", 8)
        config.vision_config.self_depth = getattr(config.vision_config, "self_depth", 8)
        config.vision_config.tower_depth = getattr(config.vision_config, "tower_depth", 8)

        # Initialize vision tower and predictor
        self.vision_tower = UniQwenVisionTransformer(config.vision_config)
        self.predictor = RegressionHeadPredictor(config)

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


    def load_weights(self, weights):
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

        # self.predictor.load_pretrained_layers(
        #     predictor_weights,  
        #     src_cross_depth=8,
        #     src_self_depth=8
        # )

        # for name, loaded_weight in vision_tower_weights:
        #     param = self.state_dict().get(name)
        #     if param is not None:
        #         param.data.copy_(loaded_weight)
        #     else:
        #         print(f"Warning: Skipping unexpected weight {name}")

        # src_cross_indices = set()
        # src_self_indices = set()
        
        # for name, _ in predictor_weights_raw:
        #     if "layers." in name:
        #         try:
        #             parts = name.split(".")
        #             if "layers" in parts:
        #                 idx_loc = parts.index("layers") + 1
        #                 idx = int(parts[idx_loc])
                        
        #                 if "cross_attn" in name:
        #                     src_cross_indices.add(idx)
        #                 elif "temporal_attn" in name or "spatial_attn" in name:
        #                     src_self_indices.add(idx)
        #         except:
        #             continue

        # src_cross_depth = len(src_cross_indices) if src_cross_indices else 8
        # src_self_depth = len(src_self_indices) if src_self_indices else 8
        # tgt_cross_depth, tgt_self_depth = 2, 2

        # if src_self_depth != tgt_self_depth or src_cross_depth != tgt_cross_depth:
        #     predictor_weights_mapped = []
        #     for name, tensor in predictor_weights_raw:
        #         new_name = name
        #         should_load = True
                
        #         if "layers." in name:
        #             parts = name.split(".")
        #             idx_loc = parts.index("layers") + 1
        #             src_idx = int(parts[idx_loc])
                    
        #             tgt_idx = None
        #             # Src: 0, 1, 2... -> Tgt: 0, 1, 2...
        #             if src_idx < src_cross_depth:
        #                 if src_idx < tgt_cross_depth:
        #                     tgt_idx = src_idx
        #                 else:
        #                     should_load = False

        #             # Src: 8, 9, 10... -> Tgt: 2, 3, 4...
        #             elif src_idx >= src_cross_depth:
        #                 self_layer_offset = src_idx - src_cross_depth
                        
        #                 if self_layer_offset < tgt_self_depth:
        #                     # Tgt Start Index = tgt_cross_depth
        #                     tgt_idx = tgt_cross_depth + self_layer_offset
        #                 else:
        #                     should_load = False
                    
        #             if should_load and tgt_idx is not None:
        #                 if src_idx != tgt_idx:
        #                     old_str = f"layers.{src_idx}."
        #                     new_str = f"layers.{tgt_idx}."
        #                     new_name = name.replace(old_str, new_str)
        #                     # print(f"  Mapping: {name} -> {new_name}")
                
        #         if should_load:
        #             predictor_weights_mapped.append((new_name, tensor))
        # else:
        #     predictor_weights_mapped = predictor_weights_raw

        # for name, loaded_weight in predictor_weights_mapped:
        #     param = self.state_dict().get(name)
        #     if param is not None:
        #         param.data.copy_(loaded_weight)
        #     else:
        #         print(f"Warning: Skipping unexpected weight {name}")

