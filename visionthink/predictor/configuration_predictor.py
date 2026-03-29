from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig, Qwen2_5_VLTextConfig


class PredictorConfig(Qwen2_5_VLVisionConfig):
    model_type = "predictor_qwen2_5_vl"
    
    def __init__(
        self,
        cross_depth=8,  
        self_depth=8,        
        tower_depth=8, 
        max_frames=4,
        scale_bins = None,
        use_discrete_action = False,
        use_text = True,
        text_embed_dim: int | None = None,
        use_text_encoder: bool = False,
        text_encoder_depth: int = 0,
        text_encoder_heads: int | None = None,
        text_encoder_dropout: float = 0.0,
        text_encoder_ff_mult: int = 4,
        use_cross_attn_gate: bool = True,
        vision_start_token_id = 151652,
        vision_end_token_id = 151653,
        vision_token_id = 151654,
        gate_temperature: float = 1.0,
        gate_query_scale: float = 1.0,
        vlp_mode: str = "mlp",
        mlp_mode: str = "mlp",
        regression_head_mode: str = "mlp",
        beta_param_scale: float = 1.0,
        beta_add_one: bool = True,
        continuous_dist: str = "beta",
        continuous_eval_quantile: float = 0.5,
        logistic_normal_init_sigma: float = 0.7,
        pool_gate_mode: str = "no_ln",
        info_fuse_mode: str = "pooled_ln",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_bins = scale_bins
        self.use_discrete_action = use_discrete_action
        self.cross_depth = cross_depth
        self.self_depth = self_depth
        self.use_text = use_text
        self.max_frames = max_frames
        self.text_embed_dim = text_embed_dim
        self.use_text_encoder = use_text_encoder
        self.text_encoder_depth = text_encoder_depth
        self.text_encoder_heads = text_encoder_heads
        self.text_encoder_dropout = text_encoder_dropout
        self.text_encoder_ff_mult = text_encoder_ff_mult
        self.use_cross_attn_gate = use_cross_attn_gate
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_token_id = vision_token_id
        self.gate_temperature = gate_temperature
        self.gate_query_scale = gate_query_scale
        self.vlp_mode = vlp_mode
        self.mlp_mode = mlp_mode
        self.regression_head_mode = regression_head_mode
        self.beta_param_scale = beta_param_scale
        self.beta_add_one = beta_add_one
        self.continuous_dist = continuous_dist
        self.continuous_eval_quantile = continuous_eval_quantile
        self.logistic_normal_init_sigma = logistic_normal_init_sigma
        self.pool_gate_mode = pool_gate_mode
        self.info_fuse_mode = info_fuse_mode
        if tower_depth is not None:
            self.tower_depth = tower_depth

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config from pretrained, handling missing V2 params gracefully."""
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Set V2 defaults if not present (for backward compatibility)
        v2_defaults = {
            "predictor_version": "v1",
            "gate_mode": "gumbel_softmax",
            "gate_k_ratio": 0.25,
            "contrastive_weight": 0.1,
            "contrastive_temperature": 0.1,
            "sim_scale_weight": 0.0,
            "sim_tau": 0.5,
            "sim_temp": 0.1,
            "sim_gamma": 0.05,
        }

        for key, default_value in v2_defaults.items():
            if not hasattr(config, key):
                setattr(config, key, default_value)

        return config


# class UniQwen2_5_VLConfig(Qwen2_5_VLConfig):
#     model_type = "uniqwen2_5_vl"
    
#     sub_configs = {
#         "vision_config": UniQwenVisionConfig, 
#         "text_config": Qwen2_5_VLTextConfig
#     }

#     def __init__(
#         self,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

__all__ = ["PredictorConfig"]
