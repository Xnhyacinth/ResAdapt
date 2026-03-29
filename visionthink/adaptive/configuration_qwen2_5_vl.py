from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig, Qwen2_5_VLTextConfig


class UniQwenVisionConfig(Qwen2_5_VLVisionConfig):
    model_type = "uniqwen2_5_vl"

    def __init__(
        self,
        cross_depth=8,  
        self_depth=8,        
        tower_depth=8, 
        scale_bins = None,
        use_discrete_action = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_bins = scale_bins
        self.use_discrete_action = use_discrete_action
        self.cross_depth = cross_depth
        self.self_depth = self_depth
        if tower_depth is not None:
            self.tower_depth = tower_depth


class UniQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "uniqwen2_5_vl"
    
    sub_configs = {
        "vision_config": UniQwenVisionConfig, 
        "text_config": Qwen2_5_VLTextConfig
    }

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

__all__ = ["UniQwen2_5_VLConfig"]