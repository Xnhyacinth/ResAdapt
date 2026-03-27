from transformers import PretrainedConfig


class SmolAllocatorConfig(PretrainedConfig):
    """
    Configuration class for the Smol allocator and RegressionHeadAllocatorSmol (v1/v2/v3).

    This configuration holds the hyperparameters for the multimodal allocator model.
    It reads fields that are mirrored by ``aznet_smol_v*`` and ``modeling_allocator_smol_*``.
    Additional keys from older saved configurations are accepted via ``**kwargs`` 
    and passed to the underlying ``PretrainedConfig``.

    Note:
        Training-time composite tags (such as ``scale_multi_modal_data``) are not stored
        in this class; they are managed separately. See ``resadapt.utils.scale_multi_modal_tags``
        and ``scripts/main.sh`` for details on composite tag resolution.

    Backbone dtype / attention:
        ``dtype`` or ``torch_dtype`` (e.g. ``"bfloat16"`` on Ampere+) is passed to
        ``AutoModelForImageTextToText.from_pretrained(dtype=...)``.
        ``attn_implementation`` is resolved by ``resadapt.allocator.attention_utils``:
        **prefer** ``flash_attention_2`` only when dtype is fp16/bf16; with ``"auto"`` or fp32 use ``sdpa``.
        Override with env ``ALLOCATOR_ATTN_IMPLEMENTATION`` (``flash_attention_2``, ``sdpa``, ``eager``).
    """

    model_type = "smol_allocator"

    def __init__(
        self,
        smol_model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        fps: float = 2.0,
        vocab_size: int = 49280,
        patch_size: int = 16,
        torch_dtype: str | None = None,
        dtype: str | None = None,
        _attn_implementation: str | None = None,
        spatial_merge_size: int = 2,
        hidden_size: int = 768,
        vision_hidden_size: int = 576,
        llm_hidden_size: int = 768,
        out_hidden_size: int = 576,
        output_dim: int = 1024,
        self_depth: int = 2,
        cross_depth: int = 1,
        dim_head: int = 64,
        num_heads: int = 16,
        max_frames: int = 8,
        min_scale: float = 0.2,
        max_scale: float = 2.0,
        use_discrete_action: bool = False,
        use_text: bool = True,
        use_differentiable_importance: bool = False,
        dropout: float = 0.0,
        ff_mult: int = 4,
        scale_bins: list | None = None,
        beta_add_one: bool = True,
        beta_init_mode: str = "uniform",
        regression_head_mode: str = "mlp",
        gate_mode: str = "gumbel_softmax",
        gate_k_ratio: float = 0.25,
        gate_temperature: float = 1.0,
        gate_query_scale: float = 1.0,
        allocator_arch: str = "framewise_v3",
        per_frame_concentration: bool = True,
        frame_refine_depth: int = 2,
        use_text_frame_cross_attn: bool = True,
        continuous_dist: str = "beta",
        continuous_eval_quantile: float = 0.5,
        beta_param_scale: float = 0.5,
        logistic_normal_init_sigma: float = 0.7,
        categorical_temperature: float = 1.0,
        pool_gate_mode: str = "patch_ln",
        info_fuse_mode: str = "pooled_ln",
        init_scale_mean: float = 1.0,
        init_concentration: float = 5.0,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        contrastive_margin: float = 0.0,
        sim_scale_weight: float = 0.15,
        sim_tau: float = 0.55,
        sim_temp: float = 0.12,
        sim_gamma: float = 0.05,
        temporal_mixer_depth: int = 1,
        temporal_use_pos: bool = True,
        dual_path_depth: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smol_model_name = smol_model_name
        self.fps = fps
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.torch_dtype = torch_dtype if torch_dtype is not None else dtype
        # Hugging Face deprecates config key ``torch_dtype`` in favor of ``dtype`` (same value).
        self.dtype = self.torch_dtype
        self._attn_implementation = _attn_implementation
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.out_hidden_size = out_hidden_size
        self.output_dim = output_dim
        self.self_depth = self_depth
        self.cross_depth = cross_depth
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.use_discrete_action = use_discrete_action
        self.use_text = use_text
        self.use_differentiable_importance = use_differentiable_importance
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.scale_bins = scale_bins
        self.beta_add_one = beta_add_one
        self.beta_init_mode = beta_init_mode
        self.regression_head_mode = regression_head_mode
        self.gate_mode = gate_mode
        self.gate_k_ratio = gate_k_ratio
        self.gate_temperature = gate_temperature
        self.gate_query_scale = gate_query_scale
        self.allocator_arch = allocator_arch
        self.per_frame_concentration = per_frame_concentration
        self.frame_refine_depth = frame_refine_depth
        self.use_text_frame_cross_attn = use_text_frame_cross_attn
        self.continuous_dist = continuous_dist
        self.continuous_eval_quantile = continuous_eval_quantile
        self.beta_param_scale = beta_param_scale
        self.logistic_normal_init_sigma = logistic_normal_init_sigma
        self.categorical_temperature = categorical_temperature
        self.pool_gate_mode = pool_gate_mode
        self.info_fuse_mode = info_fuse_mode
        self.init_scale_mean = init_scale_mean
        self.init_concentration = init_concentration
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_margin = contrastive_margin
        self.sim_scale_weight = sim_scale_weight
        self.sim_tau = sim_tau
        self.sim_temp = sim_temp
        self.sim_gamma = sim_gamma
        self.temporal_mixer_depth = temporal_mixer_depth
        self.temporal_use_pos = temporal_use_pos
        self.dual_path_depth = dual_path_depth

    def get(self, key, default=None):
        return getattr(self, key, default)


__all__ = ["SmolAllocatorConfig"]
