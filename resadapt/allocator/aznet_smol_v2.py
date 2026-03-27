import inspect
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import ModuleUtilsMixin

from resadapt.allocator.aznet_v2 import FrameWiseScaleAllocator
from resadapt.allocator.importance_allocator_v2 import DifferentiableImportanceAllocator


class RegressionHeadAllocatorSmol(ModuleUtilsMixin, nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        llm_hidden_size = int(getattr(vision_config, "llm_hidden_size", 768))
        out_hidden_size = int(getattr(vision_config, "out_hidden_size", llm_hidden_size))
        output_dim = int(getattr(vision_config, "output_dim", out_hidden_size))

        if out_hidden_size == output_dim:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Linear(out_hidden_size, output_dim, bias=False)

        vision_hidden_size = int(getattr(vision_config, "vision_hidden_size", vision_config.hidden_size))
        if vision_hidden_size == output_dim:
            self.vlp = nn.Identity()
        else:
            self.vlp = nn.Linear(vision_hidden_size, output_dim, bias=False)

        self.spatial_merge_size = vision_config.spatial_merge_size

        use_differentiable = bool(getattr(vision_config, "use_differentiable_importance", False))
        scorer_cls = DifferentiableImportanceAllocator if use_differentiable else FrameWiseScaleAllocator
        # Keys must match FrameWiseScaleAllocator (aznet_v2) or DifferentiableImportanceAllocator __init__.
        # v2 FrameWise has no cross_attn_depth; DifferentiableImportanceAllocator does.
        scorer_kwargs = dict(
            dim=output_dim,
            depth=getattr(vision_config, "self_depth", 2),
            dim_head=getattr(vision_config, "dim_head", 64),
            heads=getattr(vision_config, "num_heads", 8),
            max_frames=getattr(vision_config, "max_frames", 32),
            spatial_merge_size=vision_config.spatial_merge_size,
            use_discrete_action=getattr(vision_config, "use_discrete_action", False),
            scale_bins=getattr(vision_config, "scale_bins", None),
            min_scale=getattr(vision_config, "min_scale", 0.2),
            max_scale=getattr(vision_config, "max_scale", 2.0),
            dropout=getattr(vision_config, "dropout", 0.0),
            ff_mult=getattr(vision_config, "ff_mult", 4),
            beta_add_one=getattr(vision_config, "beta_add_one", True),
            beta_init_mode=getattr(vision_config, "beta_init_mode", "uniform"),
            beta_param_scale=getattr(vision_config, "beta_param_scale", 1.0),
            gate_temperature=getattr(vision_config, "gate_temperature", 1.0),
            gate_query_scale=getattr(vision_config, "gate_query_scale", 1.0),
            regression_head_mode=getattr(vision_config, "regression_head_mode", "mlp"),
            continuous_dist=getattr(vision_config, "continuous_dist", "beta"),
            continuous_eval_quantile=getattr(vision_config, "continuous_eval_quantile", 0.5),
            logistic_normal_init_sigma=getattr(vision_config, "logistic_normal_init_sigma", 0.7),
            categorical_temperature=getattr(vision_config, "categorical_temperature", 1.0),
            pool_gate_mode=getattr(vision_config, "pool_gate_mode", "no_ln"),
            info_fuse_mode=getattr(vision_config, "info_fuse_mode", "pooled_ln"),
            sim_scale_weight=getattr(vision_config, "sim_scale_weight", 0.1),
            sim_tau=getattr(vision_config, "sim_tau", 0.5),
            sim_temp=getattr(vision_config, "sim_temp", 0.1),
            sim_gamma=getattr(vision_config, "sim_gamma", 0.05),
        )
        if use_differentiable:
            scorer_kwargs["cross_attn_depth"] = getattr(vision_config, "cross_depth", 1)
            scorer_kwargs.update(
                gate_mode=getattr(vision_config, "gate_mode", "gumbel_softmax"),
                gate_k_ratio=getattr(vision_config, "gate_k_ratio", 0.25),
                contrastive_weight=getattr(vision_config, "contrastive_weight", 0.1),
                contrastive_temperature=getattr(vision_config, "contrastive_temperature", 0.1),
                contrastive_margin=getattr(vision_config, "contrastive_margin", 0.0),
                temporal_mixer_depth=getattr(vision_config, "temporal_mixer_depth", 1),
                temporal_use_pos=getattr(vision_config, "temporal_use_pos", True),
                dual_path_depth=getattr(vision_config, "dual_path_depth", 1),
            )
        valid_keys = set(inspect.signature(scorer_cls.__init__).parameters.keys())
        valid_keys.discard("self")
        scorer_kwargs = {k: v for k, v in scorer_kwargs.items() if k in valid_keys}
        self.scorer = scorer_cls(**scorer_kwargs)
        self.out_hidden_size = out_hidden_size
        self.output_dim = output_dim
        if hasattr(self.scorer, "post_init") and callable(self.scorer.post_init):
            self.scorer.post_init()

    def forward(
        self,
        visual_features,
        visual_grid_thw,
        text_features=None,
        visual_mask=None,
        text_mask=None,
        actions=None,
        visual_per_sample=None,
        eval_mode=None,
        scale_mask=None,
        compute_frame_metrics: bool = False,
    ):
        if text_features is not None:
            if int(text_features.shape[-1]) == self.out_hidden_size:
                if isinstance(self.mlp, nn.Identity):
                    text_out = text_features
                else:
                    text_out = self.mlp(text_features.to(dtype=self.mlp.weight.dtype))
            elif int(text_features.shape[-1]) == self.output_dim:
                text_out = text_features
            else:
                raise ValueError(f"Unsupported text_features dim: {int(text_features.shape[-1])}")
            projected_text = text_out
        else:
            projected_text = None

        B_vis = len(visual_features)
        if projected_text is not None:
            B_txt = projected_text.shape[0]
            if B_vis != B_txt:
                if visual_per_sample is not None:
                    device = projected_text.device
                    if not isinstance(visual_per_sample, torch.Tensor):
                        repeats = torch.tensor(visual_per_sample, device=device)
                    else:
                        repeats = visual_per_sample.to(device)
                else:
                    if B_txt == 0:
                        repeats = torch.tensor([], device=projected_text.device, dtype=torch.long)
                    else:
                        assert B_vis % B_txt == 0
                        k = B_vis // B_txt
                        repeats = torch.full((B_txt,), k, device=projected_text.device, dtype=torch.long)
                projected_text = projected_text.repeat_interleave(repeats, dim=0)
                if text_mask is not None:
                    text_mask = text_mask.repeat_interleave(repeats, dim=0)

        visual_batch = pad_sequence(visual_features, batch_first=True)
        if isinstance(self.vlp, nn.Identity):
            visual_batch_proj = self.vlp(visual_batch)
        else:
            visual_batch_proj = self.vlp(visual_batch.to(dtype=self.vlp.weight.dtype))

        return self.scorer(
            visual_features_batch=visual_batch_proj,
            visual_grid_thw=visual_grid_thw,
            visual_mask=visual_mask,
            text_features=projected_text,
            text_mask=text_mask,
            actions=actions,
            eval_mode=eval_mode,
            visual_per_sample=visual_per_sample,
            scale_mask=scale_mask,
            compute_frame_metrics=compute_frame_metrics,
        )

    def get_frame_metrics(self) -> dict:
        return self.scorer.get_frame_metrics()

    def get_contrastive_loss(self):
        if hasattr(self.scorer, "get_contrastive_loss") and callable(self.scorer.get_contrastive_loss):
            return self.scorer.get_contrastive_loss()
        return torch.tensor(0.0, device=self.scorer.last_scales.device) if self.scorer.last_scales is not None else None

    def get_weighted_contrastive_loss(self):
        return self.get_contrastive_loss()

    @property
    def _last_contrastive_loss(self):
        return self.get_contrastive_loss()

    @property
    def _last_sim_scale_loss(self):
        return getattr(self.scorer, "_last_sim_scale_loss", None)

    @property
    def _last_concentration_loss(self):
        return getattr(self.scorer, "_last_concentration_loss", None)

    @property
    def last_scales(self):
        return getattr(self.scorer, "last_scales", None)

    @property
    def last_frame_features(self):
        return getattr(self.scorer, "last_frame_features", None)

    @property
    def _last_scale_var(self):
        return getattr(self.scorer, "_last_scale_var", None)

    @property
    def last_entropy(self):
        return getattr(self.scorer, "last_entropy", None)
