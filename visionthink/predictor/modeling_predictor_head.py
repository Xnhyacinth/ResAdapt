import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText, PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin
from torch.nn.utils.rnn import pad_sequence

from visionthink.predictor.configuration_predictor_smol import SmolPredictorConfig
from visionthink.predictor.AZNetv3 import ProjectorBlock


class HeadOnlyPredictor(ModuleUtilsMixin, nn.Module):
    def __init__(self, vision_config: SmolPredictorConfig):
        super().__init__()
        dim = int(getattr(vision_config, "output_dim", getattr(vision_config, "out_hidden_size", 768)))
        vision_hidden_size = int(getattr(vision_config, "vision_hidden_size", vision_config.hidden_size))
        self.spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        self.use_discrete_action = bool(getattr(vision_config, "use_discrete_action", False))
        self.min_scale = float(getattr(vision_config, "min_scale", 0.2))
        self.max_scale = float(getattr(vision_config, "max_scale", 2.0))
        self.beta_add_one = bool(getattr(vision_config, "beta_add_one", True))
        self.beta_param_scale = float(getattr(vision_config, "beta_param_scale", 1.0))
        self.continuous_dist = str(getattr(vision_config, "continuous_dist", "beta"))
        self.continuous_eval_quantile = float(getattr(vision_config, "continuous_eval_quantile", 0.5))
        self.force_uniform_ab = bool(getattr(vision_config, "force_uniform_ab", False))
        self.init_head = bool(getattr(vision_config, "init_head", True))
        self.init_scale_mean = float(getattr(vision_config, "init_scale_mean", 1.0))
        self.init_concentration = float(getattr(vision_config, "init_concentration", 4.0))
        self.init_weight_std = float(getattr(vision_config, "init_weight_std", 1e-3))
        self.contrastive_weight = float(getattr(vision_config, "contrastive_weight", 0.0))
        self.contrastive_temperature = float(getattr(vision_config, "contrastive_temperature", 0.1))
        self.contrastive_margin = float(getattr(vision_config, "contrastive_margin", 0.0))
        self.sim_scale_weight = float(getattr(vision_config, "sim_scale_weight", 0.0))
        self.sim_tau = float(getattr(vision_config, "sim_tau", 0.5))
        self.sim_temp = float(getattr(vision_config, "sim_temp", 0.1))
        self.sim_gamma = float(getattr(vision_config, "sim_gamma", 0.05))
        self.max_frames = int(getattr(vision_config, "max_frames", 32))

        self.vlp = nn.Identity() if vision_hidden_size == dim else nn.Linear(vision_hidden_size, dim, bias=False)

        output_dim = getattr(vision_config, "num_bins", None) if self.use_discrete_action else 2
        if output_dim is None and self.use_discrete_action:
            scale_bins = getattr(vision_config, "scale_bins", None)
            if scale_bins is None:
                import numpy as np
                scale_bins = np.arange(self.min_scale, self.max_scale + self.min_scale, self.min_scale).tolist()
            self.register_buffer("scale_bins", torch.tensor(scale_bins, dtype=torch.float32))
            output_dim = len(scale_bins)
        # self.regression_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(output_dim)))
        self.regression_head = ProjectorBlock(input_dim=dim, output_dim=output_dim, hidden_dim=dim * 4)
        self.last_scales = None
        self.last_frame_features = None
        self._last_frame_metrics = None
        self._last_contrastive_loss = None
        self._last_sim_scale_loss = None
        self._last_entropy = None

    def post_init(self):
        if bool(getattr(self, "force_uniform_ab", False)) and not self.use_discrete_action:
            if isinstance(self.regression_head, nn.Sequential) and isinstance(self.regression_head[-1], nn.Linear):
                head = self.regression_head[-1]
                scale = max(self.beta_param_scale, 1e-6)
                with torch.no_grad():
                    nn.init.zeros_(head.weight)
                    bias_val = -10.0 if self.beta_add_one else float(torch.log(torch.expm1(torch.tensor(1.0))).item())
                    bias_val = bias_val / scale
                    if head.bias is not None and head.bias.numel() >= 2:
                        head.bias.data[0] = bias_val
                        head.bias.data[1] = bias_val
            return
        if not self.init_head or self.use_discrete_action:
            return
        head = None
        if isinstance(self.regression_head, nn.Sequential) and isinstance(self.regression_head[-1], nn.Linear):
            head = self.regression_head[-1]
        elif hasattr(self.regression_head, "get_last_layer"):
            head = self.regression_head.get_last_layer()
        if head is None or not isinstance(head, nn.Linear):
            return
        weight_std = self.init_weight_std
        if getattr(self, "use_dirichlet_budget", False) and getattr(self, "dirichlet_init_weight_std", None) is not None:
            weight_std = self.dirichlet_init_weight_std
        if head.weight is not None:
            nn.init.normal_(head.weight, mean=0.0, std=weight_std)
        scale = max(self.beta_param_scale, 1e-6)
        add_one = 1.0 if self.beta_add_one else 0.0
        if getattr(self, "use_dirichlet_budget", False):
            target_alpha = max(self.init_concentration, add_one + 1e-4)
            bias = torch.log(torch.expm1(torch.tensor(target_alpha - add_one))) / scale
            if head.bias is not None:
                head.bias.data.fill_(bias.item())
            return
        mean_scale = min(max(self.init_scale_mean, self.min_scale), self.max_scale)
        mean_action = (mean_scale - self.min_scale) / max(self.max_scale - self.min_scale, 1e-6)
        mean_action = min(max(mean_action, 1e-4), 1.0 - 1e-4)
        target_alpha = max(mean_action * self.init_concentration, add_one + 1e-4)
        target_beta = max((1.0 - mean_action) * self.init_concentration, add_one + 1e-4)
        bias_alpha = torch.log(torch.expm1(torch.tensor(target_alpha - add_one))) / scale
        bias_beta = torch.log(torch.expm1(torch.tensor(target_beta - add_one))) / scale
        if head.bias is not None and head.bias.numel() >= 2:
            head.bias.data[0] = bias_alpha.item()
            head.bias.data[1] = bias_beta.item()

    def _get_continuous_deterministic(self, params: torch.Tensor):
        if self.continuous_dist == "beta":
            add_one = 1.0 if self.beta_add_one else 0.0
            params_fp32 = (params * self.beta_param_scale).float()
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            dist = torch.distributions.Beta(alpha.float(), beta.float())
            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
                action_0_1 = dist.icdf(q)
            else:
                action_0_1 = dist.mean
        else:
            params_fp32 = params.float()
            mu = params_fp32[..., 0]
            sigma = F.softplus(params_fp32[..., 1]) + 1e-6
            sigma = sigma.clamp(min=0.1, max=5.0)
            dist = torch.distributions.Normal(mu.float(), sigma.float())
            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
                action_0_1 = torch.distributions.TransformedDistribution(dist, torch.distributions.SigmoidTransform()).icdf(q)
            else:
                action_0_1 = torch.sigmoid(mu)
        scales = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scales.to(params.dtype), None

    def _get_discrete_policy(self, logits: torch.Tensor, actions: torch.Tensor = None):
        dist = torch.distributions.Categorical(logits=logits.float())
        if actions is None:
            action_indices = dist.sample()
        else:
            action_indices = actions.long().detach()
        log_prob = dist.log_prob(action_indices).float()
        scales = self.scale_bins[action_indices].to(logits.dtype)
        return action_indices, scales, log_prob

    def _get_continuous_policy(self, params: torch.Tensor, actions: torch.Tensor = None):
        if self.continuous_dist == "beta":
            params_fp32 = (params * self.beta_param_scale).float()
            add_one = 1.0 if self.beta_add_one else 0.0
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            dist = torch.distributions.Beta(alpha.float(), beta.float())
            if actions is None:
                use_reparam = (self.sim_scale_weight > 0) or (self.contrastive_weight > 0)
                action_0_1 = dist.rsample() if use_reparam else dist.sample()
            else:
                action_0_1 = actions.float().detach()
            epsilon = 1e-6
            action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
            log_prob = dist.log_prob(action_0_1).float()
        else:
            params_fp32 = params.float()
            mu = params_fp32[..., 0]
            sigma = F.softplus(params_fp32[..., 1]) + 1e-6
            sigma = sigma.clamp(min=0.1, max=5.0)
            base = torch.distributions.Normal(mu.float(), sigma.float())
            dist = torch.distributions.TransformedDistribution(base, torch.distributions.SigmoidTransform())
            if actions is None:
                use_reparam = (self.sim_scale_weight > 0) or (self.contrastive_weight > 0)
                action_0_1 = dist.rsample() if use_reparam else dist.sample()
            else:
                action_0_1 = actions.float().detach()
            epsilon = 1e-6
            action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
            log_prob = dist.log_prob(action_0_1).float()
        scales = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scales.to(params.dtype), log_prob.to(torch.float32)

    def _get_discrete_deterministic(self, logits: torch.Tensor):
        action_indices = torch.argmax(logits, dim=-1)
        scales = self.scale_bins[action_indices]
        return action_indices, scales, None

    def forward(
        self,
        visual_features_batch: torch.Tensor,
        visual_grid_thw: torch.Tensor,
        eval_mode: bool = False,
        compute_frame_metrics: bool = False,
        actions: torch.Tensor = None,
        visual_per_sample: list = None,
    ):
        compute_aux = actions is not None and eval_mode is not True
        if not compute_frame_metrics:
            self._last_frame_metrics = None
        B, MaxTokens, D = visual_features_batch.shape
        device = visual_features_batch.device
        dtype = visual_features_batch.dtype

        object_t_dims = visual_grid_thw[:, 0].to(torch.long)
        l_raw = (visual_grid_thw[:, 1] * visual_grid_thw[:, 2]).to(torch.long)
        merge_sq = int(self.spatial_merge_size) ** 2
        l_merged = (l_raw // merge_sq).clamp_min(1)
        use_raw = object_t_dims * l_raw <= MaxTokens
        object_l_dims = torch.where(use_raw, l_raw, l_merged)
        max_t_obj = int(object_t_dims.max().item()) if object_t_dims.numel() > 0 else 0

        head_out_dim = None
        if isinstance(self.regression_head, nn.Sequential) and isinstance(self.regression_head[-1], nn.Linear):
            head_out_dim = self.regression_head[-1].out_features
        elif hasattr(self.regression_head, "get_last_layer"):
            last_layer = self.regression_head.get_last_layer()
            if hasattr(last_layer, "out_features"):
                head_out_dim = last_layer.out_features
        elif hasattr(self.regression_head, "out_features"):
            head_out_dim = self.regression_head.out_features
        if head_out_dim is None:
            raise ValueError("Unsupported regression_head: cannot resolve output dimension.")
        head_outputs_obj = torch.zeros((B, max_t_obj, head_out_dim), dtype=dtype, device=device)
        frame_features_obj = torch.zeros((B, max_t_obj, D), dtype=dtype, device=device)
        contrastive_loss_sum = 0.0
        contrastive_count = 0
        if compute_frame_metrics:
            redundancy_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            uniqueness_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            text_relevance_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            info_score_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)

        pair = torch.stack([object_t_dims, object_l_dims], dim=1)
        unique_pairs, inverse_indices = torch.unique(pair, dim=0, return_inverse=True)
        for group_idx, tl in enumerate(unique_pairs):
            t = int(tl[0].item())
            l = int(tl[1].item())
            if t <= 0 or l <= 0:
                continue
            obj_idx = (inverse_indices == group_idx).nonzero(as_tuple=False).squeeze(1).long()
            if obj_idx.numel() == 0:
                continue
            feats = visual_features_batch.index_select(0, obj_idx)[:, : t * l]
            feats = feats.reshape(-1, t, l, D)
            frame_feats = feats.mean(dim=2)
            self.last_frame_features = frame_feats
            if t > 1:
                ff = F.normalize(frame_feats.float(), dim=-1)
                if compute_frame_metrics:
                    sim_prev = (ff[:, 1:] * ff[:, :-1]).sum(dim=-1)
                    sim_prev = F.pad(sim_prev, (1, 0), value=0.0)
                    with torch.no_grad():
                        redundancy_obj[obj_idx, :t] = sim_prev.float().clamp(-1, 1)
                        if t > 1:
                            sim_matrix = torch.matmul(ff, ff.transpose(-1, -2))
                            eye = torch.eye(t, device=device, dtype=torch.bool).unsqueeze(0)
                            sim_matrix = sim_matrix.masked_fill(eye, 0.0)
                            mean_sim = sim_matrix.sum(dim=-1) / (t - 1)
                            uniqueness_obj[obj_idx, :t] = (1.0 - mean_sim).clamp(0.0, 1.0)
                        else:
                            uniqueness_obj[obj_idx, :t] = torch.ones((ff.shape[0], t), device=device, dtype=torch.float32)
            elif compute_frame_metrics:
                pass
            head_outputs = self.regression_head(frame_feats)
            head_outputs_obj[obj_idx, :t] = head_outputs
            frame_features_obj[obj_idx, :t] = frame_feats
            if t > 1:
                sim_pair = (ff[:, 1:] * ff[:, :-1]).sum(dim=-1)
                if self.training and self.contrastive_weight > 0:
                    c_loss = F.relu(sim_pair - self.contrastive_margin).mean()
                    contrastive_loss_sum += c_loss
                    contrastive_count += 1

        temporal_valid = torch.arange(max_t_obj, device=device)[None, :] < object_t_dims[:, None]
        if self.use_discrete_action:
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_deterministic(head_outputs_obj)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_policy(head_outputs_obj, actions)
            pad_val_action = -1
        else:
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_deterministic(head_outputs_obj)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_policy(head_outputs_obj, actions)
            pad_val_action = 0.0
        raw_actions_obj = raw_actions_obj.to(torch.long if self.use_discrete_action else torch.float32)
        scales_obj = scales_obj.to(torch.float32)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.to(torch.float32)
        if self.use_discrete_action:
            dist = torch.distributions.Categorical(logits=head_outputs_obj.float())
            entropy = dist.entropy()
        else:
            if self.continuous_dist == "beta":
                add_one = 1.0 if self.beta_add_one else 0.0
                params_fp32 = (head_outputs_obj * self.beta_param_scale).float()
                alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
                beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
                alpha = alpha.clamp(min=0.1, max=20.0)
                beta = beta.clamp(min=0.1, max=20.0)
                dist = torch.distributions.Beta(alpha.float(), beta.float())
                entropy = dist.entropy()
            else:
                params_fp32 = head_outputs_obj.float()
                mu = params_fp32[..., 0]
                sigma = F.softplus(params_fp32[..., 1]) + 1e-6
                sigma = sigma.clamp(min=0.1, max=5.0)
                dist = torch.distributions.Normal(mu.float(), sigma.float())
                entropy = dist.entropy()
        entropy = entropy.masked_fill(~temporal_valid, 0.0)
        denom = temporal_valid.float().sum().clamp_min(1.0)
        self._last_entropy = entropy.sum() / denom
        if compute_aux and self.sim_scale_weight > 0 and self.training and frame_features_obj.shape[1] > 1:
            f0 = frame_features_obj[:, :-1]
            f1 = frame_features_obj[:, 1:]
            sim = F.cosine_similarity(f0, f1, dim=-1)
            w = torch.sigmoid((sim - self.sim_tau) / self.sim_temp)
            m = (temporal_valid[:, :-1] & temporal_valid[:, 1:]).float()
            if self.use_discrete_action:
                probs = head_outputs_obj.softmax(dim=-1)
                soft_scales_obj = (probs * self.scale_bins).sum(dim=-1)
                s = soft_scales_obj.clamp_min(1e-6).log()
            else:
                params = head_outputs_obj * self.beta_param_scale
                add_one = 1.0 if self.beta_add_one else 0.0
                alpha = F.softplus(params[..., 0]) + add_one + 1e-6
                beta = F.softplus(params[..., 1]) + add_one + 1e-6
                alpha = alpha.clamp(min=0.1, max=20.0)
                beta = beta.clamp(min=0.1, max=20.0)
                mean_action = alpha / (alpha + beta)
                soft_scales_obj = self.min_scale + mean_action * (self.max_scale - self.min_scale)
                s = soft_scales_obj.clamp_min(1e-6).log()
            w = w[:, : s.shape[1] - 1]
            s_prev = s[:, :-1]
            s_cur = s[:, 1:]
            target = s_prev - self.sim_gamma * w
            err = F.relu(s_cur - target)
            self._last_sim_scale_loss = (err * w * m).sum() / (m.sum().clamp_min(1.0))
        else:
            self._last_sim_scale_loss = torch.tensor(0.0, device=device)
        if compute_aux and self.training and self.contrastive_weight > 0:
            if contrastive_count > 0:
                self._last_contrastive_loss = (contrastive_loss_sum / max(contrastive_count, 1)) * self.contrastive_weight
            else:
                self._last_contrastive_loss = torch.tensor(0.0, device=device)
        else:
            self._last_contrastive_loss = None

        raw_actions_obj = raw_actions_obj.masked_fill(~temporal_valid, pad_val_action)
        scales_obj = scales_obj.masked_fill(~temporal_valid, 1.0)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.masked_fill(~temporal_valid, 0.0)
        mask = temporal_valid.float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (scales_obj * mask).sum(dim=1, keepdim=True) / denom
        var = ((scales_obj - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
        self._last_scale_var = var.mean()

        if visual_per_sample is None:
            visual_per_sample = [1] * B
        scale_mask_out, valid_lengths = self.get_scale_mask(
            visual_grid_thw,
            visual_per_sample,
            self.max_frames,
            device,
        )

        if compute_frame_metrics:
            self._last_frame_metrics = {
                "redundancy": self.restructure_sequence(redundancy_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "uniqueness": self.restructure_sequence(uniqueness_obj, visual_grid_thw, scale_mask_out, 0.5, device),
                "text_relevance": self.restructure_sequence(text_relevance_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "info_score": self.restructure_sequence(info_score_obj, visual_grid_thw, scale_mask_out, 0.0, device),
            }

        raw_actions = self.restructure_sequence(raw_actions_obj, visual_grid_thw, scale_mask_out, pad_val_action, device)
        scales = self.restructure_sequence(scales_obj, visual_grid_thw, scale_mask_out, 1.0, device)
        log_probs = self.restructure_sequence(log_probs_obj, visual_grid_thw, scale_mask_out, 0.0, device) if log_probs_obj is not None else None
        self.last_scales = scales
        return raw_actions, scales, log_probs, scale_mask_out

    def get_frame_metrics(self) -> dict:
        return self._last_frame_metrics if self._last_frame_metrics is not None else {}

    def get_contrastive_loss(self):
        return self._last_contrastive_loss

    def get_scale_mask(self, visual_grid_thw, visual_per_sample, max_frames, device):
        object_t_dims = visual_grid_thw[:, 0]
        splits = torch.split(object_t_dims, visual_per_sample)
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        pad_max = max(max_frames, int(valid_lengths.max().item())) if valid_lengths.numel() > 0 else max_frames
        scale_mask = torch.arange(pad_max, device=device)[None, :] < valid_lengths[:, None]
        return scale_mask, valid_lengths

    def restructure_sequence(self, tensor, visual_grid_thw, target_mask, pad_val, device):
        if tensor is None:
            return None
        object_t_dims = visual_grid_thw[:, 0]
        max_len = tensor.shape[1]
        source_mask = torch.arange(max_len, device=device)[None, :] < object_t_dims[:, None]
        valid_values = tensor[source_mask]
        if valid_values.numel() != target_mask.sum():
            raise ValueError(
                f"Value count mismatch: Source has {valid_values.numel()} valid values, "
                f"Target expects {int(target_mask.sum().item())}"
            )
        output = torch.full(target_mask.shape, pad_val, dtype=tensor.dtype, device=device)
        output[target_mask] = valid_values
        return output

    @property
    def _last_sim_scale_loss(self):
        return self.__dict__.get("__last_sim_scale_loss", None)

    @_last_sim_scale_loss.setter
    def _last_sim_scale_loss(self, value):
        self.__dict__["__last_sim_scale_loss"] = value


class SmolPredictorForConditionalGeneration(PreTrainedModel):
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
        model_kwargs = {}
        if getattr(config, "torch_dtype", None) == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif getattr(config, "torch_dtype", None) == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif getattr(config, "_attn_implementation", None):
            model_kwargs["torch_dtype"] = torch.bfloat16
        if getattr(config, "_attn_implementation", None):
            model_kwargs["_attn_implementation"] = config._attn_implementation
        self.processor = AutoProcessor.from_pretrained(config.smol_model_name)
        self.processor.image_processor.do_image_splitting = False
        self.processor.video_processor.num_frames = int(getattr(config, "max_frames", 16))
        self.processor.video_processor.fps = int(getattr(config, "fps", 2.0))
        self.smol_model = AutoModelForImageTextToText.from_pretrained(config.smol_model_name, **model_kwargs)
        self.predictor = HeadOnlyPredictor(config)
        if hasattr(self.predictor, "post_init") and callable(self.predictor.post_init):
            self.predictor.post_init()
        self.input_proj = None
        self.layer_weighted_num_layers = int(getattr(config, "layer_weighted_num_layers", 0))
        if self.layer_weighted_num_layers > 0:
            self.layer_weight_logits = nn.Parameter(torch.zeros(self.layer_weighted_num_layers))
        self.layer_fusion_mode = str(getattr(config, "layer_fusion_mode", "none"))
        self.layer_fusion_indices = getattr(config, "layer_fusion_indices", None)
        if self.layer_fusion_mode == "gated":
            hidden_size = getattr(config, "layer_fusion_in_features", None)
            if hidden_size is None:
                hidden_size = int(getattr(config, "vision_hidden_size", getattr(config, "out_hidden_size", 576)))
            if hidden_size is None:
                hidden_size = getattr(getattr(self.smol_model, "config", None), "hidden_size", None)
            if hidden_size is None:
                hidden_size = int(getattr(config, "hidden_size", 768))
            num_layers = 4
            if isinstance(self.layer_fusion_indices, list) and len(self.layer_fusion_indices) > 0:
                num_layers = len(self.layer_fusion_indices)
            self.layer_fusion_gate = nn.Linear(hidden_size, num_layers)

    @staticmethod
    def _normalize_content(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            content = [content]
        out = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "video" and "video" not in item and "path" in item:
                    item["video"] = item["path"]
                elif t == "image" and "image" not in item and "path" in item:
                    item["image"] = item["path"]
            out.append(item)
        return out

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

    @staticmethod
    def _extract_plain_text(msg):
        if isinstance(msg, list):
            return "\n".join(filter(None, (SmolPredictorForConditionalGeneration._extract_plain_text(m) for m in msg)))
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
    
    @staticmethod
    def _real_image_mask(pixel_values):
        if pixel_values is None:
            return None
        b, n = pixel_values.shape[:2]
        flat = pixel_values.view(b * n, *pixel_values.shape[2:])
        nb_values = flat.shape[1:].numel()
        real_mask = (flat == 0.0).sum(dim=(-1, -2, -3)) != nb_values
        if not real_mask.any():
            real_mask[0] = True
        return real_mask.view(b, n)
    
    @staticmethod
    def _sample_has_video(sample):
        if isinstance(sample, list):
            for msg in sample:
                if SmolPredictorForConditionalGeneration._sample_has_video(msg):
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
    
    @staticmethod
    def _sample_hw(pixel_values, sample_index):
        if pixel_values is None:
            return None
        h, w = pixel_values[sample_index].shape[-2:]
        return h, w

    def _process_single_message(self, msg):
        normalized = self._normalize_message(msg)
        inputs = self._apply_chat_template(normalized)
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

    def _process_messages_parallel(self, messages, max_workers=None):
        if len(messages) <= 1 or max_workers == 1:
            return [self._process_single_message(m) for m in messages]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._process_single_message, messages))

    def _filter_forward_kwargs(self, kwargs: dict):
        try:
            sig = inspect.signature(self.smol_model.forward)
            allowed = set(sig.parameters.keys())
        except (TypeError, ValueError):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    def scale_multi_modal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        multi_modal_data=None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.LongTensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.LongTensor = None,
        messages: list = None,
        actions: torch.Tensor = None,
        scale_mask: torch.Tensor = None,
        **kwargs,
    ):
        eval_mode = kwargs.get("eval_mode", False)
        compute_aux = actions is not None and eval_mode is not True
        compute_frame_metrics = bool(kwargs.get("compute_frame_metrics", False)) and compute_aux
        new_actions, scales, log_probs, new_scale_mask = None, None, None, None

        merge_size = self.config.spatial_merge_size
        merge_len = merge_size ** 2

        if messages is None and pixel_values is None and pixel_values_videos is None:
            raise ValueError("Either messages or pixel_values must be provided.")

        if messages is not None:
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                messages = [messages]
            processed = self._process_messages_parallel(messages, max_workers=min(16, len(messages)))
            pad_id = self.processor.tokenizer.pad_token_id if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token_id is not None else 0
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
            image_grid_thw_list = [p["image_grid_thw"] for p in processed]
            video_grid_thw_list = [p["video_grid_thw"] for p in processed]
            grids = [g for g in image_grid_thw_list if g is not None]
            image_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None
            grids = [g for g in video_grid_thw_list if g is not None]
            video_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None
            messages = [p["normalized"] for p in processed]
        else:
            image_grid_thw_list = None
            video_grid_thw_list = None

        if input_ids is None:
            raise ValueError("Missing input_ids after preprocessing.")

        model_dtype = next(self.smol_model.parameters()).dtype
        if pixel_values is not None and pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(dtype=model_dtype)
        if pixel_values is not None:
            pixel_values = self._ensure_pixel_values_5d(pixel_values)
        if "pixel_attention_mask" in locals():
            pixel_attention_mask = self._ensure_pixel_attention_mask_4d(pixel_attention_mask)

        if image_grid_thw_list is None or all(g is None for g in image_grid_thw_list):
            if video_grid_thw_list is None or all(g is None for g in video_grid_thw_list):
                if pixel_values is not None:
                    sample_count = len(messages) if messages is not None else (input_ids.shape[0] if input_ids is not None else 0)
                    image_grid_thw_list = [None] * sample_count
                    video_grid_thw_list = [None] * sample_count
                    real_mask = self._real_image_mask(pixel_values)
                    patch_size = getattr(self.config, "patch_size", None)
                    if patch_size is None:
                        vision_config = getattr(getattr(self.smol_model, "config", None), "vision_config", None)
                        patch_size = getattr(vision_config, "patch_size", 16)
                    merge = int(getattr(self.config, "spatial_merge_size", 2))
                    for i in range(sample_count):
                        count = int(real_mask[i].sum().item()) if real_mask is not None and i < real_mask.shape[0] else 0
                        h_w = self._sample_hw(pixel_values, i)
                        if count == 0 or h_w is None:
                            continue
                        h, w = h_w
                        grid_h = max(1, h // (patch_size * merge))
                        grid_w = max(1, w // (patch_size * merge))
                        if messages is not None and self._sample_has_video(messages[i]):
                            video_grid_thw_list[i] = torch.tensor(
                                [[count, grid_h, grid_w]],
                                device=self.device,
                                dtype=torch.long,
                            )
                        else:
                            image_grid_thw_list[i] = torch.stack(
                [torch.tensor([1, grid_h, grid_w], device=self.device, dtype=torch.long) for _ in range(count)],
                                dim=0,
                            )
                    grids = [g for g in image_grid_thw_list if g is not None]
                    image_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None
                    grids = [g for g in video_grid_thw_list if g is not None]
                    video_grid_thw = torch.cat(grids, dim=0).to(self.device) if grids else None

        forward_kwargs = self._filter_forward_kwargs(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values if pixel_values is not None else None,
                "pixel_attention_mask": pixel_attention_mask if "pixel_attention_mask" in locals() else None,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "output_hidden_states": True,
                "return_dict": True,
            }
        )

        outputs = self.smol_model(**forward_kwargs)
        if hasattr(outputs, "hidden_states"):
            if self.layer_fusion_mode == "gated":
                hs = list(outputs.hidden_states)
                total = len(hs)
                if isinstance(self.layer_fusion_indices, list) and len(self.layer_fusion_indices) > 0:
                    indices = [min(max(int(i), 0), total - 1) for i in self.layer_fusion_indices]
                else:
                    indices = [
                        min(max(int(total * (1 / 3)), 0), total - 1),
                        min(max(int(total * (1 / 2)), 0), total - 1),
                        min(max(int(total * (2 / 3)), 0), total - 1),
                        total - 1,
                    ]
                selected = [hs[i] for i in indices]
                stacked = torch.stack(selected, dim=0)
                gate_in_dim = stacked[-1].shape[-1]
                if gate_in_dim != self.layer_fusion_gate.in_features:
                    raise ValueError(
                        f"layer_fusion_gate in_features ({self.layer_fusion_gate.in_features}) "
                        f"does not match hidden_state dim ({gate_in_dim}). "
                        "Set layer_fusion_in_features or vision_hidden_size to match."
                    )
                gate_input = stacked[-1].to(self.layer_fusion_gate.weight.dtype)
                logits = self.layer_fusion_gate(gate_input)
                weights = torch.softmax(logits.float(), dim=-1).to(stacked.dtype)
                hidden = (weights.permute(2, 0, 1).unsqueeze(-1) * stacked).sum(dim=0)
            elif self.layer_weighted_num_layers > 0:
                hs = outputs.hidden_states[-self.layer_weighted_num_layers:]
                stacked = torch.stack(hs, dim=0)
                weights = torch.softmax(self.layer_weight_logits.float(), dim=0).to(stacked.dtype)
                hidden = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
            else:
                hidden = outputs.hidden_states[-1]
        else:
            raise RuntimeError("SmolVLM forward did not return hidden states.")

        unified_visual_embeds = []
        unified_visual_grids = []
        visual_per_sample = []
        for b in range(hidden.shape[0]):
            seq_len = int(hidden.shape[1])
            grids = None
            if image_grid_thw_list is not None:
                per_img = image_grid_thw_list[b]
                per_vid = video_grid_thw_list[b] if video_grid_thw_list is not None else None
                if per_img is not None and per_vid is not None:
                    grids = torch.cat([per_img, per_vid], dim=0).to(self.device)
                elif per_img is not None:
                    grids = per_img.to(self.device)
                elif per_vid is not None:
                    grids = per_vid.to(self.device)
            num_objs = 1 if grids is None or grids.numel() == 0 else int(grids.shape[0])
            if grids is None or grids.numel() == 0:
                l_tokens = max(1, seq_len // 8)
                start = max(0, seq_len - l_tokens)
                vis_tokens = hidden[b, start:, :]
                unified_visual_embeds.append(vis_tokens)
                unified_visual_grids.append(torch.tensor([1, l_tokens * merge_len, 1], device=self.device, dtype=torch.long))
                visual_per_sample.append(1)
            else:
                start = seq_len
                count = 0
                for j in range(num_objs):
                    t_dim = int(grids[j, 0].item())
                    l_raw = int((grids[j, 1] * grids[j, 2]).item())
                    l_tokens = max(1, l_raw // merge_len)
                    token_count = max(1, t_dim * l_tokens)
                    start = max(0, start - token_count)
                    vis_tokens = hidden[b, start:start + token_count, :]
                    unified_visual_embeds.append(vis_tokens)
                    unified_visual_grids.append(grids[j])
                    count += 1
                visual_per_sample.append(count)

        actions_for_objects = actions
        if len(unified_visual_embeds) > 0:
            unified_grid_thw_tensor = torch.stack(unified_visual_grids) if len(unified_visual_grids) > 0 else None
            visual_batch = pad_sequence(unified_visual_embeds, batch_first=True)
            # expected_dim = None
            # if isinstance(self.predictor.regression_head, nn.Sequential) and isinstance(self.predictor.regression_head[0], nn.LayerNorm):
            #     norm_shape = self.predictor.regression_head[0].normalized_shape
            #     if isinstance(norm_shape, tuple) and len(norm_shape) > 0:
            #         expected_dim = int(norm_shape[0])
            #     elif isinstance(norm_shape, int):
            #         expected_dim = int(norm_shape)
            # if expected_dim is None:
            #     expected_dim = visual_batch.shape[-1]
            # if visual_batch.shape[-1] != expected_dim:
            #     if self.input_proj is None or self.input_proj.in_features != visual_batch.shape[-1] or self.input_proj.out_features != expected_dim:
            #         self.input_proj = nn.Linear(visual_batch.shape[-1], expected_dim).to(
            #             device=self.device,
            #             dtype=visual_batch.dtype,
            #         )
            #     visual_batch = self.input_proj(visual_batch)
            expected_dtype = None
            if isinstance(self.predictor.regression_head, nn.Sequential) and isinstance(self.predictor.regression_head[0], nn.LayerNorm):
                expected_dtype = self.predictor.regression_head[0].weight.dtype
            else:
                expected_dtype = self.predictor.regression_head.net[0].weight.dtype
            if expected_dtype is not None and visual_batch.dtype != expected_dtype:
                visual_batch = visual_batch.to(expected_dtype)
            # breakpoint()
            visual_batch_proj = self.predictor.vlp(visual_batch)
            if actions is not None and unified_grid_thw_tensor is not None:
                if not torch.is_tensor(actions):
                    actions_for_objects = torch.as_tensor(actions, device=visual_batch_proj.device)
                else:
                    actions_for_objects = actions.to(visual_batch_proj.device)
                if actions_for_objects.dim() == 1:
                    actions_for_objects = actions_for_objects.unsqueeze(0)
                if visual_per_sample and actions_for_objects.shape[0] == len(visual_per_sample):
                    repeats = torch.tensor(visual_per_sample, device=actions_for_objects.device)
                    actions_for_objects = actions_for_objects.repeat_interleave(repeats, dim=0)
                max_t_obj = int(unified_grid_thw_tensor[:, 0].max().item()) if unified_grid_thw_tensor.numel() > 0 else actions_for_objects.shape[1]
                if actions_for_objects.shape[1] > max_t_obj:
                    actions_for_objects = actions_for_objects[:, :max_t_obj]
                elif actions_for_objects.shape[1] < max_t_obj:
                    pad_len = max_t_obj - actions_for_objects.shape[1]
                    pad = torch.zeros((actions_for_objects.shape[0], pad_len), device=actions_for_objects.device, dtype=actions_for_objects.dtype)
                    actions_for_objects = torch.cat([actions_for_objects, pad], dim=1)
            new_actions, scales, log_probs, new_scale_mask = self.predictor(
                visual_features_batch=visual_batch_proj,
                visual_grid_thw=unified_grid_thw_tensor,
                eval_mode=eval_mode,
                compute_frame_metrics=compute_frame_metrics,
                actions=actions_for_objects,
                visual_per_sample=visual_per_sample,
            )

        if actions is not None:
            return {
                "log_probs": log_probs,
                "contrastive_loss": self.predictor.get_contrastive_loss() if compute_aux else None,
                "sim_scale_loss": self.predictor._last_sim_scale_loss if compute_aux else None,
                "frame_metrics": self.predictor.get_frame_metrics() if compute_aux else {},
                "scale_var": getattr(self.predictor, "_last_scale_var", None) if compute_aux else None,
                "concentration_loss": getattr(self.predictor, "_last_concentration_loss", None) if compute_aux else None,
                "entropy": getattr(self.predictor, "_last_entropy", None) if compute_aux else None,
            }

        return {
            "scales": scales if scales is not None else torch.ones((len(visual_per_sample), self.config.max_frames), device=self.device),
            "actions": new_actions,
            "scale_mask": new_scale_mask,
            "log_probs": log_probs,
        }

    def forward(self, *args, **kwargs):
        return self.scale_multi_modal(*args, **kwargs)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    config = SmolPredictorConfig(
        smol_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        _attn_implementation="flash_attention_2",
        patch_size=16,
        spatial_merge_size=2,
        hidden_size=576,
        vision_hidden_size=576,
        out_hidden_size=576,
        output_dim=1024,
        use_discrete_action=False,
        min_scale=0.2,
        max_scale=2.0,
        beta_add_one=True,
        beta_param_scale=0.5,
        continuous_dist="beta",
        continuous_eval_quantile=0.5,
        force_uniform_ab=False,
        init_head=True,
        init_scale_mean=1.0,
        init_concentration=6.0,
        init_weight_std=1e-3,
        contrastive_weight=0.0,
        contrastive_temperature=0.1,
        contrastive_margin=0.0,
        sim_scale_weight=0.0,
        sim_tau=0.5,
        sim_temp=0.1,
        sim_gamma=0.05,
        max_frames=8,
        layer_fusion_mode="gated",
        layer_fusion_indices=[8, 16, 24, 32],
    )
    # model = SmolPredictorForConditionalGeneration(config).to("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_head",
        # config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    device = model.device
    

    img_arr1 = torch.rand(3, 224, 224, device=device)
    img_arr2 = torch.rand(3, 224, 224, device=device)

    messages = [
        [{"role": "user", "content": [{"type": "image", "image": img_arr1}, {"type": "image", "image": img_arr2}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "image", "image": img_arr2}, {"type": "text", "text": "Describe this image."}]}],
    ]
    outputs = model.scale_multi_modal(messages=messages, eval_mode=True)
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in outputs.items()})


    total, trainable = count_params(model)

    print(f"Total params: {total/1e9:.3f} B")
    print(f"Trainable params: {trainable/1e9:.3f} B")
    
    from PIL import Image
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
    #  save_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_head"
    # model.save_pretrained(save_path, safe_serialization=True)
    out = model.scale_multi_modal(messages=messages, return_mm_data=False, eval_mode=True)
    out_text = model.scale_multi_modal(messages=messages_text, return_mm_data=False, eval_mode=True)
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in out.items()})
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in out_text.items()})
    breakpoint()
