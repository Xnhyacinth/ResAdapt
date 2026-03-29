"""
Cold-start training script for predictor with proper supervision.

Key fix: Use deterministic prediction (distribution mean) instead of sampling
to provide stable training signals for MSE loss.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from visionthink.predictor.configuration_predictor import PredictorConfig
from visionthink.predictor.modeling_predictorv2 import PredictorForConditionalGeneration


from transformers import AutoModel, AutoConfig

class ScalesDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" in obj and "scales" in obj:
                    self.records.append(obj)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
    messages = [item["messages"] for item in batch]
    scales = [item["scales"] for item in batch]
    return messages, scales


def _build_target_tensor(scales_list: List[List[float]], max_frames: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(scales_list)
    target = torch.zeros((batch_size, max_frames), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_frames), dtype=torch.float32, device=device)
    for i, scales in enumerate(scales_list):
        if not isinstance(scales, list):
            continue
        length = min(len(scales), max_frames)
        if length <= 0:
            continue
        target[i, :length] = torch.tensor(scales[:length], dtype=torch.float32, device=device)
        mask[i, :length] = 1.0
    return target, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--predictor_version", default="v2")
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--loss_type", default="mse", choices=["mse", "nll", "l1"])
    parser.add_argument("--use_mean_prediction", action="store_true", default=False,
                        help="Use distribution mean for deterministic prediction (recommended)")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--deepspeed_config", default=None)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_dir", default=None)
    parser.add_argument("--loss_log", default=None)
    parser.add_argument("--loss_plot", default=None)
    args = parser.parse_args()

    try:
        from accelerate import Accelerator, DeepSpeedPlugin
    except Exception as e:
        raise ModuleNotFoundError("Missing dependency: accelerate") from e

    ds_plugin = None
    if args.deepspeed_config:
        if not os.path.exists(args.deepspeed_config):
            raise FileNotFoundError(f"deepspeed_config not found: {args.deepspeed_config}")
        with open(args.deepspeed_config, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            raise ValueError(f"deepspeed_config is empty: {args.deepspeed_config}")
        if args.deepspeed_config.endswith((".yaml", ".yml")):
            import yaml
            ds_config = yaml.safe_load(raw)
        else:
            ds_config = json.loads(raw)
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
    )
    device = accelerator.device

    # config = PredictorConfig.from_pretrained(args.model_path)
    # config.predictor_version = args.predictor_version

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.min_scale = 0.2
    config.max_frames = args.max_frames
    if args.loss_type == "nll":
        setattr(config, "use_dirichlet_budget", False)
        setattr(config, "use_discrete_action", False)
    accelerator.print(f"cfg.use_dirichlet_budget={getattr(config,'use_dirichlet_budget', None)} cfg.use_discrete_action={getattr(config,'use_discrete_action', None)}")

    model_kwargs: Dict[str, Any] = {}
    if args.mixed_precision == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.attn_implementation and args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.model_path, config=config, trust_remote_code=True, **model_kwargs)
    model.to(device)
    model.train()

    dataset = ScalesDataset(args.train_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    accelerator.print(f"dataloader_len={len(loader)}")

    predictor_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if "predictor" in name or "regression_head" in name or "scale" in name:
            param.requires_grad = True
            predictor_params.append(param)
        else:
            param.requires_grad = False
            frozen_params.append(param)
    optimizer = torch.optim.AdamW(predictor_params, lr=args.lr)
    accelerator.print(f"trainable_params={sum(p.numel() for p in predictor_params)} frozen_params={sum(p.numel() for p in frozen_params)}")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    loss_log_path = args.loss_log or os.path.join(args.save_dir, "loss.csv")
    loss_plot_path = args.loss_plot or os.path.join(args.save_dir, "loss.png")

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w", encoding="utf-8") as f:
                f.write("global_step,epoch,loss\n")
        def _update_loss_plot():
            try:
                import matplotlib.pyplot as plt

                steps = []
                losses = []
                with open(loss_log_path, "r", encoding="utf-8") as f:
                    next(f, None)
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) != 3:
                            continue
                        steps.append(int(parts[0]))
                        losses.append(float(parts[2]))
                if steps:
                    plt.figure()
                    plt.plot(steps, losses)
                    plt.xlabel("step")
                    plt.ylabel("loss")
                    plt.tight_layout()
                    plt.savefig(loss_plot_path)
                    plt.close()
            except Exception:
                pass
    else:
        def _update_loss_plot():
            return None

    global_step = 0
    start_epoch = 0
    if args.resume_dir:
        accelerator.load_state(args.resume_dir)
        meta_path = os.path.join(args.resume_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            global_step = int(meta.get("global_step", 0))
            start_epoch = int(meta.get("epoch_0based", meta.get("epoch", 0)))

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        printed_debug = False
        
        for messages, scales_list in loader:
            with accelerator.accumulate(model):

                if accelerator.sync_gradients:
                    optimizer.zero_grad(set_to_none=True)

                pred_scales = None

                if args.loss_type == "nll":
                    if bool(getattr(config, "use_discrete_action", False)) or bool(getattr(config, "use_dirichlet_budget", False)):
                        raise NotImplementedError("NLL is only supported for continuous Beta actions")

                    min_scale = float(getattr(config, "min_scale", 0.2))
                    max_scale = float(getattr(config, "max_scale", 2.0))

                    target, mask = _build_target_tensor(scales_list, args.max_frames, device)
                    mask = mask.float()
                    scale_mask_bool = mask.bool()

                    denom = max_scale - min_scale
                    if denom <= 0:
                        raise ValueError("max_scale must be greater than min_scale")

                    actions = (target - min_scale) / denom
                    actions = actions.clamp(1e-6, 1.0 - 1e-6).float()

                    outputs = model(
                        messages=messages,
                        return_mm_data=False,
                        eval_mode=False,
                        actions=actions,
                        scale_mask=scale_mask_bool,
                    )

                    log_probs = outputs.get("log_probs")
                    if log_probs is None:
                        loss = torch.zeros([], device=device, dtype=torch.float32)
                    else:
                        loss = -(log_probs.float() * mask).sum() / mask.sum().clamp_min(1.0)

                else:
                    outputs = model(messages=messages, return_mm_data=False, eval_mode=bool(args.use_mean_prediction))
                    pred_scales = outputs.get("scales")

                    if pred_scales is None:
                        loss = torch.zeros([], device=device, dtype=torch.float32)
                    else:
                        scale_mask = outputs.get("scale_mask")
                        if scale_mask is not None:
                            scale_mask = scale_mask.to(device=device, dtype=torch.float32)

                        target, mask = _build_target_tensor(scales_list, pred_scales.shape[1], device)

                        mask = mask.float()
                        if scale_mask is not None:
                            mask = mask * scale_mask[:, :mask.shape[1]]

                        pred_scales_f = pred_scales.float()

                        if args.loss_type == "mse":
                            loss = ((pred_scales_f - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
                        elif args.loss_type == "l1":
                            loss = (torch.abs(pred_scales_f - target) * mask).sum() / mask.sum().clamp_min(1.0)
                        else:
                            raise ValueError(f"Unknown loss_type: {args.loss_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(predictor_params, max_norm=1.0)
                    optimizer.step()

                
                # Logging
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                if global_step % 10 == 0:
                    accelerator.print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}")
                if accelerator.is_main_process and global_step % 10 == 0 and pred_scales is not None:
                    pred_sample = pred_scales[:3].detach().float().cpu().tolist()
                    gt_sample = scales_list[:3]
                    accelerator.print(f"debug_gt_scales={gt_sample}")
                    accelerator.print(f"debug_pred_scales={pred_sample}")
                if accelerator.is_main_process:
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step},{epoch+1},{loss.item()}\n")
                if args.save_every > 0 and global_step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    step_dir = os.path.join(args.save_dir, f"step_{global_step}")
                    accelerator.save_state(step_dir)
                    if accelerator.is_main_process:
                        meta_path = os.path.join(step_dir, "metadata.json")
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump({"epoch_0based": epoch, "global_step": global_step}, f, ensure_ascii=False)
                        _update_loss_plot()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        accelerator.print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(args.save_dir)
        print(f"Model saved to {args.save_dir}")
        _update_loss_plot()


if __name__ == "__main__":
    main()
