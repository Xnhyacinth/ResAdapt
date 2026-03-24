# code is adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lmms_eval/loggers/evaluation_tracker.py
import json
import os
import re
import time
import contextlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

from lmms_eval.utils import (
    eval_logger,
    get_file_datetime,
    get_file_task_name,
    get_results_filenames,
    get_sample_results_filenames,
    handle_non_serializable,
    hash_string,
    sanitize_list,
    sanitize_model_name,
    sanitize_task_name,
)

def _env_true(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).lower() in ("1", "true", "yes", "y", "t")

@contextlib.contextmanager
def _tokenizers_parallelism_disabled():
    prev = os.getenv("TOKENIZERS_PARALLELISM", None)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev


def _write_sample_frames_plot_static(scales, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    values = np.asarray(scales, dtype=float)
    indices = np.arange(values.size)
    window = max(3, min(25, values.size // 15)) if values.size > 0 else 3
    if values.size >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(values, kernel, mode="same")
    else:
        smooth = values
    min_idx = int(np.argmin(values)) if values.size > 0 else None
    max_idx = int(np.argmax(values)) if values.size > 0 else None
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 4.5))
    plt.plot(indices, values, color="#54A24B", linewidth=1.0, alpha=0.5, label="raw")
    plt.plot(indices, smooth, color="#4C78A8", linewidth=1.6, label="smoothed")
    if min_idx is not None and max_idx is not None:
        plt.scatter([min_idx, max_idx], [values[min_idx], values[max_idx]], color="#F58518", s=18, zorder=3, label="extremes")
    plt.title(title)
    plt.xlabel("frame index")
    plt.ylabel("scale")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_rescaled_media_visual_static(
    rescaled_mm_data,
    question: str,
    answer: str,
    scales,
    task_name: str,
    doc_id: str,
    root_path: str,
    date_id: str,
    max_frames: Optional[int] = None,
    padding: int = 8,
):
    import numpy as np
    import textwrap
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    try:
        figs_dir = Path(root_path).joinpath("figs_rescaled")
        figs_dir.mkdir(parents=True, exist_ok=True)
        task_dir = figs_dir.joinpath(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir.joinpath(f"{date_id}_{doc_id}_rescaled.png")

        items = []
        imgs = rescaled_mm_data.get("images", None)
        if imgs is None:
            imgs = rescaled_mm_data.get("image", None)
        vids = rescaled_mm_data.get("videos", None)
        if vids is None:
            vids = rescaled_mm_data.get("video", None)
        if imgs is not None and not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if vids is not None and not isinstance(vids, (list, tuple)):
            vids = [vids]

        def to_pil(img):
            try:
                if isinstance(img, Image.Image):
                    return img
                if hasattr(img, "cpu"):
                    if hasattr(img, "dtype"):
                        try:
                            import torch
                            if img.dtype in (torch.bfloat16, torch.float16):
                                img = img.float()
                        except Exception:
                            pass
                    arr = img.detach().cpu()
                    if arr.dim() == 3 and arr.shape[0] in (1, 3):
                        arr = arr.permute(1, 2, 0)
                    arr = arr.numpy()
                elif isinstance(img, np.ndarray):
                    arr = img
                else:
                    return None
                if arr.dtype != np.uint8:
                    if np.max(arr) <= 1.0:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.ndim == 2:
                    return Image.fromarray(arr, mode="L")
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                    return Image.fromarray(arr)
                return None
            except Exception:
                return None

        def make_strip(frames):
            if not frames:
                return None
            max_h = max(fr.height for fr in frames)
            total_w = sum(fr.width for fr in frames) + padding * (len(frames) + 1)
            strip = Image.new("RGB", (total_w, max_h + padding * 2), "white")
            x = padding
            for fr in frames:
                y = padding + (max_h - fr.height) // 2
                strip.paste(fr, (x, y))
                x += fr.width + padding
            return strip

        if imgs:
            for im in imgs:
                p = to_pil(im)
                if p is not None:
                    items.append(p)

        if vids:
            for item in vids:
                vf = None
                if isinstance(item, tuple) and len(item) >= 1:
                    vf = item[0]
                elif isinstance(item, dict):
                    vf = item.get("video", None) or item.get("frames", None)
                else:
                    vf = item
                frames = []
                if hasattr(vf, "cpu"):
                    arr = vf.detach().cpu().numpy()
                    if arr.ndim == 4:
                        frame_count = arr.shape[0] if max_frames is None else min(arr.shape[0], max_frames)
                        for t in range(frame_count):
                            frames.append(arr[t])
                elif isinstance(vf, list):
                    frames = vf if max_frames is None else vf[:max_frames]
                pil_frames = []
                for fr in frames:
                    p = to_pil(fr)
                    if p is not None:
                        pil_frames.append(p)
                strip = make_strip(pil_frames)
                if strip is not None:
                    items.append(strip)

        if not items:
            return

        font = None
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        scale_summary = ""
        if isinstance(scales, (list, tuple)) and scales:
            try:
                vals = [float(v) for v in scales if isinstance(v, (int, float))]
                if vals:
                    scale_summary = f"scale_mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}"
            except Exception:
                scale_summary = ""
        header_text = f"Q: {str(question or '')}\nA: {str(answer or '')}"
        wrapped = textwrap.fill(header_text, width=80)
        if scale_summary:
            wrapped = wrapped + "\n" + scale_summary
        text_lines = wrapped.splitlines()
        max_w = max([it.width for it in items] + [400])
        txt_w = max_w
        txt_h = (len(text_lines) + 1) * 14
        items_h = sum(it.height for it in items) + padding * (len(items) + 1)
        total_h = txt_h + items_h + padding
        canvas = Image.new("RGB", (txt_w + padding * 2, total_h), "white")
        draw = ImageDraw.Draw(canvas)
        y_text = padding
        for line in text_lines:
            draw.text((padding, y_text), line, fill="black", font=font)
            y_text += 14
        y = txt_h + padding * 2
        use_per_item = isinstance(scales, (list, tuple)) and len(scales) == len(items)
        for idx, img in enumerate(items):
            x = padding + (max_w - img.width) // 2
            canvas.paste(img, (x, y))
            if use_per_item:
                val = scales[idx]
                if isinstance(val, (int, float)):
                    s_txt = f"scale={val:.3f}"
                else:
                    s_txt = f"scale={val}"
                draw.text((x, max(0, y - 14)), s_txt, fill="blue", font=font)
            y += img.height + padding
        canvas.save(out_path)
    except Exception:
        return

def _prepare_mm_for_ipc(obj):
    try:
        import torch
    except Exception:
        torch = None
    if obj is None:
        return None
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(obj):
        try:
            return obj.detach().cpu().numpy()
        except Exception:
            return None
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj
    except Exception:
        pass
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return np.array(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _prepare_mm_for_ipc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _prepare_mm_for_ipc(v) for v in obj ]
    return obj

def _write_sample_means_plot_static(entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(np.mean(entry["scales"])) for entry in entries]
    indices = np.arange(len(means))
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 4.5))
    plt.plot(indices, means, color="#4C78A8", linewidth=1.5, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel("mean scale")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_sample_means_dist_plot_static(entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(np.mean(entry["scales"])) for entry in entries]
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(means, bins=30, color="#4C78A8", alpha=0.85)
    axes[0].set_title("histogram")
    axes[0].set_xlabel("mean scale")
    axes[0].set_ylabel("count")
    axes[1].boxplot(means, vert=True, patch_artist=True, boxprops=dict(facecolor="#F58518", alpha=0.6))
    axes[1].set_title("boxplot")
    axes[1].set_ylabel("mean scale")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_sample_mean_std_plot_static(entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(np.mean(entry["scales"])) for entry in entries]
    stds = [float(np.std(entry["scales"])) for entry in entries]
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(means, stds, s=18, color="#4C78A8", alpha=0.75)
    plt.title(title)
    plt.xlabel("mean scale")
    plt.ylabel("std scale")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_task_means_plot_static(task_means, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(task_means.keys())
    values = [task_means[t] for t in tasks]
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(max(6, len(tasks) * 0.6), 4.5))
    plt.bar(tasks, values, color="#F58518", alpha=0.85)
    plt.title(title)
    plt.xlabel("task")
    plt.ylabel("mean scale")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_task_sample_means_boxplot_static(task_sample_means, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(task_sample_means.keys())
    data = [task_sample_means[t] for t in tasks]
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(max(7, len(tasks) * 0.6), 4.8))
    plt.boxplot(data, labels=tasks, patch_artist=True, boxprops=dict(facecolor="#54A24B", alpha=0.6))
    plt.title(title)
    plt.xlabel("task")
    plt.ylabel("sample mean scale")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_sample_visual_len_plot_static(entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(entry["visual_len_mean"]) for entry in entries]
    indices = np.arange(len(means))
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 4.5))
    plt.plot(indices, means, color="#7C3AED", linewidth=1.5, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel("visual length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_sample_visual_len_dist_plot_static(entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(entry["visual_len_mean"]) for entry in entries]
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(means, bins=30, color="#7C3AED", alpha=0.85)
    axes[0].set_title("histogram")
    axes[0].set_xlabel("visual length")
    axes[0].set_ylabel("count")
    axes[1].boxplot(means, vert=True, patch_artist=True, boxprops=dict(facecolor="#7C3AED", alpha=0.6))
    axes[1].set_title("boxplot")
    axes[1].set_ylabel("visual length")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_task_visual_len_plot_static(task_len_means, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(task_len_means.keys())
    values = [task_len_means[t] for t in tasks]
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(max(6, len(tasks) * 0.6), 4.5))
    plt.bar(tasks, values, color="#7C3AED", alpha=0.85)
    plt.title(title)
    plt.xlabel("task")
    plt.ylabel("visual length")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_task_visual_len_boxplot_static(task_len_samples, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(task_len_samples.keys())
    data = [task_len_samples[t] for t in tasks]
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(max(7, len(tasks) * 0.6), 4.8))
    plt.boxplot(data, labels=tasks, patch_artist=True, boxprops=dict(facecolor="#7C3AED", alpha=0.6))
    plt.title(title)
    plt.xlabel("task")
    plt.ylabel("visual length")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_generation_time_plot_static(time_entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = [float(entry["generation_s_mean"]) for entry in time_entries]
    indices = np.arange(len(means))
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 4.5))
    plt.plot(indices, means, color="#F58518", linewidth=1.5, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel("generation seconds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_generation_time_dist_plot_static(time_entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = [float(entry["generation_s_mean"]) for entry in time_entries]
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(means, bins=30, color="#F58518", alpha=0.85)
    axes[0].set_title("histogram")
    axes[0].set_xlabel("generation seconds")
    axes[0].set_ylabel("count")
    axes[1].boxplot(means, vert=True, patch_artist=True, boxprops=dict(facecolor="#F58518", alpha=0.6))
    axes[1].set_title("boxplot")
    axes[1].set_ylabel("generation seconds")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_scale_vs_len_plot_static(scale_len_pairs, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    if scale_len_pairs and isinstance(scale_len_pairs[0], dict):
        scales = np.array([p.get("mean_scale") for p in scale_len_pairs], dtype=float)
        lens = np.array([p.get("mean_len") for p in scale_len_pairs], dtype=float)
    else:
        scales = np.array([p[0] for p in scale_len_pairs], dtype=float)
        lens = np.array([p[1] for p in scale_len_pairs], dtype=float)
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(lens, scales, s=16, alpha=0.7, color="#4C78A8")
    plt.title(title)
    plt.xlabel("visual length")
    plt.ylabel("mean scale")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _write_scale_heatmap_static(multi_frame_entries, output_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    max_len = max(len(e.get("scales", [])) for e in multi_frame_entries)
    data = []
    for e in multi_frame_entries:
        vals = np.asarray(e.get("scales", []), dtype=float)
        if vals.size < max_len:
            pad = np.full((max_len - vals.size,), np.nan, dtype=float)
            vals = np.concatenate([vals, pad], axis=0)
        data.append(vals)
    mat = np.vstack(data)
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 4.5))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.xlabel("frame index")
    plt.ylabel("sample index")
    plt.colorbar(label="scale")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

@dataclass(init=False)
class GeneralConfigTracker:
    """
    Tracker for the evaluation parameters.

    Attributes:
        model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
        model_name (str): Name of the model.
        model_name_sanitized (str): Sanitized model name for directory creation.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_seconds (str): Inferred total evaluation time in seconds (from the start and end times).
    """

    model_source: str = None
    model_name: str = None
    model_name_sanitized: str = None
    system_instruction: str = None
    system_instruction_sha: str = None
    fewshot_as_multiturn: bool = None
    chat_template: str = None
    chat_template_sha: str = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_seconds: str = None

    def __init__(self) -> None:
        """Starts the evaluation timer."""
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:
        """Extracts the model name from the model arguments."""

        def extract_model_name(model_args: str, key: str) -> str:
            """Extracts the model name from the model arguments using a key."""
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # order does matter, e.g. peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "model_version=", "model_name=", "model_id=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ""

    def log_experiment_args(
        self,
        model_source: str,
        model_args: str,
        system_instruction: str,
        chat_template: str,
        fewshot_as_multiturn: bool,
    ) -> None:
        """Logs model parameters and job ID."""
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = sanitize_model_name(self.model_name)
        self.system_instruction = system_instruction
        self.system_instruction_sha = hash_string(system_instruction) if system_instruction else None
        self.chat_template = chat_template
        self.chat_template_sha = hash_string(chat_template) if chat_template else None
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def log_end_time(self) -> None:
        """Logs the end time of the evaluation and calculates the total evaluation time."""
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)


class EvaluationTracker:
    """
    Keeps track and saves relevant information of the evaluation process.
    Compiles the data from trackers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    def __init__(
        self,
        output_path: str = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        details_repo_name: str = "",
        results_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str = "",
        leaderboard_url: str = "",
        point_of_contact: str = "",
        gated: bool = False,
    ) -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            output_path (str): Path to save the results. If not provided, the results won't be saved.
            hub_results_org (str): The Hugging Face organization to push the results to. If not provided, the results will be pushed to the owner of the Hugging Face token.
            hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will be pushed to `lm-eval-results`.
            details_repo_name (str): The name of the Hugging Face repository to push the details to. If not provided, the results will be pushed to `lm-eval-results`.
            result_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will not be pushed and will be found in the details_hub_repo.
            push_results_to_hub (bool): Whether to push the results to the Hugging Face hub.
            push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub.
            public_repo (bool): Whether to push the results to a public or private repository.
            token (str): Token to use when pushing to the Hugging Face hub. This token should have write access to `hub_results_org`.
            leaderboard_url (str): URL to the leaderboard on the Hugging Face hub on the dataset card.
            point_of_contact (str): Contact information on the Hugging Face hub dataset card.
            gated (bool): Whether to gate the repository.
        """
        self.general_config_tracker = GeneralConfigTracker()

        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = HfApi(token=token) if token else None
        self.gated_repo = gated

        if not self.api and (push_results_to_hub or push_samples_to_hub):
            raise ValueError("Hugging Face token is not defined, but 'push_results_to_hub' or 'push_samples_to_hub' is set to True. " "Please provide a valid Hugging Face token by setting the HF_TOKEN environment variable.")

        if self.api and hub_results_org == "" and (push_results_to_hub or push_samples_to_hub):
            hub_results_org = self.api.whoami()["name"]
            eval_logger.warning(f"hub_results_org was not specified. Results will be pushed to '{hub_results_org}'.")

        if hub_repo_name == "":
            details_repo_name = details_repo_name if details_repo_name != "" else "lmms-eval-results"
            results_repo_name = results_repo_name if results_repo_name != "" else details_repo_name
        else:
            details_repo_name = hub_repo_name
            results_repo_name = hub_repo_name
            eval_logger.warning("hub_repo_name was specified. Both details and results will be pushed to the same repository. Using hub_repo_name is no longer recommended, details_repo_name and results_repo_name should be used instead.")

        self.details_repo = f"{hub_results_org}/{details_repo_name}"
        self.details_repo_private = f"{hub_results_org}/{details_repo_name}-private"
        self.results_repo = f"{hub_results_org}/{results_repo_name}"
        self.results_repo_private = f"{hub_results_org}/{results_repo_name}-private"

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict,
        datetime_str: str,
    ) -> None:
        """
        Saves the aggregated results and samples to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            results (dict): The aggregated results to save.
            samples (dict): The samples results to save.
            datetime_str (str): The datetime string to use for the results file.
        """
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                eval_logger.info("Saving results aggregated")

                # calculate cumulative hash for each task - only if samples are provided
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [f"sample_{index}" + s["doc_hash"] for index, s in enumerate(task_samples)]
                        task_hashes[task_name] = hash_string("".join(sample_hashes))

                # update initial results dict
                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                self.date_id = datetime_str.replace(":", "-")
                file_results_aggregated = path.joinpath(f"{self.date_id}_results.json")
                file_results_aggregated.open("w", encoding="utf-8").write(dumped)

                if self.api and self.push_results_to_hub:
                    repo_id = self.results_repo if self.public_repo else self.results_repo_private
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(path.joinpath(f"{self.date_id}_results.json")),
                        path_in_repo=os.path.join(
                            self.general_config_tracker.model_name,
                            f"{self.date_id}_results.json",
                        ),
                        repo_type="dataset",
                        commit_message=f"Adding aggregated results for {self.general_config_tracker.model_name}",
                    )
                    eval_logger.info("Successfully pushed aggregated results to the Hugging Face Hub. " f"You can find them at: {repo_id}")

            except Exception as e:
                eval_logger.warning("Could not save results aggregated")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving results aggregated")

    def save_results_samples(
        self,
        task_name: str,
        samples: dict,
    ) -> None:
        """
        Saves the samples results to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            task_name (str): The task name to save the samples for.
            samples (dict): The samples results to save.
        """
        if self.output_path:
            try:
                if not getattr(self, "date_id", None):
                    self.date_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                file_results_samples = path.joinpath(f"{self.date_id}_samples_{task_name}.jsonl")
                file_results_scales = path.joinpath(f"{self.date_id}_scales_{task_name}.jsonl")
                file_results_len = path.joinpath(f"{self.date_id}_len_{task_name}.jsonl")
                sample_entries = self._collect_sample_scales(samples)
                len_entries = self._collect_sample_visual_lens(samples)
                has_len = bool(len_entries)
                plot_rescaled = _env_true("PLOT_RESCALED", "0")
                plot_scales = _env_true("PLOT_SCALES", "0")

                vis_workers = int(os.getenv("LMMS_EVAL_VIS_WORKERS", "4"))
                rescaled_tasks = []
                for sample in samples:
                    rescaled_mm_data = sample.get("rescaled_mm_data", None)
                    has_allocator_scale = sample.get("has_allocator_scale", None)
                    if isinstance(has_allocator_scale, (list, tuple)):
                        has_allocator_scale = any(bool(v) for v in has_allocator_scale)
                    if rescaled_mm_data is not None and has_allocator_scale:
                        doc_id = sample.get("doc_id", None)
                        question = None
                        arguments = sample.get("arguments", None)
                        if isinstance(arguments, (list, tuple)) and arguments:
                            question = arguments[0]
                        if question is None:
                            question = sample.get("input", "")
                        scales_for_vis = sample.get("scales", None)
                        answer = sample.get("target", "")
                        if plot_rescaled:
                            rescaled_tasks.append(
                                (
                                    _prepare_mm_for_ipc(rescaled_mm_data),
                                    question,
                                    answer,
                                    scales_for_vis,
                                    task_name,
                                    doc_id,
                                    path,
                                )
                            )
                        sample.pop("rescaled_mm_data", None)
                    scales = sample.get("scales", None)
                    generation_s = sample.get("generation_s", None)
                    scale_preprocess_s = sample.get("scale_preprocess_s", None)
                    scale_stats = sample.get("scale_stats", None)

                    scales_dump = (
                        json.dumps(
                            {
                                "doc_id": sample.get("doc_id", None),
                                "doc_hash": sample.get("doc_hash", None),
                                "scales": scales,
                                "scale_stats": scale_stats,
                                "visual_len": sample.get("visual_len", None),
                                "generation_s": generation_s,
                                "scale_preprocess_s": scale_preprocess_s,
                                "scale_time_s": sample.get("scale_time_s", scale_preprocess_s),
                            },
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if scales is not None:
                        with open(file_results_scales, "a", encoding="utf-8") as f:
                            f.write(scales_dump)

                    if has_len:
                        len_value = sample.get("visual_len", None)
                        if isinstance(len_value, (list, tuple)):
                            len_numbers = [int(v) for v in len_value if isinstance(v, (int, float)) and v > 0]
                            len_mean = float(sum(len_numbers) / len(len_numbers)) if len_numbers else None
                        elif isinstance(len_value, (int, float)) and len_value > 0:
                            len_numbers = [int(len_value)]
                            len_mean = float(len_value)
                        else:
                            len_numbers = None
                            len_mean = None

                        len_dump = (
                            json.dumps(
                                {
                                    "doc_id": sample.get("doc_id", None),
                                    "doc_hash": sample.get("doc_hash", None),
                                    "visual_len": len_value,
                                    "visual_len_mean": len_mean,
                                },
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if len_numbers is not None:
                            with open(file_results_len, "a", encoding="utf-8") as f:
                                f.write(len_dump)

                    # we first need to sanitize arguments and resps
                    # otherwise we won't be able to load the dataset
                    # using the datasets library
                    # arguments = {}
                    args = sample.get("arguments", None)
                    if isinstance(args, (list, tuple)) and args:
                        sample["input"] = args[0]
                    else:
                        sample["input"] = sample.get("input", "")
                    if "boxed" in task_name:
                        sample["resps"] = self._trim_think_from_container(sample["resps"])
                        sample["filtered_resps"] = self._trim_think_from_container(sample["filtered_resps"])
                    sample["resps"] = sanitize_list(sample["resps"])
                    sample["filtered_resps"] = sanitize_list(sample["filtered_resps"])
                    if sample["filtered_resps"] == sample["resps"][0] or sample["filtered_resps"] == sample["resps"]:
                        sample.pop("resps")
                    sample["target"] = str(sample["target"])
                    sample.pop("arguments")
                    sample.pop("doc")

                    sample_dump = (
                        json.dumps(
                            sample,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    with open(file_results_samples, "a", encoding="utf-8") as f:
                        f.write(sample_dump)

                if plot_rescaled and rescaled_tasks and vis_workers > 0:
                    with _tokenizers_parallelism_disabled():
                        with ProcessPoolExecutor(max_workers=min(vis_workers, len(rescaled_tasks))) as executor:
                            futures = [
                                executor.submit(
                                    _write_rescaled_media_visual_static,
                                    t[0],
                                    t[1],
                                    t[2],
                                    t[3],
                                    t[4],
                                    t[5],
                                    str(t[6]),
                                    self.date_id,
                                )
                                for t in rescaled_tasks
                            ]
                            for fut in as_completed(futures):
                                try:
                                    fut.result()
                                except Exception:
                                    pass

                has_scales = bool(sample_entries)
                if plot_scales and has_scales:
                    figs_dir = path.joinpath("figs")
                    figs_dir.mkdir(parents=True, exist_ok=True)
                    task_dir = figs_dir.joinpath(task_name)
                    task_dir.mkdir(parents=True, exist_ok=True)

                    plot_jobs = []
                    sample_means_plot_path = figs_dir.joinpath(f"{self.date_id}_sample_means_{task_name}.png")
                    plot_jobs.append((self._write_sample_means_plot, (sample_entries, sample_means_plot_path, f"{task_name} sample mean scales")))

                    sample_means_dist_path = figs_dir.joinpath(f"{self.date_id}_sample_means_dist_{task_name}.png")
                    plot_jobs.append((self._write_sample_means_dist_plot, (sample_entries, sample_means_dist_path, f"{task_name} sample mean distribution")))

                    sample_mean_std_path = figs_dir.joinpath(f"{self.date_id}_sample_mean_std_{task_name}.png")
                    plot_jobs.append((self._write_sample_mean_std_plot, (sample_entries, sample_mean_std_path, f"{task_name} sample mean vs std")))

                    plot_tasks = []
                    for entry in sample_entries:
                        doc_id = entry["doc_id"]
                        scales = entry["scales"]
                        frame_plot_path = task_dir.joinpath(f"{self.date_id}_{doc_id}.png")
                        plot_tasks.append((scales, str(frame_plot_path), f"{task_name} doc {doc_id} scales"))
                    if plot_tasks and vis_workers > 0:
                        with _tokenizers_parallelism_disabled():
                            with ProcessPoolExecutor(max_workers=min(vis_workers, len(plot_tasks))) as executor:
                                futures = [executor.submit(_write_sample_frames_plot_static, t[0], t[1], t[2]) for t in plot_tasks]
                                for fut in as_completed(futures):
                                    try:
                                        fut.result()
                                    except Exception:
                                        pass
                    else:
                        for entry in sample_entries:
                            doc_id = entry["doc_id"]
                            scales = entry["scales"]
                            frame_plot_path = task_dir.joinpath(f"{self.date_id}_{doc_id}.png")
                            self._write_sample_frames_plot(scales, frame_plot_path, f"{task_name} doc {doc_id} scales")

                    task_means = self._extract_task_means_from_scales_files(path, f"{self.date_id}_scales_*.jsonl")
                    if task_means:
                        overall_plot_path = figs_dir.joinpath(f"{self.date_id}_task_mean_scales.png")
                        plot_jobs.append((self._write_task_means_plot, (task_means, overall_plot_path, "task mean scales")))

                    task_sample_means = self._extract_task_sample_means_from_scales_files(path, f"{self.date_id}_scales_*.jsonl")
                    if task_sample_means:
                        overall_boxplot_path = figs_dir.joinpath(f"{self.date_id}_task_sample_means_boxplot.png")
                        plot_jobs.append((self._write_task_sample_means_boxplot, (task_sample_means, overall_boxplot_path, "task sample mean distribution")))

                    if plot_jobs and vis_workers > 0:
                        static_jobs = [
                            (_write_sample_means_plot_static, (sample_entries, str(sample_means_plot_path), f"{task_name} sample mean scales")),
                            (_write_sample_means_dist_plot_static, (sample_entries, str(sample_means_dist_path), f"{task_name} sample mean distribution")),
                            (_write_sample_mean_std_plot_static, (sample_entries, str(sample_mean_std_path), f"{task_name} sample mean vs std")),
                        ]
                        if task_means:
                            static_jobs.append((_write_task_means_plot_static, (task_means, str(overall_plot_path), "task mean scales")))
                        if task_sample_means:
                            static_jobs.append((_write_task_sample_means_boxplot_static, (task_sample_means, str(overall_boxplot_path), "task sample mean distribution")))
                        with _tokenizers_parallelism_disabled():
                            with ProcessPoolExecutor(max_workers=min(vis_workers, len(static_jobs))) as executor:
                                futures = [executor.submit(job[0], *job[1]) for job in static_jobs]
                                for fut in as_completed(futures):
                                    try:
                                        fut.result()
                                    except Exception:
                                        pass
                    else:
                        for job in plot_jobs:
                            job[0](*job[1])

                if has_len:
                    len_figs_dir = path.joinpath("figs_len")
                    len_figs_dir.mkdir(parents=True, exist_ok=True)
                    len_task_dir = len_figs_dir.joinpath(task_name)
                    len_task_dir.mkdir(parents=True, exist_ok=True)

                    len_jobs = []
                    sample_len_plot_path = len_figs_dir.joinpath(f"{self.date_id}_sample_visual_len_{task_name}.png")
                    len_jobs.append((self._write_sample_visual_len_plot, (len_entries, sample_len_plot_path, f"{task_name} sample visual length")))

                    sample_len_dist_path = len_figs_dir.joinpath(f"{self.date_id}_sample_visual_len_dist_{task_name}.png")
                    len_jobs.append((self._write_sample_visual_len_dist_plot, (len_entries, sample_len_dist_path, f"{task_name} sample visual length distribution")))

                    task_len_means = self._extract_task_visual_len_means_from_len_files(path, f"{self.date_id}_len_*.jsonl")
                    if task_len_means:
                        overall_len_plot_path = len_figs_dir.joinpath(f"{self.date_id}_task_visual_len.png")
                        len_jobs.append((self._write_task_visual_len_plot, (task_len_means, overall_len_plot_path, "task visual length")))

                    task_len_samples = self._extract_task_visual_lens_from_len_files(path, f"{self.date_id}_len_*.jsonl")
                    if task_len_samples:
                        overall_len_boxplot = len_figs_dir.joinpath(f"{self.date_id}_task_visual_len_boxplot.png")
                        len_jobs.append((self._write_task_visual_len_boxplot, (task_len_samples, overall_len_boxplot, "task visual length distribution")))
                    if len_jobs and vis_workers > 0:
                        static_jobs = [
                            (_write_sample_visual_len_plot_static, (len_entries, str(sample_len_plot_path), f"{task_name} sample visual length")),
                            (_write_sample_visual_len_dist_plot_static, (len_entries, str(sample_len_dist_path), f"{task_name} sample visual length distribution")),
                        ]
                        if task_len_means:
                            static_jobs.append((_write_task_visual_len_plot_static, (task_len_means, str(overall_len_plot_path), "task visual length")))
                        if task_len_samples:
                            static_jobs.append((_write_task_visual_len_boxplot_static, (task_len_samples, str(overall_len_boxplot), "task visual length distribution")))
                        with _tokenizers_parallelism_disabled():
                            with ProcessPoolExecutor(max_workers=min(vis_workers, len(static_jobs))) as executor:
                                futures = [executor.submit(job[0], *job[1]) for job in static_jobs]
                                for fut in as_completed(futures):
                                    try:
                                        fut.result()
                                    except Exception:
                                        pass
                    else:
                        for job in len_jobs:
                            job[0](*job[1])

                # --- NEW: Generation time plots ---
                time_entries = self._collect_sample_generation_times(samples)
                has_timing = bool(time_entries)
                if has_timing:
                    timing_figs_dir = path.joinpath("figs_timing")
                    timing_figs_dir.mkdir(parents=True, exist_ok=True)

                    # Save timing JSONL file
                    file_results_timing = path.joinpath(f"{self.date_id}_timing_{task_name}.jsonl")
                    for entry in time_entries:
                        timing_dump = (
                            json.dumps(
                                {
                                    "doc_id": entry["doc_id"],
                                    "generation_s": entry["generation_s"],
                                    "generation_s_mean": entry["generation_s_mean"],
                                },
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        with open(file_results_timing, "a", encoding="utf-8") as f:
                            f.write(timing_dump)

                    # Per-task generation time line plot
                    timing_jobs = []
                    gen_time_plot = timing_figs_dir.joinpath(f"{self.date_id}_generation_time_{task_name}.png")
                    timing_jobs.append((self._write_generation_time_plot, (time_entries, gen_time_plot, f"{task_name} generation time")))

                    gen_time_dist = timing_figs_dir.joinpath(f"{self.date_id}_generation_time_dist_{task_name}.png")
                    timing_jobs.append((self._write_generation_time_dist_plot, (time_entries, gen_time_dist, f"{task_name} generation time distribution")))
                    if timing_jobs and vis_workers > 0:
                        static_jobs = [
                            (_write_generation_time_plot_static, (time_entries, str(gen_time_plot), f"{task_name} generation time")),
                            (_write_generation_time_dist_plot_static, (time_entries, str(gen_time_dist), f"{task_name} generation time distribution")),
                        ]
                        with _tokenizers_parallelism_disabled():
                            with ProcessPoolExecutor(max_workers=min(vis_workers, len(static_jobs))) as executor:
                                futures = [executor.submit(job[0], *job[1]) for job in static_jobs]
                                for fut in as_completed(futures):
                                    try:
                                        fut.result()
                                    except Exception:
                                        pass
                    else:
                        for job in timing_jobs:
                            job[0](*job[1])

                # --- NEW: Scale vs visual length correlation plot ---
                if has_scales and has_len:
                    scale_len_pairs = self._collect_scale_len_pairs(samples)
                    if scale_len_pairs:
                        scale_len_plot = figs_dir.joinpath(f"{self.date_id}_scale_vs_len_{task_name}.png")
                        if vis_workers > 0:
                            with _tokenizers_parallelism_disabled():
                                with ProcessPoolExecutor(max_workers=1) as executor:
                                    futures = [executor.submit(_write_scale_vs_len_plot_static, scale_len_pairs, str(scale_len_plot), f"{task_name} scale vs visual length")]
                                    for fut in as_completed(futures):
                                        try:
                                            fut.result()
                                        except Exception:
                                            pass
                        else:
                            self._write_scale_vs_len_plot(scale_len_pairs, scale_len_plot, f"{task_name} scale vs visual length")

                # --- NEW: Scale heatmap for video tasks ---
                if has_scales:
                    # Only create heatmap if samples have multi-frame scales
                    multi_frame_entries = [e for e in sample_entries if len(e.get("scales", [])) > 5]
                    if len(multi_frame_entries) >= 5:
                        heatmap_path = figs_dir.joinpath(f"{self.date_id}_scale_heatmap_{task_name}.png")
                        if vis_workers > 0:
                            with _tokenizers_parallelism_disabled():
                                with ProcessPoolExecutor(max_workers=1) as executor:
                                    futures = [executor.submit(_write_scale_heatmap_static, multi_frame_entries, str(heatmap_path), f"{task_name} scale heatmap")]
                                    for fut in as_completed(futures):
                                        try:
                                            fut.result()
                                        except Exception:
                                            pass
                        else:
                            self._write_scale_heatmap(multi_frame_entries, heatmap_path, f"{task_name} scale heatmap")

                eval_logger.info(f"Saving samples to {file_results_samples}")

                if self.api and self.push_samples_to_hub:
                    repo_id = self.details_repo if self.public_repo else self.details_repo_private
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    try:
                        if self.gated_repo:
                            headers = build_hf_headers()
                            r = get_session().put(
                                url=f"https://huggingface.co/api/datasets/{repo_id}/settings",
                                headers=headers,
                                json={"gated": "auto"},
                            )
                            hf_raise_for_status(r)
                    except Exception as e:
                        eval_logger.warning("Could not gate the repository")
                        eval_logger.info(repr(e))
                    self.api.upload_folder(
                        repo_id=repo_id,
                        folder_path=str(path),
                        path_in_repo=self.general_config_tracker.model_name_sanitized,
                        repo_type="dataset",
                        commit_message=f"Adding samples results for task: {task_name} to {self.general_config_tracker.model_name}",
                    )
                    eval_logger.info(f"Successfully pushed sample results for task: {task_name} to the Hugging Face Hub. " f"You can find them at: {repo_id}")
            except Exception as e:
                eval_logger.warning("Could not save sample results")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving sample results")

    def _write_rescaled_media_visual(
        self,
        *,
        rescaled_mm_data: dict,
        question: str,
        answer: str,
        scales,
        task_name: str,
        doc_id: str,
        root_path: Path,
        max_frames: Optional[int] = None,
        padding: int = 8,
    ):
        try:
            eval_logger.debug(f"[Rescaled Visual] Processing {task_name} doc {doc_id}, data keys: {list(rescaled_mm_data.keys()) if rescaled_mm_data else 'None'}")
            figs_dir = root_path.joinpath("figs_rescaled")
            figs_dir.mkdir(parents=True, exist_ok=True)
            task_dir = figs_dir.joinpath(task_name)
            task_dir.mkdir(parents=True, exist_ok=True)
            date_id = getattr(self, "date_id", None)
            if not date_id:
                date_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path = task_dir.joinpath(f"{date_id}_{doc_id}_rescaled.png")

            items = []
            imgs = rescaled_mm_data.get("images", None)
            if imgs is None:
                imgs = rescaled_mm_data.get("image", None)
            vids = rescaled_mm_data.get("videos", None)
            if vids is None:
                vids = rescaled_mm_data.get("video", None)
            if imgs is not None and not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
            if vids is not None and not isinstance(vids, (list, tuple)):
                vids = [vids]
            eval_logger.debug(f"[Rescaled Visual] imgs count: {len(imgs) if imgs else 0}, vids count: {len(vids) if vids else 0}")

            def to_pil(img):
                try:
                    if isinstance(img, Image.Image):
                        return img
                    if hasattr(img, "cpu"):
                        if hasattr(img, "dtype"):
                            try:
                                import torch
                                if img.dtype in (torch.bfloat16, torch.float16):
                                    img = img.float()
                            except Exception:
                                pass
                        arr = img.detach().cpu()
                        if arr.dim() == 3 and arr.shape[0] in (1, 3):
                            arr = arr.permute(1, 2, 0)
                        arr = arr.numpy()
                    elif isinstance(img, np.ndarray):
                        arr = img
                    else:
                        return None
                    if arr.dtype != np.uint8:
                        if np.max(arr) <= 1.0:
                            arr = arr * 255.0
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    if arr.ndim == 2:
                        return Image.fromarray(arr, mode="L")
                    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                        return Image.fromarray(arr)
                    return None
                except Exception:
                    return None

            def make_strip(frames):
                if not frames:
                    return None
                max_h = max(fr.height for fr in frames)
                total_w = sum(fr.width for fr in frames) + padding * (len(frames) + 1)
                strip = Image.new("RGB", (total_w, max_h + padding * 2), "white")
                x = padding
                for fr in frames:
                    y = padding + (max_h - fr.height) // 2
                    strip.paste(fr, (x, y))
                    x += fr.width + padding
                return strip

            if imgs:
                for im in imgs:
                    p = to_pil(im)
                    if p is not None:
                        items.append(p)

            if vids:
                for item in vids:
                    vf = None
                    if isinstance(item, tuple) and len(item) >= 1:
                        vf = item[0]
                    elif isinstance(item, dict):
                        vf = item.get("video", None) or item.get("frames", None)
                    else:
                        vf = item
                    frames = []
                    if hasattr(vf, "cpu"):
                        arr = vf.detach().cpu().numpy()
                        if arr.ndim == 4:
                            frame_count = arr.shape[0] if max_frames is None else min(arr.shape[0], max_frames)
                            for t in range(frame_count):
                                frames.append(arr[t])
                    elif isinstance(vf, np.ndarray):
                        if vf.ndim == 4:
                            frame_count = vf.shape[0] if max_frames is None else min(vf.shape[0], max_frames)
                            for t in range(frame_count):
                                frames.append(vf[t])
                        elif vf.ndim == 3:
                            frames.append(vf)
                    elif isinstance(vf, list):
                        frames = vf if max_frames is None else vf[:max_frames]
                    eval_logger.debug(f"[Rescaled Visual] video item type={type(vf)}, extracted {len(frames)} frames")
                    pil_frames = []
                    for fr in frames:
                        p = to_pil(fr)
                        if p is not None:
                            pil_frames.append(p)
                    strip = make_strip(pil_frames)
                    if strip is not None:
                        items.append(strip)

            if not items:
                eval_logger.warning(
                    f"[Rescaled Visual] No valid frames for {task_name} doc {doc_id}. "
                    f"imgs_type={type(imgs)} vids_type={type(vids)} keys={list(rescaled_mm_data.keys())}"
                )
                return

            font = None
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            scale_summary = ""
            if isinstance(scales, (list, tuple)) and scales:
                try:
                    vals = [float(v) for v in scales if isinstance(v, (int, float))]
                    if vals:
                        scale_summary = f"scale_mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}"
                except Exception:
                    scale_summary = ""
            header_text = f"Q: {str(question or '')}\nA: {str(answer or '')}"
            wrapped = textwrap.fill(header_text, width=80)
            if scale_summary:
                wrapped = wrapped + "\n" + scale_summary
            temp_img = Image.new("RGB", (1, 1), "white")
            draw = ImageDraw.Draw(temp_img)
            box = draw.multiline_textbbox((0, 0), wrapped, font=font)
            q_h = box[3] - box[1]

            max_w = max(img.width for img in items)
            total_h = q_h + padding * 2 + sum(img.height for img in items) + padding * (len(items) + 1)
            canvas = Image.new("RGB", (max_w + padding * 2, total_h), "white")
            draw = ImageDraw.Draw(canvas)
            draw.multiline_text((padding, padding), wrapped, fill="black", font=font)

            y = q_h + padding * 2
            use_per_item = isinstance(scales, (list, tuple)) and len(scales) == len(items)
            for idx, img in enumerate(items):
                x = padding + (max_w - img.width) // 2
                canvas.paste(img, (x, y))
                if use_per_item:
                    val = scales[idx]
                    if isinstance(val, (int, float)):
                        s_txt = f"scale={val:.3f}"
                    else:
                        s_txt = f"scale={val}"
                    draw.text((x, max(0, y - 14)), s_txt, fill="blue", font=font)
                y += img.height + padding

            canvas.save(out_path)
            eval_logger.info(f"[Rescaled Visual] Saved {out_path}")
        except Exception as e:
            import traceback
            eval_logger.warning(f"Failed to write rescaled media visual for {task_name} doc {doc_id}: {e}\n{traceback.format_exc()}")

    def _collect_sample_scales(self, samples):
        entries = []
        for index, sample in enumerate(samples):
            sample_scales = sample.get("scales", None)
            if sample_scales is None:
                continue
            if hasattr(sample_scales, "tolist"):
                sample_scales = sample_scales.tolist()
            if isinstance(sample_scales, (int, float)):
                sample_scales = [float(sample_scales)]
            if not isinstance(sample_scales, (list, tuple)) or not sample_scales:
                continue
            scales = [float(s) for s in sample_scales if isinstance(s, (int, float))]
            if not scales:
                continue
            doc_id = sample.get("doc_id", index)
            entries.append({"doc_id": doc_id, "scales": scales})
        return entries

    def _collect_sample_visual_lens(self, samples):
        entries = []
        for index, sample in enumerate(samples):
            vlen = sample.get("visual_len", None)
            if isinstance(vlen, (list, tuple)):
                values = [int(v) for v in vlen if isinstance(v, (int, float)) and v > 0]
            elif isinstance(vlen, (int, float)) and vlen > 0:
                values = [int(vlen)]
            else:
                values = []
            if not values:
                continue
            doc_id = sample.get("doc_id", index)
            entries.append({"doc_id": doc_id, "visual_len": values, "visual_len_mean": float(sum(values) / len(values))})
        return entries

    def _collect_sample_generation_times(self, samples):
        """Collect generation time data from samples for plotting."""
        entries = []
        for index, sample in enumerate(samples):
            gen_s = sample.get("generation_s", None)
            if isinstance(gen_s, (list, tuple)):
                values = [float(v) for v in gen_s if isinstance(v, (int, float)) and v >= 0]
            elif isinstance(gen_s, (int, float)) and gen_s >= 0:
                values = [float(gen_s)]
            else:
                values = [0.0]
            doc_id = sample.get("doc_id", index)
            entries.append({
                "doc_id": doc_id,
                "generation_s": values,
                "generation_s_mean": float(sum(values) / len(values)),
            })
        return entries

    def _collect_scale_len_pairs(self, samples):
        """Collect paired scale-length data for correlation plotting."""
        pairs = []
        for sample in samples:
            # Get scales
            sample_scales = sample.get("scales", None)
            if sample_scales is None:
                continue
            if hasattr(sample_scales, "tolist"):
                sample_scales = sample_scales.tolist()
            if isinstance(sample_scales, (int, float)):
                sample_scales = [float(sample_scales)]
            if not isinstance(sample_scales, (list, tuple)) or not sample_scales:
                continue
            scales = [float(s) for s in sample_scales if isinstance(s, (int, float))]
            if not scales:
                continue
            mean_scale = float(sum(scales) / len(scales))

            # Get visual_len
            vlen = sample.get("visual_len", None)
            if isinstance(vlen, (list, tuple)):
                lens = [int(v) for v in vlen if isinstance(v, int) and v > 0]
            elif isinstance(vlen, int) and vlen > 0:
                lens = [int(vlen)]
            else:
                lens = []
            if not lens:
                continue
            mean_len = float(sum(lens) / len(lens))

            pairs.append({"mean_scale": mean_scale, "mean_len": mean_len})
        return pairs

    def _extract_task_visual_len_means_from_len_files(self, directory: Path, pattern: str):
        task_means = {}
        for file_path in directory.glob(pattern):
            task_name = file_path.name.replace(f"{self.date_id}_len_", "").replace(".jsonl", "")
            sample_means = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        mean_val = payload.get("visual_len_mean", None)
                        if isinstance(mean_val, (int, float)) and mean_val > 0:
                            sample_means.append(float(mean_val))
            except Exception:
                continue
            if sample_means:
                task_means[task_name] = float(sum(sample_means) / len(sample_means))
        return task_means

    def _extract_task_visual_lens_from_len_files(self, directory: Path, pattern: str):
        task_lens = {}
        for file_path in directory.glob(pattern):
            task_name = file_path.name.replace(f"{self.date_id}_len_", "").replace(".jsonl", "")
            values = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        mean_val = payload.get("visual_len_mean", None)
                        if isinstance(mean_val, (int, float)) and mean_val > 0:
                            values.append(float(mean_val))
            except Exception:
                continue
            if values:
                task_lens[task_name] = values
        return task_lens

    def _trim_think_from_container(self, value):
        if isinstance(value, str):
            return self._trim_think(value)
        if isinstance(value, list):
            return [self._trim_think_from_container(v) for v in value]
        return value

    def _trim_think(self, text: str):
        marker = "</think>"
        if marker in text:
            return text.split(marker, 1)[1].lstrip()
        return text

    def _extract_task_means_from_scales_files(self, directory: Path, pattern: str):
        task_means = {}
        for file_path in directory.glob(pattern):
            task_name = file_path.name.replace(f"{self.date_id}_scales_", "").replace(".jsonl", "")
            sample_means = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        sample_scales = payload.get("scales", None)
                        if sample_scales is None:
                            continue
                        if hasattr(sample_scales, "tolist"):
                            sample_scales = sample_scales.tolist()
                        if isinstance(sample_scales, (int, float)):
                            sample_scales = [float(sample_scales)]
                        if not isinstance(sample_scales, (list, tuple)) or not sample_scales:
                            continue
                        values = [float(s) for s in sample_scales if isinstance(s, (int, float))]
                        if values:
                            sample_means.append(float(np.mean(values)))
            except Exception:
                continue
            if sample_means:
                task_means[task_name] = float(np.mean(sample_means))
        return task_means

    def _extract_task_sample_means_from_scales_files(self, directory: Path, pattern: str):
        task_sample_means = {}
        for file_path in directory.glob(pattern):
            task_name = file_path.name.replace(f"{self.date_id}_scales_", "").replace(".jsonl", "")
            sample_means = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        sample_scales = payload.get("scales", None)
                        if sample_scales is None:
                            continue
                        if hasattr(sample_scales, "tolist"):
                            sample_scales = sample_scales.tolist()
                        if isinstance(sample_scales, (int, float)):
                            sample_scales = [float(sample_scales)]
                        if not isinstance(sample_scales, (list, tuple)) or not sample_scales:
                            continue
                        values = [float(s) for s in sample_scales if isinstance(s, (int, float))]
                        if values:
                            sample_means.append(float(np.mean(values)))
            except Exception:
                continue
            if sample_means:
                task_sample_means[task_name] = sample_means
        return task_sample_means

    def _write_sample_means_plot(self, entries, output_path: Path, title: str):
        means = [float(np.mean(entry["scales"])) for entry in entries]
        indices = np.arange(len(means))
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(9, 4.5))
        plt.plot(indices, means, color="#4C78A8", linewidth=1.5, marker="o", markersize=2)
        plt.title(title)
        plt.xlabel("sample index")
        plt.ylabel("mean scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_sample_visual_len_plot(self, entries, output_path: Path, title: str):
        means = [float(entry["visual_len_mean"]) for entry in entries]
        indices = np.arange(len(means))
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(9, 4.5))
        plt.plot(indices, means, color="#7C3AED", linewidth=1.5, marker="o", markersize=2)
        plt.title(title)
        plt.xlabel("sample index")
        plt.ylabel("visual length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_sample_mean_std_plot(self, entries, output_path: Path, title: str):
        means = [float(np.mean(entry["scales"])) for entry in entries]
        stds = [float(np.std(entry["scales"])) for entry in entries]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(means, stds, s=18, color="#4C78A8", alpha=0.75)
        plt.title(title)
        plt.xlabel("mean scale")
        plt.ylabel("std scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_sample_frames_plot(self, scales, output_path: Path, title: str):
        values = np.asarray(scales, dtype=float)
        indices = np.arange(values.size)
        window = max(3, min(25, values.size // 15)) if values.size > 0 else 3
        if values.size >= window:
            kernel = np.ones(window, dtype=float) / float(window)
            smooth = np.convolve(values, kernel, mode="same")
        else:
            smooth = values
        min_idx = int(np.argmin(values)) if values.size > 0 else None
        max_idx = int(np.argmax(values)) if values.size > 0 else None
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(9, 4.5))
        plt.plot(indices, values, color="#54A24B", linewidth=1.0, alpha=0.5, label="raw")
        plt.plot(indices, smooth, color="#4C78A8", linewidth=1.6, label="smoothed")
        if min_idx is not None and max_idx is not None:
            plt.scatter([min_idx, max_idx], [values[min_idx], values[max_idx]], color="#F58518", s=18, zorder=3, label="extremes")
        plt.title(title)
        plt.xlabel("frame index")
        plt.ylabel("scale")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_task_means_plot(self, task_means, output_path: Path, title: str):
        tasks = list(task_means.keys())
        values = [task_means[t] for t in tasks]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(max(6, len(tasks) * 0.6), 4.5))
        plt.bar(tasks, values, color="#F58518", alpha=0.85)
        plt.title(title)
        plt.xlabel("task")
        plt.ylabel("mean scale")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_task_sample_means_boxplot(self, task_sample_means, output_path: Path, title: str):
        tasks = list(task_sample_means.keys())
        data = [task_sample_means[t] for t in tasks]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(max(7, len(tasks) * 0.6), 4.8))
        plt.boxplot(data, labels=tasks, patch_artist=True, boxprops=dict(facecolor="#54A24B", alpha=0.6))
        plt.title(title)
        plt.xlabel("task")
        plt.ylabel("sample mean scale")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_sample_means_dist_plot(self, entries, output_path: Path, title: str):
        means = [float(np.mean(entry["scales"])) for entry in entries]
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(means, bins=30, color="#4C78A8", alpha=0.85)
        axes[0].set_title("histogram")
        axes[0].set_xlabel("mean scale")
        axes[0].set_ylabel("count")
        axes[1].boxplot(means, vert=True, patch_artist=True, boxprops=dict(facecolor="#F58518", alpha=0.6))
        axes[1].set_title("boxplot")
        axes[1].set_ylabel("mean scale")
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_sample_visual_len_dist_plot(self, entries, output_path: Path, title: str):
        means = [float(entry["visual_len_mean"]) for entry in entries]
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(means, bins=30, color="#7C3AED", alpha=0.85)
        axes[0].set_title("histogram")
        axes[0].set_xlabel("visual length")
        axes[0].set_ylabel("count")
        axes[1].boxplot(means, vert=True, patch_artist=True, boxprops=dict(facecolor="#34D399", alpha=0.6))
        axes[1].set_title("boxplot")
        axes[1].set_ylabel("visual length")
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_task_visual_len_plot(self, task_means, output_path: Path, title: str):
        tasks = list(task_means.keys())
        values = [task_means[t] for t in tasks]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(max(6, len(tasks) * 0.6), 4.5))
        plt.bar(tasks, values, color="#7C3AED", alpha=0.85)
        plt.title(title)
        plt.xlabel("task")
        plt.ylabel("visual length")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_task_visual_len_boxplot(self, task_visual_lens, output_path: Path, title: str):
        tasks = list(task_visual_lens.keys())
        data = [task_visual_lens[t] for t in tasks]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(max(7, len(tasks) * 0.6), 4.8))
        plt.boxplot(data, labels=tasks, patch_artist=True, boxprops=dict(facecolor="#34D399", alpha=0.6))
        plt.title(title)
        plt.xlabel("task")
        plt.ylabel("visual length")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_generation_time_plot(self, entries, output_path: Path, title: str):
        """Line plot of per-sample generation time."""
        times = [float(entry["generation_s_mean"]) for entry in entries]
        indices = np.arange(len(times))
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(9, 4.5))
        plt.plot(indices, times, color="#E45756", linewidth=1.5, marker="o", markersize=2)
        plt.title(title)
        plt.xlabel("sample index")
        plt.ylabel("generation time (s)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_generation_time_dist_plot(self, entries, output_path: Path, title: str):
        """Histogram and boxplot of generation times."""
        times = [float(entry["generation_s_mean"]) for entry in entries]
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(times, bins=30, color="#E45756", alpha=0.85)
        axes[0].set_title("histogram")
        axes[0].set_xlabel("generation time (s)")
        axes[0].set_ylabel("count")
        axes[1].boxplot(times, vert=True, patch_artist=True, boxprops=dict(facecolor="#FF9DA6", alpha=0.6))
        axes[1].set_title("boxplot")
        axes[1].set_ylabel("generation time (s)")
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_task_generation_time_plot(self, task_times, output_path: Path, title: str):
        """Bar chart of mean generation time per task."""
        tasks = list(task_times.keys())
        values = [task_times[t] for t in tasks]
        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(max(6, len(tasks) * 0.6), 4.5))
        plt.bar(tasks, values, color="#E45756", alpha=0.85)
        plt.title(title)
        plt.xlabel("task")
        plt.ylabel("mean generation time (s)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_scale_vs_len_plot(self, pairs, output_path: Path, title: str):
        """Scatter plot of scale vs visual length for correlation analysis."""
        if not pairs:
            return
        lens = [p["mean_len"] for p in pairs]
        scales = [p["mean_scale"] for p in pairs]
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(lens, scales, s=20, color="#4C78A8", alpha=0.6)
        ax.set_xlabel("visual length")
        ax.set_ylabel("mean scale")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Add trend line
        if len(lens) >= 3:
            try:
                z = np.polyfit(lens, scales, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(lens), max(lens), 50)
                ax.plot(x_line, p(x_line), color="#F58518", linewidth=2, linestyle="--", alpha=0.8, label=f"trend: y={z[0]:.4f}x+{z[1]:.3f}")
                ax.legend(loc="best", frameon=False, fontsize=8)
            except Exception:
                pass
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _write_scale_heatmap(self, entries, output_path: Path, title: str, max_samples: int = 200, max_frames: int = 300):
        """Heatmap visualization of per-frame scales across samples (y: sample, x: frame)."""
        if not entries:
            return
        # Limit samples for performance
        entries = entries[:max_samples]
        # Find max frame length
        max_len = min(max(len(e["scales"]) for e in entries), max_frames)
        # Build matrix (pad with NaN for missing frames)
        matrix = np.full((len(entries), max_len), np.nan)
        for i, entry in enumerate(entries):
            scales = entry["scales"][:max_len]
            matrix[i, :len(scales)] = scales

        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(12, max(5, len(entries) * 0.05)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("frame index")
        ax.set_ylabel("sample index")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("scale")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def recreate_metadata_card(self) -> None:
        """
        Creates a metadata card for the evaluation results dataset and pushes it to the Hugging Face hub.
        """

        eval_logger.info("Recreating metadata card")
        repo_id = self.details_repo if self.public_repo else self.details_repo_private

        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = get_results_filenames(files_in_repo)
        sample_files = get_sample_results_filenames(files_in_repo)

        # Build a dictionary to store the latest evaluation datetime for:
        # - Each tested model and its aggregated results
        # - Each task and sample results, if existing
        # i.e. {
        #     "org__model_name__gsm8k": "2021-09-01T12:00:00",
        #     "org__model_name__ifeval": "2021-09-01T12:00:00",
        #     "org__model_name__results": "2021-09-01T12:00:00"
        # }
        latest_task_results_datetime = defaultdict(lambda: datetime.min.isoformat())

        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            results_datetime = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            # Results and sample results for the same model and task will have the same datetime
            samples_key = f"{model_name}__{task_name_sanitized}"
            results_key = f"{model_name}__results"
            latest_datetime = max(
                latest_task_results_datetime[samples_key],
                results_datetime,
            )
            latest_task_results_datetime[samples_key] = latest_datetime
            latest_task_results_datetime[results_key] = max(
                latest_task_results_datetime[results_key],
                latest_datetime,
            )

        # Create metadata card
        card_metadata = MetadataConfigs()

        # Add the latest aggregated results to the metadata card for easy access
        for file_path in results_files:
            file_path = Path(file_path)
            results_filename = file_path.name
            model_name = file_path.parent
            eval_date = get_file_datetime(results_filename)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(results_filename).name
            config_name = f"{model_name}__results"
            sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", latest_task_results_datetime[config_name])

            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all results files are listed in the metadata card
                current_results = card_metadata.get(config_name, {"data_files": []})
                current_results["data_files"].append({"split": eval_date_sanitized, "path": [str(results_filename)]})
                card_metadata[config_name] = current_results
                # If the results file is the newest, update the "latest" field in the metadata card
                card_metadata[config_name]["data_files"].append({"split": "latest", "path": [str(results_filename)]})

        # Add the tasks details configs
        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = get_file_task_name(filename)
            eval_date = get_file_datetime(filename)
            task_name_sanitized = sanitize_task_name(task_name)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(filename).name
            config_name = f"{model_name}__{task_name_sanitized}"
            sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", latest_task_results_datetime[config_name])
            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all sample results files are listed in the metadata card
                current_details_for_task = card_metadata.get(config_name, {"data_files": []})
                current_details_for_task["data_files"].append({"split": eval_date_sanitized, "path": [str(results_filename)]})
                card_metadata[config_name] = current_details_for_task
                # If the samples results file is the newest, update the "latest" field in the metadata card
                card_metadata[config_name]["data_files"].append({"split": "latest", "path": [str(results_filename)]})

        # Get latest results and extract info to update metadata card examples
        latest_datetime = max(latest_task_results_datetime.values())
        latest_model_name = max(latest_task_results_datetime, key=lambda k: latest_task_results_datetime[k])
        last_results_file = [f for f in results_files if latest_datetime.replace(":", "-") in f][0]
        last_results_file_path = hf_hub_url(repo_id=repo_id, filename=last_results_file, repo_type="dataset")
        latest_results_file = load_dataset("json", data_files=last_results_file_path, split="train")
        results_dict = latest_results_file["results"][0]
        new_dictionary = {"all": results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)

        dataset_summary = "Dataset automatically created during the evaluation run of model "
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += f"[{self.general_config_tracker.model_name}](https://huggingface.co/{self.general_config_tracker.model_name})\n"
        else:
            dataset_summary += f"{self.general_config_tracker.model_name}\n"
        dataset_summary += (
            f"The dataset is composed of {len(card_metadata)-1} configuration(s), each one corresponding to one of the evaluated task.\n\n"
            f"The dataset has been created from {len(results_files)} run(s). Each run can be found as a specific split in each "
            'configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.\n\n'
            'An additional configuration "results" store all the aggregated results of the run.\n\n'
            "To load the details from a run, you can for instance do the following:\n"
        )
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += "```python\nfrom datasets import load_dataset\n" f'data = load_dataset(\n\t"{repo_id}",\n\tname="{latest_model_name}",\n\tsplit="latest"\n)\n```\n\n'
        dataset_summary += (
            "## Latest results\n\n"
            f'These are the [latest results from run {latest_datetime}]({last_results_file_path.replace("/resolve/", "/blob/")}) '
            "(note that there might be results for other tasks in the repos if successive evals didn't cover the same tasks. "
            'You find each in the results and the "latest" split for each eval):\n\n'
            f"```python\n{results_string}\n```"
        )
        card_data = DatasetCardData(
            dataset_summary=dataset_summary,
            repo_url=f"https://huggingface.co/{self.general_config_tracker.model_name}",
            pretty_name=f"Evaluation run of {self.general_config_tracker.model_name}",
            leaderboard_url=self.leaderboard_url,
            point_of_contact=self.point_of_contact,
        )
        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")
