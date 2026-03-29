import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def _extract_video_path(row: Dict[str, Any]) -> str:
    video_path = ""
    videos = row.get("videos")
    if isinstance(videos, list) and videos:
        first = videos[0]
        if isinstance(first, dict) and isinstance(first.get("video"), str):
            video_path = first["video"]
    if not video_path:
        extra = row.get("extra_info")
        if isinstance(extra, dict) and isinstance(extra.get("video_path"), str):
            video_path = extra["video_path"]
    return video_path


def _iter_rows(parquet_path: str) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches():
        for row in batch.to_pylist():
            yield row


def _reservoir_sample_rows(
    parquet_path: str,
    *,
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0
    for row in _iter_rows(parquet_path):
        video_path = _extract_video_path(row)
        if not video_path:
            continue
        if not os.path.exists(video_path):
            continue
        seen += 1
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = rng.randrange(seen)
            if j < k:
                reservoir[j] = row
    return reservoir


def _load_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    images: List[Image.Image] = []
    for frame in frames:
        images.append(Image.fromarray(frame).convert("RGB"))
    return images


def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    samples = _reservoir_sample_rows(args.input_parquet, k=args.sample_size, seed=args.seed)
    if not samples:
        raise ValueError("No valid video samples found in parquet.")

    index_path = os.path.join(output_dir, "samples.jsonl")
    with open(index_path, "w", encoding="utf-8") as index_f:
        for i, row in enumerate(samples):
            sample_id = f"sample_{i:04d}"
            sample_dir = os.path.join(output_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            video_path = _extract_video_path(row)
            frames = _load_video_frames(video_path, args.num_frames)

            frame_paths: List[str] = []
            for j, img in enumerate(frames):
                frame_name = f"frame_{j:04d}.jpg"
                frame_path = os.path.join(sample_dir, frame_name)
                img.save(frame_path, format="JPEG", quality=95)
                frame_paths.append(frame_path)

            meta = {
                "sample_id": sample_id,
                "video_path": video_path,
                "frame_paths": frame_paths,
                "data": row,
            }
            _safe_write_json(os.path.join(sample_dir, "data.json"), meta)
            index_f.write(json.dumps(meta, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
