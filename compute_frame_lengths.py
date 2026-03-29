import argparse
import os
from transformers import AutoProcessor, AutoTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset


def build_config(args):
    return {
        "cache_dir": args.cache_dir,
        "prompt_key": args.prompt_key,
        "image_key": args.image_key,
        "video_key": args.video_key,
        "max_prompt_length": args.max_prompt_length,
        "filter_overlong_prompts": False,
        "return_raw_chat": False,
        "return_full_prompt": False,
        "return_multi_modal_inputs": True,
        "apply_chat_template_kwargs": {},
        "filter_prompts": False,
        "shuffle": False,
        "seed": None,
        "video2image": args.video2image,
        "video2list": args.video2list,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_parquet", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--frames", default="32,64,128")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--image_key", default="images")
    parser.add_argument("--video_key", default="videos")
    parser.add_argument("--cache_dir", default="~/.cache/verl/rlhf")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--video2image", action="store_true")
    parser.add_argument("--video2list", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_parquet = args.output_parquet
    if output_parquet is None:
        base, ext = os.path.splitext(args.input_parquet)
        output_parquet = f"{base}_with_frame_lengths{ext}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    config = build_config(args)

    dataset = RLHFDataset(
        data_files=args.input_parquet,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=-1,
    )

    frames_list = [int(x) for x in args.frames.split(",") if x.strip()]
    ds = dataset.dataframe
    for frames in frames_list:
        column = f"len_frames_{frames}"
        ds = ds.map(
            lambda doc, f=frames, c=column: {c: dataset.doc_to_len(doc, max_frames=f)},
            num_proc=160,
            # batched=True, 
            # batch_size=1000, 
            desc=f"compute {column}",
        )

    ds.to_parquet(output_parquet)
    print(f"saved: {output_parquet}")


if __name__ == "__main__":
    main()
