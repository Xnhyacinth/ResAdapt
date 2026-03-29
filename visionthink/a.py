from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

if __name__ == '__main__':
    MODEL_PATH = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/qwen2.5_vl-3b"
    video_path ="/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4"

    llm = LLM(
        model=MODEL_PATH,  
        gpu_memory_utilization=0.8,  
        tensor_parallel_size=1, 
        max_model_len=16384,  
        dtype="bfloat16", 
        # enforce_eager=True,  
        # limit_mm_per_prompt={"image": 10, "video": 10},  
    )


    sampling_params = SamplingParams(
        temperature=0.1,  
        top_p=0.001,  
        repetition_penalty=1.05,  
        max_tokens=8192,  
        stop_token_ids=[],  
    )

    video_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": "描述这个视频。"},
                {
                    "type": "video", 
                    "video": "file:///" + video_path, 
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28,
                    "max_frames": 2,
                }
            ]
        },
    ]


    messages = video_messages


    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    from visionthink.adaptive.utils import tensor_to_pil_list, expand_video_prompt
    images = []
    # for video in video_inputs:
    #     images.extend(tensor_to_pil_list(video))
    # prompt = expand_video_prompt(prompt, video_inputs)
    # video_inputs = None

    # if os.environ.get("REMOVEPAD", None):
    #     prompt = expand_video_prompt(prompt, video_inputs)
    # else:
    #     new_messages = []
    #     for msg in messages:
    #         new_msg = msg.copy()
    #         if msg['role'] == 'user' and isinstance(msg.get('content'), list):
    #             new_content = []
    #             for content_item in msg['content']:
    #                 if content_item.get('type') == 'video':
    #                     for image in images:
    #                         new_content.append({"type": "image", "image": image})
    #                 else:
    #                     new_content.append(content_item)
    #             new_msg['content'] = new_content
    #         new_messages.append(new_msg)

    #     messages = new_messages
    #     prompt = processor.apply_chat_template(
    #         new_messages, add_generation_prompt=True, tokenize=False
    #     )

    # breakpoint()
    # print("images", images)


    mm_data = {}

    if video_inputs is not None:
        mm_data["video"] = video_inputs

    if len(images) > 0:
        mm_data["image"] = images

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": {
            "image_timestamps": [float(i * 2) for i in range(len(images))],
        }
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text


    print(generated_text)
