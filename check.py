import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
# image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModel.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    trust_remote_code=True,
).to(DEVICE)

# Create input messages
image = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/VisionThink/scissor.png")
image1 = Image.open("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Pretrain/images/00000/000000010.jpg")
messages = [
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "image", "image": image1}, {"type": "text", "text": "Describe this image."}]}],
        [{"role": "user", "content": [{"type": "video", "video": "/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K/gpt4o_caption_prompt/83FR0RjX7qA.mp4", "max_frames": 8}, {"type": "text", "text": "Describe this video."}]}],
    ]

# Prepare inputs
inputs = processor.apply_chat_template(messages[2], add_generation_prompt=True, tokenize=True)
# inputs = processor(text=prompt, images=[image], return_tensors="pt")
# inputs = inputs.to(DEVICE)
# from qwen_vl_utils import process_vision_info
# prompt= processor.apply_chat_template(messages[2], add_generation_prompt=True)
# images, videos = process_vision_info(messages[2], image_patch_size=16)
# inputsxx = processor(text=prompt, images=images, videos=videos, return_tensors="pt")
breakpoint()

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
