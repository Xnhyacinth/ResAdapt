import traceback
import datasets

import re
from io import BytesIO
from PIL import Image

from verl.utils.dataset import RLHFDataset

###
from visionthink.adaptive.utils import tensor_to_tensor_list, tensor_to_temporal_stack_list, split_video_metadata
###

class CustomRLHFDataset(RLHFDataset):
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                            ###
                            if self.config.get("video2image", False):
                                if images is None:
                                    images = []
                                for video in videos:
                                    # images_tensors = tensor_to_tensor_list(video)
                                    # images.extend([image.numpy() for image in images_tensors])
                                    images.extend(tensor_to_tensor_list(video))
                                
                                # if os.environ.get("REMOVEPAD", None):
                                #     raw_prompt = expand_image_prompt(raw_prompt, videos)
                                # else:
                                new_messages = []
                                for msg in messages:
                                    new_msg = msg.copy()
                                    if msg['role'] == 'user' and isinstance(msg.get('content'), list):
                                        new_content = []
                                        for content_item in msg['content']:
                                            if content_item.get('type') == 'video':
                                                for image in images:
                                                    new_content.append({"type": "image", "image": image})
                                            else:
                                                new_content.append(content_item)
                                        new_msg['content'] = new_content
                                    new_messages.append(new_msg)

                                messages = new_messages
                                raw_prompt = self.processor.apply_chat_template(
                                    new_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                                )
                                videos, videos_kwargs = None, {}
                                # multi_modal_data.pop("video", None)

                            elif self.config.get("video2list", False):
                                temporal_patch_size = self.processor.video_processor.temporal_patch_size
                                new_videos, new_video_metadata = [], []

                                for video_tensor, meta in zip(videos, video_metadata):
                                    chunks = tensor_to_temporal_stack_list(video_tensor, temporal_patch_size)
                                    new_videos.extend(chunks)
                                    new_video_metadata.extend(split_video_metadata(meta, temporal_patch_size))

                                new_messages = []
                                for msg in messages:
                                    new_msg = msg.copy()
                                    if msg['role'] == 'user' and isinstance(msg.get('content'), list):
                                        new_content = []
                                        for content_item in msg['content']:
                                            if content_item.get('type') == 'video':
                                                for video in new_videos:
                                                    new_content.append({"type": "video", "video": video})
                                            else:
                                                new_content.append(content_item)
                                        new_msg['content'] = new_content
                                    new_messages.append(new_msg)

                                messages = new_messages
                                raw_prompt = self.processor.apply_chat_template(
                                    new_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                                )

                                videos = new_videos
                                video_metadata = new_video_metadata
                                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                                # multi_modal_data["video"] = [
                                #     (video, metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                                # ]
                            ###
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    
    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[self.prompt_key]
        images = example.pop(self.image_key, [])
        videos = example.pop(self.video_key, [])

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                    elif isinstance(image, dict) and "bytes" in image:
                        image = Image.open(BytesIO(image["bytes"]))
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", "video": videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages