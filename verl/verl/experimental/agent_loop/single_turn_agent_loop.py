# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        ###
        if "multi_modal_data" in kwargs:
            multi_modal_data = kwargs["multi_modal_data"]
        else:
            multi_modal_data = await self.process_vision_info(messages)
        ###
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        ###
        if videos is not None and self.dataset_config.get("video2image", False):
            from visionthink.adaptive.utils import video2images
            
            messages, images = video2images(messages, videos, images)
            videos = None

            multi_modal_data["images"] = images
            multi_modal_data.pop("videos", None)

        elif videos is not None and self.dataset_config.get("video2list", False):
            from visionthink.adaptive.utils import video2list

            messages, videos = video2list(messages, videos, self.processor.video_processor.temporal_patch_size)
            
            multi_modal_data["videos"] = videos

        if "scales" in kwargs:
            from visionthink.adaptive.utils import apply_adaptive_scaling

            patch_size = self.processor.image_processor.patch_size
            image_factor = self.processor.video_processor.merge_size * patch_size

            multi_modal_data = apply_adaptive_scaling(
                multi_modal_data=[multi_modal_data],
                scales=kwargs.get("scales"),
                new_scale_mask=kwargs.get("scale_mask"),
                processor=self.processor,
                patch_size=patch_size,
                image_factor=image_factor,
                temporal_patch_size=self.processor.video_processor.temporal_patch_size,
            )[0]

            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
        ###

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        ###
        if multi_modal_data.get("videos"):
            cleaned_videos = []
            for video, meta in multi_modal_data["videos"]:
                if isinstance(meta, dict) and "video_timestamps" in meta:
                    meta = {k: v for k, v in meta.items() if k != "video_timestamps"}
                cleaned_videos.append((video, meta))
            multi_modal_data["videos"] = cleaned_videos
        ###

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
