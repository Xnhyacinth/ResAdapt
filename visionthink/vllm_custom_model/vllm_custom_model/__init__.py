# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import ModelRegistry


def register():
    # Test directly passing the model
    # from .my_qwenvl import UniQwenForConditionalGeneration

    # if "MyOPTForCausalLM" not in ModelRegistry.get_supported_archs():
    #     ModelRegistry.register_model("MyOPTForCausalLM", MyOPTForCausalLM)

    # # Test passing lazy model
    # if "MyGemma2Embedding" not in ModelRegistry.get_supported_archs():
    #     ModelRegistry.register_model(
    #         "MyGemma2Embedding",
    #         "vllm_add_dummy_model.my_gemma_embedding:MyGemma2Embedding",
    #     )

    if "UniQwenForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("UniQwenForConditionalGeneration", "vllm_custom_model.my_qwenvl:UniQwenForConditionalGeneration")
        ModelRegistry.register_model("PredictorForConditionalGeneration", "vllm_custom_model.predictor:PredictorForConditionalGeneration")
        ModelRegistry.register_model("MyQwen2_5_VLForConditionalGeneration", "vllm_custom_model.qwen_vl:MyQwen2_5_VLForConditionalGeneration")