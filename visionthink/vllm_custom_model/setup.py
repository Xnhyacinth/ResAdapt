# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm_custom_model",
    version="0.1",
    packages=["vllm_custom_model"],
    entry_points={
        "vllm.general_plugins": ["register_dummy_model = vllm_custom_model:register"]
    },
)