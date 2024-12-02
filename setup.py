#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"



setup(
    name="omniflow",
    version="0.1",
    author="Shufan Li",
    description="OmniFlow",
    packages=find_packages(""),
    # package_dir={"":'omniflow'},
    python_requires=">=3.9",
)
