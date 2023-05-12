#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='comfy',
        version='0.0.1',
        description='Comfy is a framework for running and testing models',
        author='Some Guy',
        packages=find_packages(),
        install_requires=["Pillow", "torch", "torchdiffeq", "torchsde", "einops", "open-clip-torch", "transformers>=4.25.1", "safetensors>=0.3.0", "pytorch_lightning", "aiohttp", "accelerate", "pyyaml"]
    )