#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='comfy',
        version='0.0.1',
        description='Comfy is a framework for running and testing models',
        author='Some Guy',
        packages=find_packages(),
        
        install_requires=["img2texture @ git+https://github.com/WASasquatch/img2texture.git#egg=img2texture", "Pillow", "torch", "torchdiffeq", 
                          "torchsde", "einops", "open-clip-torch", "transformers==4.26.1", "safetensors>=0.3.0", "pytorch_lightning", "aiohttp",
                          "accelerate", "pyyaml", "pilgram", "pythonperlin", "matplotlib", "numpy < 1.24 , >=1.18", "scikit-learn", "scikit-image==0.20.0", 
                          "scipy", "opencv-python-headless[ffmpeg]", "timm>=0.4.12", "gitpython", "fairscale>=0.4.4", "face_recognition", "imageio", "tqdm", "joblib"]
                        
    )