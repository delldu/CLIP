"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="CLIP",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="CLIP Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/CLIP.git",
    packages=["CLIP"],
    package_data={"CLIP": ["bpe_simple_vocab_16e6.txt.gz", "models/ViT-B-16.pth", "models/ViT-B-32.pth", "models/ViT-L-14.pth"]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "todos >= 1.0.0",
    ],
)
