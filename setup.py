"""
Setup script for Ez Vision - Computer Vision Made Simple
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ez-vision",
    version="1.0.0",
    author="Ez Vision Team",
    author_email="contact@ez-vision.ai",
    description="Computer Vision Made Simple - Train powerful image recognition models in just a few lines of code",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ez-vision",
    packages=find_packages(),
    py_modules=["ez_vision", "nn_utils"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "twine>=3.0",
        ],
    },
    keywords="computer vision, machine learning, deep learning, image classification, ai, tensorflow, beginner-friendly",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ez-vision/issues",
        "Source": "https://github.com/yourusername/ez-vision",
        "Documentation": "https://github.com/yourusername/ez-vision#readme",
    },
    include_package_data=True,
    zip_safe=False,
)
