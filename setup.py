# setup.py
from setuptools import find_packages, setup

setup(
    name="campd_visualizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.9",
)
