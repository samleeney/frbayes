"""Setup script for FRBayes JAX implementation"""

from setuptools import setup, find_packages

setup(
    name="frbayes-jax",
    version="0.1.0",
    description="JAX implementation of FRBayes with BlackJAX nested sampling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "blackjax>=1.0.0",
        "anesthetic>=2.0.0",
        "fgivenx>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pytest>=7.0.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8",
)