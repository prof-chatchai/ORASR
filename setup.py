"""
ORASR: Operational Reasoning-Action Safety Routing

Package configuration for the ORASR framework, which implements multi-pathway
safety routing for critical AI systems.

Author: Chatchai Tritham
Supervisor: Chakkrit Snae Namahoot
Institution: Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand
"""

from setuptools import setup, find_packages
from pathlib import Path

# Grab the README content
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_desc = readme_path.read_text(encoding="utf-8")
else:
    long_desc = ""

setup(
    name="orasr",
    version="1.3.1",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchait66@nu.ac.th, chakkrits@nu.ac.th",
    description="ORASR: Operational Reasoning-Action Safety Routing for Critical AI Systems",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/prof-chatchai/ORASR",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
