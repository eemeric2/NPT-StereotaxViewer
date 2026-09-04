# NPT-StereotaxViewer/setup.py
from setuptools import setup, find_packages

setup(
    name="npt-stereotax-viewer",
    version="0.1.0",
    description="Interactive stereotaxic MRI viewer for non-human primate neurophysiology",
    author="NPT Team",
    author_email="contact@npt.example.com",
    url="https://github.com/your-org/NPT-StereotaxViewer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "nibabel>=3.0.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)