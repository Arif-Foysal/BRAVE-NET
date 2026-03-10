from setuptools import setup, find_packages

setup(
    name="brave_net",
    version="0.1.0",
    description=(
        "BRAVE-Net: Burg Residual Augmented Vision Transformer "
        "for Parkinson's Disease Detection from Dysarthric Speech"
    ),
    author="MD Arif Faysal Nayem",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.1",
        "torch>=2.1.0",
        "timm>=0.9.12",
        "scikit-learn>=1.3.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
