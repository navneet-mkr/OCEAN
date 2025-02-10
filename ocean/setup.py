from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ocean-rca",
    version="1.0.0",
    author="Sherlog",
    author_email="navneet.nmk@gmail.com",
    description="Online Multi-modal Causal Structure Learning for Root Cause Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ocean-team/ocean-rca",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ocean-prepare=ocean.scripts.prepare_data:main",
            "ocean-train=ocean.train:main",
            "ocean-evaluate=ocean.evaluate:main",
        ]
    },
) 