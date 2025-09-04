from setuptools import setup, find_packages

setup(
    name="dda-py",
    version="0.2.0",
    author="Simon Draeger",
    author_email="sdraeger@salk.edu",
    description="A Python wrapper for DDA with APE (Actually Portable Executable) binary support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sdraeger/dda-py",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={"test": ["pytest"]},
)
