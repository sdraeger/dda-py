from setuptools import setup, find_packages

setup(
    name="dda-py",
    version="0.1.4",
    author="Simon Draeger",
    author_email="sdraeger@salk.edu",
    description="A Python wrapper for the DDA software",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sdraeger/dda-py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=2.2.4",
    ],
    extras_require={"test": ["pytest"]},
)
