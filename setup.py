import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="retriv",
    version="0.1.4",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="retriv: A Blazing-Fast Python Search Engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/retriv",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "nltk",
        "numba>=0.54.1",
        "tqdm",
        "optuna",
        "krovetzstemmer",
        "pystemmer==2.0.1",
        "cyhunspell",
        "unidecode",
        "scikit-learn",
        "ranx",
        "indxr",
        "oneliner_utils",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=["information retrieval", "search engine", "bm25", "numba"],
    python_requires=">=3.7",
)
