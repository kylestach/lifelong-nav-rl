from setuptools import setup, find_packages

setup(
    name='liren',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.9',
    # Parse requirements.txt, ignore comments and empty lines
    install_requires=[
        "jax>=0.4.13",
        "flax",
        "distrax",
        "chex",
        "tensorflow-datasets"
    ]
)