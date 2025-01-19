from setuptools import setup, find_packages

# Read dependencies from requirements.txt
def read_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='mlutils',
    version='0.0.2',
    description="ML utils library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/gitgeekhack/ml-utils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=read_requirements('requirements.txt'),
)
