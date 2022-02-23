import setuptools

setuptools.setup(
    name='mlutils',
    version='0.0.2',
    description="ML utils library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/gitgeekhack/ml-utils",
    packages=['mlutils', 'mlutils.data_ops', 'mlutils.file_ops'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
