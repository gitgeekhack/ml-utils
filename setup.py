import setuptools

setuptools.setup(
    name='mlutils',
    version='0.0.2',
    description="ML utils library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/gitgeekhack/ml-utils",
    packages=['mlutils', 'mlutils.data', 'mlutils.file', 'mlutils.exceptions', 'mlutils.image'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy==1.22.0',
                      'opencv-python==4.5.5.64',
                      'tqdm==4.63.1'],
)
