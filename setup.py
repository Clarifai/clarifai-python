import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

packages = setuptools.find_packages(include=["clarifai_utils*"])

setuptools.setup(
    name="clarifai-utils",
    version="0.0.9",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai Python Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python-utils",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.6',
    install_requires=[
        "clarifai-grpc>=7.12.0rc1",
    ],
    package_data={p: ["*.pyi"]
                  for p in packages},
    include_package_data=True)
