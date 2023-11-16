import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

with open("VERSION", "r") as f:
  version = f.read().strip()

with open("requirements.txt", "r") as fh:
  install_requires = fh.read().split('\n')

if install_requires and install_requires[-1] == '':
  # Remove the last empty line
  install_requires = install_requires[:-1]

packages = setuptools.find_namespace_packages(include=["clarifai*"])

setuptools.setup(
    name="clarifai",
    version=f"{version}",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai Python SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'all': ["pycocotools==2.0.6"],
    },
    entry_points={
        "console_scripts": [
            "clarifai-model-upload-init = clarifai.models.model_serving.cli.repository:model_upload_init",
            "clarifai-triton-zip = clarifai.models.model_serving.cli.model_zip:main",
            "clarifai-upload-model = clarifai.models.model_serving.cli.deploy_cli:main"
        ],
    },
    include_package_data=True)
