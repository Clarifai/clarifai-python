## Multimodal Embedder Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy image classification models to the Clarifai platform. See the required files section for each model below.

* ### [CLIP](./clip/)

	Required files to run tests locally:

	* Download the [model checkpoint from huggingface](https://huggingface.co/openai/clip-vit-base-patch32) and store it under `clip/1/checkpoint/`
	```
	huggingface-cli download openai/clip-vit-base-patch32 --local-dir clip/1/checkpoint/ --local-dir-use-symlinks False --exclude *.msgpack *.h5
	```
