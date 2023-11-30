## Visual Embedding Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy visual embedding models to the Clarifai platform. See the required files section for each model below.

* ### [vit-base](./vit-base/)

	Requirements to run tests locally:

	* Download the [model checkpoint & sentencepiece bpe model from huggingface](https://huggingface.co/google/vit-base-patch16-224/tree/main) and store it under `vit-base/1/checkpoint`
	```
	huggingface-cli download google/vit-base-patch16-224 --local-dir vit-base/1/checkpoint --local-dir-use-symlinks False --exclude *.msgpack *.h5 *.safetensors
	```
