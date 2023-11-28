## Text to Image Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy text to image models to the Clarifai platform. See the required files section for each model below.

* ### [sd-v1.5 (Stable-Diffusion-v1.5)](./sd-v1.5/)

	* Download the [model checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and store it under `sd-v1.5/1/checkpoint`
	```
	huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir sd-v1.5/1/checkpoint --local-dir-use-symlinks False
	```
