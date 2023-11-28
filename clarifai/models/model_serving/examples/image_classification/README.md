## Image Classification Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy image classification models to the Clarifai platform. See the required files section for each model below.

* ### [VIT Age Classifier](./age_vit/)

	Required files to run tests locally:

	* Download the [model checkpoint from huggingface](https://huggingface.co/nateraw/vit-age-classifier/tree/main) and store it under `age_vit/1/checkpoint/`
	```
	huggingface-cli download nateraw/vit-age-classifier --local-dir age_vit/1/checkpoint/ --local-dir-use-symlinks False
	```
