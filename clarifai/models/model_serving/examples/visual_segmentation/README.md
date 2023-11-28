## Visual Segmentation Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy visual segmentation models to the Clarifai platform. See the required files section for each model below.

* ### [segformer-b2](./segformer-b2/)

	Requirements to run tests locally:

	* Download/Clone the [huggingface model](https://huggingface.co/mattmdjaga/segformer_b2_clothes) into the **segformer-b2/1/checkpoint** directory then start the triton server.
	```
	huggingface-cli download mattmdjaga/segformer_b2_clothes --local-dir segformer-b2/1/checkpoint --local-dir-use-symlinks False --exclude *.safetensors optimizer.pt
	```
