## Text Embedding Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy text embedding models to the Clarifai platform. See the required files section for each model below.

* ### [Instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

	Requirements to run tests locally:

	* Download/Clone the [huggingface model](https://huggingface.co/hkunlp/instructor-xl) into the **instructor-xl/1/** directory then start the triton server.
	```
	huggingface-cli download hkunlp/instructor-xl --local-dir instructor-xl/1/checkpoint/sentence_transformers/hkunlp_instructor-xl --local-dir-use-symlinks False
	```
