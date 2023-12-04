## Text-to-Text Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy all models that take a text input and yield a text output prediction e.g. text generation, summarization and translation models to the Clarifai platform. See the required files section for each model below.

* ### [Bart-paper2slides-summarizer](https://huggingface.co/com3dian/Bart-large-paper2slides-summarizer)

	Requirements to run tests locally:

	* Download/Clone the [huggingface model](https://huggingface.co/com3dian/Bart-large-paper2slides-summarizer) and store it under the **bart-summarize/1/checkpoint** directory.
		```
		huggingface-cli download com3dian/Bart-large-paper2slides-summarizer --local-dir bart-summarize/1/checkpoint --local-dir-use-symlinks False --exclude *.safetensors
		```
