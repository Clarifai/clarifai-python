## Text Classification Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy text classification models to the Clarifai platform. See the required files section for each model below.

* ### [XLM-Roberta Tweet Sentiment Classifier](./xlm-roberta/)

	Required files to run tests locally:

	* Download the [model checkpoint](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment/tree/main) and store it under `xlm-roberta/1/checkpoint/`
	```
	huggingface-cli download cardiffnlp/twitter-xlm-roberta-base-sentiment --local-dir xlm-roberta/1/checkpoint/ --local-dir-use-symlinks False
	```
