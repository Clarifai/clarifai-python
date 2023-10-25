## Clarifai Model Types

Models on the clarifai platform are deployed using the [Triton Inference Server Python Backend](https://github.com/triton-inference-server/python_backend) to allow for pre and post processing of data to and from the model.

Inputs into the models are passed as numpy arrays and the predictions are similarly returned as numpy arrays.
The predictions from user defined models in the [inference script](../README.md#the-inference-script) file have to match certain formats and shapes for the models to be upload compatible.

Clarifai [model types](../models/model_types.py) are decorator functions that are responsible for passing input batch requests to user defined inference models to get predictions and format the resultant predictions into Triton Inference responses that are sent by the server for each client inference request.

## Supported Model Types Wrapper Functions:

- visual_detector
- visual_classifier
- text_classifier
- text_to_text
- text_embedder
- text_to_image
- visual_embedder
- visual_segmenter
- multimodal_embedder
