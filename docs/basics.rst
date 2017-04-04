==============
Basic Concepts
==============

This page lists a few basic concepts used in the Clarifai API.

Image
============
Image is straightforward. It represents a picture in digital format. 

In Clarifai API, an image could be represented by an url, a local filename, an opened io stream, raw bytes of the image, or bytes encoded in base64.

There is a Image() class to construct an Image object in Clarifai API.

Along with the image bytes, an image could also be associated with an unique ID, as well as one or more concepts indicating what it is and what is it not.

Image object could be used in uploading, prediction, as well as search.

Input
============
Input is a generalized image, because input could be image, or video in the near future. In the current API, we use Image and Input interchangably because Image is the only Input type the API supports.

Similarly, input could be associated with ID and concepts. And Input is used in uploading, prediction, and search.

Concept
============
Concept represents a class to associate with images. It could be a concrete concept like "dog" or "cat", or a abstract concept like "love". It is the output of the concept Models.
Clarifai Concept has an unique ID, and a name. Sometimes the concept name is also referred to label or tag for the predictions. They are exchangably used somewhere.

Model
============
Model is machine learning algorithm that takes input, such as image or video, and output some results for prediction. For custom training, the models trained are all concept models. 
The output of a concept model is a list of concepts in prediction, associated with probabilities for condidence of the prediction.

Image Crop
============
Image crop is a the definition of the crop box within an image. We use this in visual search so user does not have to crop an image before the search.

We use percentage coordinates instead of pixel coordinates to specify the crop box.

A four-element-tuple represents a crop box, in (top_y, left_x, bottom_y, right_x) order.

So a (0.3, 0.2, 0.6, 0.4) represents a box horizontally spanning from 20%-40% and vertically spanning 30%-60% of the image.

