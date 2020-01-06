==============
Basic Concepts
==============

This page lists a few basic notions used in the Clarifai API.

Image
============
Image is a straightforward notion. It represents a picture in a digital format.

In the Clarifai API, an image could be represented by a url, a local filename, raw bytes of the image, or bytes encoded in base64. To construct a new Image object, use the Image() constructor.

Video
============
In the Clarifai API, Video is considered a sequence of frames where each frame represents one second of the video. This means
that after running prediction, the models will return results for every second of the video.

Video is used similarly to Image. Video can be represented by a url, a local filename, raw bytes of the video, or bytes encoded in base64. To construct a new Video object, use the Video() constructor.

Input
============
Input is a more general notion; it can be either Image or Video. Input is used for uploading, prediction, search, etc.

Each Input has a unique ID and can be associated with Concepts.

Concept
============
Concept represents a word to associate with Inputs. Concept can be a concrete notion like "dog" or "cat", or a abstract notion like "love".
All models of type `concept` return a set of Concepts in the prediction for the given Input.
Each Concept has a unique ID, and a name (in the data science nomenclature also sometimes referred to as a label, or a tag).

Model
============
Model is a machine learning algorithm that takes Input, such as Image or Video, runs a prediction, and outputs the results.
There are several types of models: Concept Model, Face Detection Model, Color Model, etc. They all return different types
of results. For example, Concept Model returns associated Concepts for each Input, Face Detection Model returns the locations of
faces, Color Model returns dominant colors, etc.

There are public (pre-defined) models you can use, and custom models that you train yourself with your own Inputs.

Custom models
============
Custom models must be of type `concept`.

When creating a custom model, you provide it with your Inputs and associated Concepts.
After being trained, the model can predict what Concepts are associated with never-before-seen Inputs, together
with probabilities for confidence of the prediction.

Models can also be evaluated for measuring their prediction capability.

Workflow
============
Workflow enables you to run prediction on Inputs using several Models in one request.
