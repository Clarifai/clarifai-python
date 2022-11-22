## Data Upload into your app dataset in the Clarifai platform

The functionality here allows a user to upload datasets of the specified types and all sizes from a local directory to the Clarifai platform datasets

Supported dataset types currently are:
* Image classification
* Object detection
* Text classification

The `datasets.py` script holds methods to preprocess input data and generate input protos that are then sent as request objects in `upload.py` to the Clarifai api to upload into a particular dataset.

## Usage

* Create a dataset under any of your apps in the Clarifai platform.

* Copy the Clarifai app id, pat, dataset id and your user id into the `config.yaml` file

* Update the respective `config.yaml` data directory variables depending on your task. See assumptions & limitations section below for the structuring of the data and `schemas` for example file structures.
	* if uploading visual detection data, an extra boolean flag indicating whether to load annotations from a text file has to be set in the `upload.py` script under;
	```python
	...
	elif task == "visual_det":
    dataset_obj = VisualDetectionDataset(
        config.data["visual_det_image_dir"],
        config.data["visual_det_labels_dir"],
        config.data["dataset_id"],
        config["split"],
        labels_from_text_file=False)
	```
	False is the default behaviour implying that read annotations from an xml file

* Set the task parameter in the `config.yaml` file to either of `text_clf`, `visual_clf` or `visual_det`. `visual_clf` is the default task. Each of these corresponds to the respective supported dataset type (See supported dataset types above).

* Finally run, ```python3 upload.py```

**NOTE**: For text classification datasets, change the base workflow in your clarifai app settings to Text for a successful upload.


## Assumptions & Limitations

The scripts currently support upload of data that has the same format as that presented in the sample files under `schemas`. Feel free to open a pull request to contribute more.

* Text and image classification data upload tasks assume that you have a csv file of your data. However if this is not the case, you may generate the required csv file first basing on the structure presented in the schemas and then proceed with the upload.

* Text classification data upload assumes a csv file with two columns named `text` and `label` with text holding the text and label the labels respectively.

* For image classification datasets, the assumption made is that you have a csv file with two string type columns named `image_path` and `label`. The image_path is the absolute path to where the image's location on your computer and the label its associated class.

* For object detection tasks, the `utils.py` script defines functions to read, preprocess data and return a dataframe with the image path, labels and annotations. The funtions are built on the assumption that your annotations files have the same structure as either of xml or text_file annotations presented in `schemas`. The image dimensions and all elements present in the sample xml file in `schemas` must be present in your xml file as well. For the text file annotations, the bounding box coordinates should be already computed proportions.

* Both image and object detection utils here assume that the images and corresponding annotations/label files have the same naming with the exception of file extension. Since no image input validation is done currently,  ensure that all images are valid and have the correct file extensions.
