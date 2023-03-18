## Data Upload into your app dataset in the Clarifai platform

The functionality here allows a user to upload datasets of the specified types and all sizes from a local directory to the Clarifai platform datasets

Supported dataset types currently are:
* Image classification
* Object detection
* Text classification
* Image segmentation

The `datasets` package holds methods to preprocess input data and generate input protos that are then sent as request objects in `upload.py` to the Clarifai api to upload into a particular dataset.

## Usage

* Create a dataset under any of your apps in the Clarifai platform.

#### Upload dataset from dataset package
* To upload the dataset from a (python)package, create a folder with the structure and files as below.

- Package Structure:
  ---------------------------
      <folder_name>/
      ├──__init__.py
      ├── <Your local dir dataset>/
      └──<Your local dir dataset>/dataset.py
  `dataset.py` must implement a class named following the convention, `<dataset_name>Dataset`. This class must accept `split` as the only argument in the `__init__` method and must have a `dataloader()` generator method that formats your local dir dataset and yields either of `VisualClassificationFeatures()`, `VisualDetectionFeatures()`, `VisualSegmentationFeatures()` or `TextFeatures()` as defined in [clarifai/data_upload/datasets/features.py](datasets/features.py). Other methods can be added in the class as seen fit but `dataloader()` is the main method and must be named dataloader.

- In a python script (or in the commandline), import the `UploadConfig` class from upload module and then specify the dataset module path in the `from_module` parameter of the  `UploadConfig` .i.e.

	```python
	from clarifai.data_upload.upload import UploadConfig

	upload_obj = UploadConfig(
		user_id="",
		app_id="",
		pat="", # Clarifai user PAT (not Clarifai app PAT)
		dataset_id="",
		task="<task-name>", # see supported tasks below
		from_module="./path/to/dataset_package/<package-folder-name>",
		split="val" # train, val or test depending on the dataset
		)
	# execute data upload to Clarifai app dataset
	upload_obj.upload_to_clarifai()
	```
	See `examples/` and `examples.py` for reference.

For data upload from dataset zoo, see [clarifai/data_upload/datasets/zoo](datasets/zoo)
* Supported tasks:
	* `text_clf` for text classification.
	* `visual_clf` for image classification.
	* `visual_detection` for object detection.
	* `visual_segmentation` for image segmentation.


**NOTE**: For text classification datasets, change the base workflow in your clarifai app settings to a Text workflow for a successful upload.

## Notes

* For datasets not available in the datasets zoo, the user has to handle the preprocessing for their local datasets to convert them into compatible upload formats.

* An individual image can have multiple bounding boxes for the same or different classes and so `VisualDetectionFeatures()` classes and bounding boxes lists must match in length with each element of bounding boxes being a list of bbox coordinates ([`x_min, y_min, x_max, y_max`]) corresponding to a single class name in class_names.

* For Segmentation tasks, a single image can have multiple masks corresponding to different or the same classes, hence `VisualSegmentationFeatures()` classes and polygons must be lists of the same length as well. Polygons in turn contain lists with each list in turn having an `[x, y]` list points.
