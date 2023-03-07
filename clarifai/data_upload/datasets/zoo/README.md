## Datasets Zoo

A collection of data preprocessing modules for popular public datasets to allow for compatible upload into Clarifai user app datasets.

## Usage

If a dataset module exists in the zoo, uploading the specific dataset can be easily done by simply creating a python script (or via commandline) and specifying the dataset module name in the `from_zoo` parameter of the  `UploadConfig` class .i.e.

```python
from clarifai.data_upload.upload import UploadConfig

upload_obj = UploadConfig(
	user_id="",
	app_id="",
	pat="", # Clarifai user PAT (not Clarifai app PAT)
	dataset_id="",
	task="",
	from_zoo="coco_detection",
	split="val" # train, val or test depending on the dataset
	)
# execute data upload to Clarifai app dataset
upload_obj.upload_to_clarifai()
```

## Zoo Datasets

 | dataset name | task | module name (.py) | splits |
 | --- | --- | --- | --- |
 | [COCO 2017](https://cocodataset.org/#download) | Detection | `coco_detection` | `train`, `val` |
 |        | Segmentation | `coco_segmentation` | `train`, `val` |
 |       | Captions | `coco_captions` | `train`, `val` |

## Contributing Modules

A dataset (preprocessing) module is a python script that contains a dataset class which implements data download (to download the dataset from a source to local disk dir) & extraction and dataloader methods.

The class naming convention is `<datasetname>Dataset`. The dataset class must accept `split` as the only argument in the `__init__` method and the `dataloader` method must be a generator that yields either of `VisualClassificationFeatures()`, `VisualDetectionFeatures()`, `VisualSegmentationFeatures()` or `TextFeatures()` as defined in [clarifai/data_upload/datasets/features.py](datasets/features.py). Other methods can be added as seen fit but `dataloader()` is the main method and must strictly be named `dataloader`.
Reference can be taken from the existing dataset modules in the zoo for development.
