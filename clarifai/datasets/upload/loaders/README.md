## Dataset Loaders

A collection of data preprocessing modules for popular public datasets to allow for compatible upload into Clarifai user app datasets.

## Usage

If a dataset module exists in the zoo, uploading the specific dataset can be easily done by simply creating a python script (or via commandline) and specifying the dataset module name in the `dataset_loader` parameter of the  `Dataset` class, `upload_dataset` method .i.e.

```python
from clarifai.client.app import App

app = App(app_id="", user_id="")
# Create a dataset in Clarifai App
dataset = app.create_dataset(dataset_id="")
# execute data upload to Clarifai app dataset
dataset.upload_dataset(task='visual_segmentation', split="train", dataset_loader='coco_segmentation')
```

## Dataset Loaders

 | dataset name | task | module name (.py) | splits |
 | --- | --- | --- | --- |
 | [COCO 2017](https://cocodataset.org/#download) | Detection | `coco_detection` | `train`, `val` |
 |        | Segmentation | `coco_segmentation` | `train`, `val` |
 |       | Captions | `coco_captions` | `train`, `val` |
 |[xVIEW](http://xviewdataset.org/)  | Detection | `xview_detection` | `train`
 | [ImageNet](https://www.image-net.org/)  | Classification | `imagenet_classification` | `train`
## Contributing Modules

A dataloader (preprocessing) module is a python script that contains a dataloader class which implements data download (to download the dataloader from a source to local disk dir) & extraction and dataloader methods.

The class naming convention is `<datasetname>DataLoader`. The dataset class must accept `split` as the only argument in the `__init__` method and the `__getitem__` method must return either of `VisualClassificationFeatures()`, `VisualDetectionFeatures()`, `VisualSegmentationFeatures()` or `TextFeatures()` as defined in [clarifai/datasets/upload/features.py](../features.py). Other methods can be added as seen fit but must be inherited from parent `ClarifaiDataLoader` base class [clarifai/datasets/upload/base.py](../base.py).
Reference can be taken from the existing dataset modules in the zoo for development.

## Notes

* Dataloaders in the zoo by default first create a `data` directory in the zoo directory then download the data into this `data` directory, preprocess the data and finally execute upload to a Clarifai app dataset. For instance with the COCO dataset modules above, the coco2017 dataset is by default downloaded first into a `data` directory, extracted and then preprocessing is performed on it and finally uploaded to Clarifai.

* Taking the above into consideration, to avoid the scripts re-downloading data you already have locally, create a `data` directory in the loaders directory and move your extracted data there. **Ensure that the extracted folder/file names and file structure MATCH those when the downloaded zips are extracted.**

* COCO Format: To reuse the coco modules above on your coco format data, ensure the criteria in the two points above is adhered to first. If so, pass the coco module name from any of the above in the loaders to the `dataset_loader=` parameter in `upload_dataset()`.

* xVIEW Dataset: To upload, you have to register and download images,label from [xviewdataset](http://xviewdataset.org/#dataset) follow the above mentioned steps to place extracted folder in `data` directory. Finally pass the xview module name to `dataset_loader=` parameter in `upload_dataset()`.

* ImageNet Dataset: ImageNet Dataset should be downloaded and placed in the 'data' folder along with the [label mapping file](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=LOC_synset_mapping.txt).

		<data>/
      	├── train/
      	├── LOC_synset_mapping.txt
