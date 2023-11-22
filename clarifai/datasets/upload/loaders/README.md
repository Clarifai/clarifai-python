## Dataset Loaders

A collection of data preprocessing modules for popular public datasets to allow for compatible upload into Clarifai user app datasets.

## Usage

If a dataset module exists in the zoo, uploading the specific dataset can be easily done by simply creating a python script (or via commandline) and specifying the dataloader object in the `dataloader` parameter of the  `Dataset` class, `upload_dataset` method .i.e.

```python
from clarifai.client.app import App
from clarifai.datasets.upload.loaders.coco_segmentation import COCOSegmentationDataLoader

app = App(app_id="", user_id="")
# Create a dataset in Clarifai App
dataset = app.create_dataset(dataset_id="")
# instantiate dataloader object
coco_seg_dataloader = COCOSegmentationDataLoader(images_dir="", label_filepath="")
# execute data upload to Clarifai app dataset
dataset.upload_dataset(dataloader=coco_seg_dataloader)
```

## Dataset Loaders

 | dataset name | task | module name (.py)
 | --- | --- | ---
 | [COCO 2017](https://cocodataset.org/#download) | Detection | `coco_detection` |
 |        | Segmentation | `coco_segmentation` |
 |       | Captions | `coco_captions` |
 |[xVIEW](http://xviewdataset.org/)  | Detection | `xview_detection` |
 | [ImageNet](https://www.image-net.org/)  | Classification | `imagenet_classification` |
## Contributing To Loaders

A dataloader (preprocessing) module is a python script that contains a dataloader class which implements dataloader methods.

The class naming convention is `<datasetname>DataLoader`. The dataset class must inherit from `ClarifaiDataLoader` and the `__getitem__` method must return either of `VisualClassificationFeatures()`, `VisualDetectionFeatures()`, `VisualSegmentationFeatures()` or `TextFeatures()` as defined in [clarifai/datasets/upload/features.py](../features.py). Other methods can be added as seen fit but must be inherited from parent `ClarifaiDataLoader` base class [clarifai/datasets/upload/base.py](../base.py).
Reference can be taken from the existing dataset modules in the zoo for development.

## Notes

* COCO Format: To reuse the coco modules above on your coco format data, ensure the criteria in the two points above is adhered to first. If so, pass the coco images_dir and labels_filepath from any of the above in the loaders to the `dataloader=` parameter in `upload_dataset()`.

* xVIEW Dataset: To upload, you have to register and download images,label from [xviewdataset](http://xviewdataset.org/#dataset) follow the above mentioned steps to place extracted folder in `data` directory. Finally pass the xview data_dir to `dataloader=` parameter in `upload_dataset()`.

		<data>/
      	├── train_images/
      	├── xview_train.geojson

* ImageNet Dataset: ImageNet Dataset should be downloaded and placed in the 'data' folder along with the [label mapping file](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=LOC_synset_mapping.txt).

		<data>/
      	├── train/
      	├── LOC_synset_mapping.txt
