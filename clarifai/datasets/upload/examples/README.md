## Dataset upload from local directory

Examples of how to upload your local directory datasets into clarifai app using `module_dir` feature from `Dataset`.

**Note:**
**Note:**

- Ensure that the `CLARIFAI_PAT` environment variable is set.
- Ensure that the appropriate base workflow is being set for indexing respective input type.


## Image Classification - Cifar10
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="visual_classification", split="train", module_dir="path_to_cifar10_module")
```

## Image Classification - [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="visual_classification", split="train", module_dir="path_to_food-101_module")
```

## Text Classification - IMDB Reviews
```python
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(task="text_clf", split="train", module_dir="path_to_imdb_reviews_module")
```
