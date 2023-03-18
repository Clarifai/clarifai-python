#! Execute dataset upload using the `from_module` upload feature

from clarifai.data_upload.upload import UploadConfig

text_upload_obj = UploadConfig(
    user_id="",
    app_id="",
    pat="",
    dataset_id="",
    task="visual_clf",
    from_module="./examples/image_classification/cifar10",
    split="train",
    portal="clarifai"  #clarifai(prod), dev or staging
)
## change the task and from_module arguments in UploadConfig() to upload
## example food-101 dataset

if __name__ == "__main__":
  text_upload_obj.upload_to_clarifai()
