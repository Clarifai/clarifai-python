## Visual Detection Triton Model Examples
These can be used on the fly with minimal or no changes to test deploy visual detection models to the Clarifai platform. See the required files section for each model below.

## [YOLOF](https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc3/configs/yolof)

Requirements to run tests locally:

Download checkpoint and save it in yolof/1/config/:
```bash
wget -P yolof/1/config https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth
```
Install dependecies
```bash
pip install -r yolof/requirements.txt
```
Test locally:
```bash
pip install pytest
python -m pytest -s yolof/1/test.py
```
Zip it and upload to Clarifai
```bash
clarifai-triton-zip --triton_model_repository yolof --zipfile_name yolof.zip
# then upload your zip to cloud and obtain url
clarifai-upload-model --model_type visual-detector --model_id <your model id> --url <url>
```

## Torch serve model format [Faster-RCNN](https://github.com/pytorch/serve/tree/master/examples/object_detector/fast-rcnn)
To utilize a Torch serve model (.mar file) created by running torch-model-archiver – essentially a zip file containing the model checkpoint, Python code, and other components – within this module, follow these steps:

1. Unzip the .mar file to obtain your checkpoint.
2. Implement your postprocess method in inference.py.

For example: faster-rcnn, suppose you already have .mar file following the torch serve example

unzip it to ./faster-rcnn_torchserve/1/model_store/hub/checkpoints
```bash
unzip faster_rcnn.mar -d ./faster-rcnn_torchserve/1/model_store/
```

```bash
# in model_store you will have
model_store/
├── MAR-INF
│   └── MANIFEST.json
├── model.py
└── fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```
```bash
# then relocate the checkpoint to ./faster-rcnn_torchserve/1/model_store/hub/checkpoints
# as the Torch cache is configured to use this folder in inference.py.
mv ./faster-rcnn_torchserve/1/model_store/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ./faster-rcnn_torchserve/1/model_store/hub/checkpoints/
```

test, zip and upload the model
```bash
# zip
clarifai-triton-zip --triton_model_repository faster-rcnn_torchserve --zipfile_name faster-rcnn_torchserve.zip
# then upload your zip to cloud and obtain url
clarifai-upload-model --model_type visual-detector --model_id <your model id> --url <url>
```
