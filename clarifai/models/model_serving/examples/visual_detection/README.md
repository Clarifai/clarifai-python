## Visual Detection Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy visual detection models to the Clarifai platform. See the required files section for each model below.

* [YOLOF](https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc3/configs/yolof)

	Requirements to run tests locally:

	- Download checkpoint and save it in `yolof/1/config/`:
	```bash
	wget -P yolof/1/config https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth
	```

	- Install dependecies
	```bash
	pip install -r yolof/requirements.txt
	```

	- Test locally:
	```bash
	pip install pytest
	python -m pytest -s yolof/1/test.py
	```

	- Zip it and upload to Clarifai
	```bash
	clarifai-triton-zip --triton_model_repository yolof --zipfile_name yolof.zip
	# then upload your zip to cloud and obtain url
	clarifai-upload-model --model_type visual-detector --model_id <your model id> --url <url>
	```
