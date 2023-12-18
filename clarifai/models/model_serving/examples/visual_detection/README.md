## Visual Detection Triton Model Examples

These can be used on the fly with minimal or no changes to test deploy visual detection models to the Clarifai platform. See the required files section for each model below.

* ### [Yolov5x](./yolov5x/)

	Required files (not included here due to upload size limits):

	* Download the `Yolov5 repo` and the `yolov5-x checkpoint` and store them under the `1/` directory of the yolov5x folder.
		```
		cd yolov5x/1/
		git clone https://github.com/ultralytics/yolov5.git
		wget -O model.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
		```
	* zip and test deploy to your Clarifai app

* ### Torch serve model format [yolov8](./yolov8_torchserve/)

	To utilize a Torch serve model (`.mar` file) created by running `torch-model-archiver` – essentially a zip file containing the model checkpoint, Python code, and other components – within this module, follow these steps:

	1. Unzip the `.mar` file to obtain **your checkpoint**.
	2. Implement your postprocess method in `inference.py`.

	For example: [yolov8](./yolov8_torchserve/), suppose you already have `.mar` file following the [torch serve example](https://github.com/pytorch/serve/blob/master/examples/object_detector/yolo/yolov8/README.md)

	* unzip it to `1/model_store`
	```bash
	unzip yolov8n.mar -d 1/model_store/
	```
	```bash
	# in 1/model_store you will have
	1/model_store/
	├── MAR-INF
	│   └── MANIFEST.json
	├── custom_handler.py
	└── yolov8n.pt
	```

	* test, zip and upload the model
