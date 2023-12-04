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
