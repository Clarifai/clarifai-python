# Clarifai Inputs Interface


The Clarifai platform offers support for various input formats, including images, text, audio, and video files. This Software Development Kit (SDK) effectively utilizes the distinct functionalities of each input type.

## Supported Input Formats
The Clarifai platform has been designed to accommodate a diverse range of input formats, making it a versatile solution for various data types:

* **Image**: Whether you need to analyze photos, illustrations, or any other visual content, Clarifai's image processing capabilities have you covered.
* **Text**: The platform can process textual data, making it suitable for tasks like natural language processing, sentiment analysis, and more.
* **Audio**: If your project involves sound data, the Clarifai platform supports audio input, enabling applications like audio recognition and classification.
* **Video**: With support for video input, you can utilize Clarifai for tasks like video analysis, content moderation, and beyond.


## Getting Started
```cmd
export CLARIFAI_PAT={your personal access token}
```
### Image

```python
from clarifai.client.input.image import Image
image_obj = Image(user_id = 'user_id', app_id = 'demo_app')

# Upload from url
img_obj.upload_from_url(input_id='demo', url='https://samples.clarifai.com/metro-north.jpg')

# Upload from folder
img_obj.upload_from_folder(folder_path='demo_folder')

```

### Video

```python
from clarifai.client.input.video import Video
video_obj = Video(user_id = 'user_id', app_id = 'demo_app')

# Upload from filename
video_obj.upload_from_filename(input_id='demo', filename='demo.mp4')

```


### Text

```python
from clarifai.client.input.text import Text
text_obj = Text(user_id = 'user_id', app_id = 'demo_app')

# Upload text
text_obj.upload(input_id='demo', raw_text = 'This is a test')

```

### Audio

```python
from clarifai.client.input.audio import Audio
audio_obj = Audio(user_id = 'user_id', app_id = 'demo_app')

# Upload from bytes
audio = open('demo.mp3', 'rb').read()
audio_obj.upload_from_bytes(input_id = 'demo', bytes = audio)

```




## Note
Feel free to explore the code examples to better understand how to effectively utilize the platform's potential for each input type.
