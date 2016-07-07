Clarifai Python Client
====================

This Python client provides a simple wrapper around our powerful image recognition <a href="http://developer.clarifai.com">API</a>.

The constructor takes your APP_ID and APP_SECRET created in your Clarifai Account. You can also
set these variables in your environment as:
CLARIFAI_APP_ID
CLARIFAI_APP_SECRET

This client uses your APP_ID and APP_SECRET to get an access token. Since this expires every so
often, the client is setup to renew the token for you automatically using your credentials so you
don't have to worry about it. Additionally, we highly recommend that you have PIL or Pillow
installed if you plan to send images from your local machine to our service. The client will
automatically determine your allowed limits and resize any images you wish to process automatically
before sending. If you do not have PIL or Pillow then you must do this yourself to ensure your API
calls are processed without fail.

Installation
---------------------
<pre>
pip install git+git://github.com/Clarifai/clarifai-python.git
export CLARIFAI_APP_ID=&lt;an_application_id_from_your_account&gt;
export CLARIFAI_APP_SECRET=&lt;an_application_secret_from_your_account&gt;
</pre>


Tag API Usage
---------------------

#### Tag from disk

A complete example of using this Python client is as follows. Suppose you want to tag an image on
your local drive:

<pre>

from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi() # assumes environment variables are set.
result = clarifai_api.tag_images(open('/path/to/local/image.jpeg', 'rb'))
</pre>

This will return the tagging result for the given image read off your local storage system (see the
<a href="https://developer.clarifai.com/guide/">Docs</a> for response format). The operations
supported by the client can all handle batches of images. Keeping tagging as the running example,
this would look like:

<pre>
result = clarifai_api.tag_images([open('/path/to/local/image.jpeg', 'rb'),
                                  open('/path/to/local/image2.jpeg', 'rb')])
</pre>
The result will now contain all the results of the tagging for each image in the batch. When
sending large batches of images, you must adhere to your application limits for the maximum batch
size per request.

#### Tag from URL

If your images live remotely at a public url, you can also use tag_image_urls:
<pre>
from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi()  # assumes environment variables are set.
result = clarifai_api.tag_image_urls('http://www.clarifai.com/img/metro-north.jpg')
</pre>

If you have multiple urls to tag, you can also call tag_image_urls with an array of urls:
<pre>
from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi()  # assumes environment variables are set.
result = clarifai_api.tag_image_urls(['http://www.clarifai.com/static/img_ours/autotag_examples/metro-north.jpg',
                                      'http://www.clarifai.com/static/img_ours/autotag_examples/dog.jpg'])
</pre>


Color API Usage
---------------------

#### Color from disk

color API is similar to tag API. Suppose you want to get color from an image on
your local drive:

<pre>

from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi() # assumes environment variables are set.
result = clarifai_api.color(open('/path/to/local/image.jpeg', 'rb'))
</pre>

This will return the color result for the given image read off your local storage system (see the
<a href="https://developer.clarifai.com/guide/color#color">Docs</a> for response format). The operations
supported by the client can all handle batches of images. Keeping tagging as the running example,
this would look like:

<pre>
result = clarifai_api.color([open('/path/to/local/image.jpeg', 'rb'),
                             open('/path/to/local/image2.jpeg', 'rb')])
</pre>
The result will now contain all the results of the colors for each image in the batch. When
sending large batches of images, you must adhere to your application limits for the maximum batch
size per request.

#### Color from URL

If your images live remotely at a public url, you can also use color_urls:
<pre>
from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi()  # assumes environment variables are set.
result = clarifai_api.color_urls('http://www.clarifai.com/img/metro-north.jpg')
</pre>

If you have multiple urls to color, you can also call color_urls with an array of urls:
<pre>
from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi()  # assumes environment variables are set.
result = clarifai_api.color_urls(['http://www.clarifai.com/static/img_ours/autotag_examples/metro-north.jpg',
                                  'http://www.clarifai.com/static/img_ours/autotag_examples/dog.jpg'])
</pre>

The same result format is returned whether provided image bytes or urls.

Please check out the full documentation for our API at <a href="https://developer.clarifai.com/docs">developer.clarifai.com</a>.
