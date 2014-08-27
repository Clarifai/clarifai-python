
This Python client provides a simple wrapper around our powerful image recognition API.

The constructor takes your app_id and app_secrete created in your Clarifai Account. You can also
set these variables as environment variables using:
CLARIFAI_APP_ID
CLARIFAI_APP_SECRET

This client uses your ID and secret to get an access token. Since this expires every so often, the
client is setup to renew the token for you automatically using your credentials so you don't have
to worry about it. Additionally, we highly recomment that you have PIL or Pillow installed if you
plan to send images from your local machine to our service. This client will automatically
determine you allowed limits and resize any images you wish to process automatically before
sending. If you do not have PIL or Pillow then you must do this yourself to ensure your API calls
are processed without fail.

As an example, to tag an image on your local drive you can do the following:

<pre>
from api.py.client import ClarifaiApi
clarifai_api = ClarifaiApi()
result = clarifai_api.tag_images(open('/path/to/local/image.jpeg'))
</pre>

This will return the tagging result for the given image read off your local storage system. The
operations supported by the client can all handle batches of images. Keeping tagging as the
running example, this would look like:

<code>
result = clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                                  open('/path/to/local/image2.jpeg')])
</code>

The result will now contain all the results of the tagging for each image in the batch. When
sending large batches of images, you must adhere to your application limits for the maximum batch
size per request.
