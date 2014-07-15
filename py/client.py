import base64
import json
import urllib2

from mime_util import post_images_multipart

class ApiError(Exception):
  """Api error."""
  pass

SUPPORTED_OPS = ['classify','embed']

class ClarifaiApi(object):
  def __init__(self, base_url='http://clarifai.com'):
    self._base_url = base_url
    self._urls = {
      'classify': '%s/%s' % (self._base_url, 'api/call/'),
      'embed': '%s/%s' % (self._base_url, 'api/call/'),
      'upload': '%s/%s' % (self._base_url, 'api/upload/')
      }

  def _url_for_op(self, op):
    return self._urls.get(op)

  def tag_image(self, image_file):
    """Autotag an image.

    :param image_file: an open file-like object containing the encodeed image bytes. The read
    method is called on this object to get the encoded bytes so it can be a file handle or
    StringIO buffer.

    Returns:
      results: A list of (tag, probability) tuples.

    Example:
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_image(open('/path/to/local/image.jpeg'))
    """
    data = {'encoded_image': base64.encodestring(image_file.read())}
    return self._single_image_op(data, 'classify')

  def batch_tag_images(self, images):
    """Autotag an image.

    :param images: list of (file, name) tuples, where file is an open file-like object
       containing the encoded image bytes.

    Returns:
      results: A list of (tag, probability) tuples.
    """
    image_data = []
    for image_file, name in images:
      data = bytes(image_file.read())
      image_data.append((data, name))
    data = {
      'op': 'classify',
    }
    url = self._url_for_op(data['op'])
    response = post_images_multipart(image_data, data, url)
    return self._parse_response(response)

  def tag_image_url(self, image_url):
    """Autotag an image from a URL. As above, but takes an image URL."""
    data = {'image_url': image_url}
    return self._single_image_op(data, 'classify')

  def embed_image(self, image_file):
    data = {'encoded_image': base64.encodestring(image_file.read())}
    return self._single_image_op(data,'embed')

  def _single_image_op(self, data, op):
    if op not in SUPPORTED_OPS:
      raise Exception('Unsupported op: %s, ops available: %s' % (op, str(SUPPORTED_OPS)))
    headers = self._get_headers()
    data['op'] = op
    url = self._url_for_op(data['op'])
    response = self._get_response(url, data, headers)
    return self._parse_response(response, op)[0]

  def _parse_response(self, response, op):
    try:
      response = json.loads(response)
    except ValueError as e:
      raise ApiError(e)
    results = []
    if op == 'classify':
      num_imgs = len(response[op]['predictions']['classes'])
      for i in range(num_imgs):
        results.append(
            zip(response[op]['predictions']['classes'][i],
                response[op]['predictions']['probs'][i]))
    elif op == 'embed':
      results = response[op]['features']
    return results

  def _get_headers(self):
    headers = {"content-type": "application/json"}
    return headers

  def _get_response(self, url, data, headers):
    data = json.dumps(data)
    req = urllib2.Request(url, data, headers)
    try:
      response = urllib2.urlopen(req)
      raw_response = response.read()
    except urllib2.HTTPError as e:
      raise ApiError(e)
    return raw_response
