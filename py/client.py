import base64
import json
import httplib
import urllib2
from urlparse import urlparse

class ApiError(Exception):
  """Api error."""
  pass

class ClarifaiApi(object):
  def __init__(self, base_url='http://clarifai.com'):
    self._base_url = base_url
    self._urls = {
      'classify': '%s/%s' % (self._base_url, 'api/call/')
      }

  def _url_for_op(self, op):
    return self._urls.get(op)

  def tag_image(self, image_file):
    """Autotag an image.

    Args:
      image_file: an open file-like object containing the encodeed image bytes. The read method is
      called on this object to get the encoded bytes so it can be a file handle or StringIO buffer.

    Returns:
      results: A list of (tag, probability) tuples.

    Example:
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_image(open('/path/to/local/image.jpeg'))
    """
    data = {'encoded_image': base64.encodestring(image_file.read())}
    return self._classify_image(data)

  def tag_image_url(self, image_url):
    """Autotag an image from a URL. As above, but takes an image URL."""
    data = {'image_url': image_url}
    return self._classify_image(data)

  def _classify_image(self, data):
    headers = self._get_headers()
    data['op'] = 'classify'
    url = self._url_for_op(data['op'])
    response = self._get_response(url, data, headers)
    return zip(response['classify']['predictions']['classes'][0],
               response['classify']['predictions']['probs'][0])

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
    try:
      return json.loads(raw_response)
    except ValueError as e:
      raise ApiError(e)
