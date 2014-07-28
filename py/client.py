import base64
import json
import urllib2

from mime_util import post_images_multipart

class ApiError(Exception):
  """Api error."""
  pass

SUPPORTED_OPS = ['classify','embed','classify,embed']

class ClarifaiApi(object):
  def __init__(self, base_url='http://clarifai.com'):
    self._base_url = base_url
    self._urls = {
      'classify': '%s/%s' % (self._base_url, 'api/call/'),
      'embed': '%s/%s' % (self._base_url, 'api/call/'),
      'classify,embed': '%s/%s' % (self._base_url, 'api/call/'),
      'upload': '%s/%s' % (self._base_url, 'api/upload/')
      }

  def _url_for_op(self, op):
    return self._urls.get(op)

  def tag_image(self, image_file):
    """Autotag an image.

    Args:
      image_file: an open file-like object containing the encodeed image bytes. The read
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

  def embed_image(self, image_file):
    data = {'encoded_image': base64.encodestring(image_file.read())}
    return self._single_image_op(data, 'embed')

  def tag_images(self, image_files):
    """Autotag a batch of image file objects.

    Args:
      image_files: list of (file, name) tuples, where file is an open file-like object
        containing the encoded image bytes.

    Returns:
      results: A list of (tag, probability) tuples.

    Example:
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                               open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, 'classify')

  def embed_images(self, image_files):
    return self._multi_image_op(image_files, 'embed')

  def batch_tag_and_embed_images(self, image_files):
    return self._multi_image_op(image_files, 'classify,embed')


  def _multi_image_op(self, image_files, op):
    ''' Supports both list of tuples (image_file, name) or a list of image_files where a name will
    be created as the index into the list. '''
    if not isinstance(image_files, list):
      image_files = [image_files]
    image_data = []
    c = 0
    for image_file in image_files:
      if isinstance(image_file, tuple):
        image_data.append((bytes(image_file[0].read()), image_file[1]))
      else:
        data = bytes(image_file.read())
        image_data.append((data, str(c)))
      c += 1
    data = {'op': op}
    url = self._url_for_op(op)
    response = post_images_multipart(image_data, data, url)
    return self._parse_response(response, op)

  def tag_image_url(self, image_url):
    """Autotag an image from a URL. As above, but takes an image URL."""
    data = {'image_url': image_url}
    return self._single_image_op(data, 'classify')

  def embed_image_url(self, image_url):
    """ Embed an image from a URL. As above, but takes an image URL."""
    data = {'image_url': image_url}
    return self._single_image_op(data, 'embed')

  def batch_tag_and_embed_imageurls(self, image_urls):
    ''' Take in a list of image urls, downloading them on the server side and returning both
    classifications and embeddings. '''
    data = {'image_url': image_urls}
    return self._multi_imagepath_op(data, 'classify,embed')

  def _single_image_op(self, data, op):
    if op not in SUPPORTED_OPS:
      raise Exception('Unsupported op: %s, ops available: %s' % (op, str(SUPPORTED_OPS)))
    data['op'] =  op
    headers = self._get_headers()
    url = self._url_for_op(data['op'])
    response = self._get_response(url, data, headers)
    return dict([(k, v[0]) for k, v in self._parse_response(response, op).items()])

  def _multi_imagepath_op(self, data, op):
    ''' If sending image_url or image_file strings, then we can send as json directly instead of the
    multipart form. '''
    if op not in SUPPORTED_OPS:
      raise Exception('Unsupported op: %s, ops available: %s' % (op, str(SUPPORTED_OPS)))
    data['op'] =  op
    headers = self._get_headers()
    url = self._url_for_op(data['op'])
    response = self._get_response(url, data, headers)
    return self._parse_response(response, op)

  def _parse_response(self, response, all_ops):
    try:
      response = json.loads(response)
    except ValueError as e:
      raise ApiError(e)
    results = {}
    for op in all_ops.split(','):
      op_results = []
      if op == 'classify':
        num_imgs = len(response[op]['predictions']['classes'])
        for i in range(num_imgs):
          op_results.append(
              zip(response[op]['predictions']['classes'][i],
                  response[op]['predictions']['probs'][i]))
      elif op == 'extract':
        op_results = response[op]['features']
      elif op == 'embed':
        op_results = response[op]['embeddings']
      results[op] = op_results
    return results

  def _get_headers(self):
    headers = {"content-type": "application/json"}
    return headers

  def _get_response(self, url, data, headers):
    # data['blah'] = ['http://davidfeldmanshow.com/wp-content/uploads/2014/01/dogs-wallpaper.jpg' for
    # i in range(100)]
    # print data
    data = json.dumps(data)
    # import pdb
    # pdb.set_trace()
    req = urllib2.Request(url, data, headers)
    try:
      response = urllib2.urlopen(req)
      raw_response = response.read()
    except urllib2.HTTPError as e:
      raise ApiError(e)
    return raw_response
