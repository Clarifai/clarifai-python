import base64, json, os, urllib2, urllib

try:
  from PIL import Image
  CAN_RESIZE = True
except Exception, e:
  CAN_RESIZE = False
  print ("It is recommended to install PIL with the desired image format support so that "
         "image resizing to the correct dimesions will be handled for you.")
from cStringIO import StringIO
from mime_util import post_images_multipart

class ApiError(Exception):
  """Api error."""
  def __init__(self, msg):
    self.msg = msg
  def __str__(self):
    return repr(self.msg)

SUPPORTED_OPS = ['tag','embed']

IM_QUALITY = 95
API_VERSION = 'v1'

class ClarifaiApi(object):
  """
  The constructor for API access. You must sign up at developer.clarifai.com first and create an
  application in order to generate your credentials to provide here.
  Args:
    app_id: the client_id for an application you've created in your Clarifai account.
    app_secret: the client_secret for the same application.
  """
  def __init__(self, app_id=None, app_secret=None, base_url='https://api.clarifai.com'):
    if app_id is None:
      self.CLIENT_ID = os.environ.get('CLARIFAI_APP_ID', None)
    else:
      self.CLIENT_ID = app_id
    if app_secret is None:
      self.CLIENT_SECRET = os.environ.get('CLARIFAI_APP_SECRET', None)
    else:
      self.CLIENT_SECRET = app_secret

    self._base_url = base_url
    self._urls = {
      'tag': os.path.join(self._base_url, '%s/tag/' % API_VERSION),
      'embed': os.path.join(self._base_url, '%s/embed/' % API_VERSION),
      'multiop': os.path.join(self._base_url, '%s/multiop/' % API_VERSION),
      'token': os.path.join(self._base_url, '%s/token/' % API_VERSION),
      'info': os.path.join(self._base_url, '%s/info/' % API_VERSION),
      }
    self.access_token = None
    self.api_info = None

  def get_access_token(self, renew=False):
    """ Get an access token using your app_id and app_secret.

    Args:
      renew: if True, then force the client to get a new token (even if not expired). By default if
      there is already an access token in the client then this method is a no-op.
    """
    if self.access_token is None or renew:
      headers = {}  # don't use json here, juse urlencode.
      url = self._url_for_op('token')
      data = urllib.urlencode({'grant_type': 'client_credentials',
                               'client_id':self.CLIENT_ID,
                               'client_secret':self.CLIENT_SECRET})
      req = urllib2.Request(url, data, headers)
      try:
        response = urllib2.urlopen(req).read()
        response = json.loads(response)
      except urllib2.HTTPError as e:
        raise ApiError(e.reason)
      except Exception, e:
        raise ApiError(e)
      self.access_token = response['access_token']
    return self.access_token

  def get_info(self):
    url = self._url_for_op('info')
    data= None # This will be a GET request since data is None
    response = self._get_raw_response(self._get_json_headers,
                                      self._get_json_response, url, data)
    response = json.loads(response)
    self.api_info = response['results']
    return self.api_info

  def _url_for_op(self, ops):
    if not isinstance(ops, list):
      ops = [ops]
    if len(ops) > 1:
      return self._urls.get('multiop')
    else:
      return self._urls.get(ops[0])

  def tag_images(self, image_files):
    """ Autotag a single image from an open file object or multiples images from a list of open file
    objects.

    The only method used on the file object is read() to get the bytes of the compressed
    image representation.

    Args:
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
      open file-like object containing the encoded image bytes.

    Returns:
      results: a tuple of (tag, probability) if a single image is processed or a list of (tag,
      probability) tuples if multiple images are processed.

    Example:
      from api.py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                               open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, ['tag'])

  def embed_images(self, image_files):
    """ Embed a single image from an open file object or multiples images from a list of open file
    objects.

    The only method used on the file object is read() to get the bytes of the compressed
    image representation.

    Args:
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
      open file-like object containing the encoded image bytes.

    Returns:
      results: a tuple of (tag, probability) if a single image is processed or a list of (tag,
      probability) tuples if multiple images are processed.

    Example:
      from api.py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                               open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, ['embed'])

  def tag_and_embed_images(self, image_files):
    return self._multi_image_op(image_files, ['tag','embed'])

  def tag_image_urls(self, image_urls):
    """ Tag an image from a url or images from a list of urls.
      image_urls: a single url for the input image to be processed or a list of urls for a set of
      images to be processed.

    Returns:
      results: a (tag, probability) tuple if a single image was used, or a list of (tag,
      probability) tuples when multiple images are input.

    Example:
      from api.py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_image_url(['http://www.clarifai.com/img/metro-north.jpg',
                                  'http://www.clarifai.com/img/metro-north.jpg'])

    """
    return self._multi_imageurl_op(image_urls, ['tag'])

  def embed_image_urls(self, image_urls):
    """ Embed an image from a url or images from a list of urls.

    Args:
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed.

    Returns:

    """
    return self._multi_imageurl_op(image_urls, ['embed'])

  def tag_and_embed_image_urls(self, image_urls):
    """ Take in a list of image urls, downloading them on the server side and returning both
    classifications and embeddings.

    Args:
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed.

    Returns:

    """
    return self._multi_imageurl_op(image_urls, ['tag','embed'])

  def _resize_image_tuple(self, image_tup):
    """ Resize the (image, name) so that it falls between MIN_SIZE and MAX_SIZE as the minimum
    dimension.
    """
    if self.api_info is None:
      self.get_info()  # sets the image size and other such info from server.
    try:
      MIN_SIZE = self.api_info['min_image_size']
      MAX_SIZE = self.api_info['max_image_size']
      img = Image.open(image_tup[0])
      ms = min(img.size)
      min_ratio = float(MIN_SIZE) / ms
      max_ratio = float(MAX_SIZE) / ms
      def get_newsize(img, ratio, SIZE):
        if img.size[0] == ms:
          newsize = (SIZE, int(round(ratio * img.size[1])))
        else:
          newsize = (int(round(ratio * img.size[0])), SIZE)
        return newsize
      im_changed = False
      # Only resample if min size is > 512 or < 256
      if max_ratio < 1.0:  # downsample to MAX_SIZE
        newsize = get_newsize(img, max_ratio, MAX_SIZE)
        img = img.resize(newsize, Image.BILINEAR)
        im_changed = True
      elif min_ratio > 1.0:  # upsample to MIN_SIZE
        newsize = get_newsize(img, min_ratio, MIN_SIZE)
        img = img.resize(newsize, Image.BICUBIC)
        im_changed = True
      else:  # no changes needed so rewind file-object.
        image_tup[0].seek(0)
      # Finally make sure we have RGB images.
      if img.mode != "RGB":
        img = img.convert("RGB")
        im_changed = True
      if im_changed:
        io = StringIO()
        img.save(io, 'jpeg', quality=IM_QUALITY)
        io.seek(0)  # rewind file-object to read() below is good to go.
        image_tup = (io, image_tup[1])
    except IOError, e:
      print "Could not open image file: %s, still sending to server." % image_tup[1]
    return image_tup

  def _process_image_files(self, input_files):
    # Handle single file-object as arg.
    if not isinstance(input_files, list):
      input_files = [input_files]
    # Handle unnames images as lists of file objects. Named by index in list.
    image_files = []
    for i, tup in enumerate(input_files):
      if not isinstance(tup, tuple):
        image_files.append((tup, str(i)))
        assert hasattr(image_files[i][0], 'read'), (
            'image_files[%d] has wrong type: %s. Must be file-object with read method.') % (
                i, type(image_files[i][0]))
      else:  # already tuples passed in.
        image_files.append(tup)
    # Resize any images such that the min dimension is in range.
    if CAN_RESIZE:
      for i, image_tup in enumerate(image_files):
        image_files[i] = self._resize_image_tuple(image_tup)
    # Return a list of (bytes, name) tuples of the encoded image bytes.
    image_data = []
    for image_file in image_files:
      image_data.append((bytes(image_file[0].read()), image_file[1]))
    return image_data

  def _multi_image_op(self, image_files, ops):
    """ Supports both list of tuples (image_file, name) or a list of image_files where a name will
    be created as the index into the list. """
    if len(set(ops).intersection(SUPPORTED_OPS)) != len(ops):
      raise Exception('Unsupported op: %s, ops available: %s' % (str(ops), str(SUPPORTED_OPS)))
    image_data = self._process_image_files(image_files)
    data = {'op': ','.join(ops)}
    url = self._url_for_op(ops)
    raw_response = self._get_raw_response(self._get_multipart_headers,
                                          post_images_multipart, image_data, data, url)
    return self._parse_response(raw_response, ops)

  def _multi_imageurl_op(self, image_urls, ops):
    """ If sending image_url or image_file strings, then we can send as json directly instead of the
    multipart form. """
    if len(set(ops).intersection(SUPPORTED_OPS)) != len(ops):
      raise Exception('Unsupported op: %s, ops available: %s' % (str(ops), str(SUPPORTED_OPS)))
    if not isinstance(image_urls, list):
      image_urls = [image_urls]
    if not isinstance(image_urls[0], basestring):
      raise Exception("image_urls must be strings")
    data =  {'op': ','.join(ops),
             'url': image_urls}
    url = self._url_for_op(ops)
    raw_response = self._get_raw_response(self._get_json_headers,
                                          self._get_json_response, url, data)
    return self._parse_response(raw_response, ops)

  def _parse_response(self, response, all_ops):
    try:
      parsed_response = json.loads(response)
    except Exception, e:
      raise ApiError(e)
    if 'error' in parsed_response:  # needed anymore?
      raise ApiError(parsed_response['error'])
    # Return the true API return value.
    return parsed_response

  def _get_authorization_headers(self):
    access_token = self.get_access_token()
    return {'Authorization': 'Bearer %s' % access_token}

  def _get_multipart_headers(self):
    return self._get_authorization_headers()

  def _get_json_headers(self):
    headers = self._get_authorization_headers()
    headers['content-type'] = 'application/json'
    return headers

  def _get_raw_response(self, header_func, request_func, *args):
    """ Get a raw_response from the API, retrying on TOKEN_EXPIRED errors.

    Args:
      header_func: function to generate dict of HTTP headers for this request, passed as kwarg to
                   request_func.
      request_func: function to make the request, using the remaining args.
      args: passed to request_func.
    """
    headers = header_func()
    attempts = 3
    while attempts > 0:
      attempts -= 1
      try:
        # Try the request.
        raw_response = request_func(*args, headers=headers)
        return raw_response
      except urllib2.HTTPError as e:
        response = e.read()  # get error response
        try:
          response = json.loads(response)
          if response['status_code'] == 'TOKEN_EXPIRED':
            print 'Getting new access token.'
            self.get_access_token(renew=True)
            headers = header_func()
          else:
            raise ApiError(response)  # raise original error
        except ValueError as e2:
          raise ApiError(response) # raise original error.
        except Exception as e2:
          raise ApiError(response) # raise original error.


  def _get_json_response(self, url, data, headers):
    """ Get the response for sending json dumped data. """
    if data:
      data = json.dumps(data)
    req = urllib2.Request(url, data, headers)
    response = urllib2.urlopen(req)
    raw_response = response.read()
    return raw_response

  def old_tag_image(self, image_files):
    """ DEPRECATED: use tag_images which is more efficient and support single or multiple images.

    Autotag an image.

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
    data = {'encoded_image': base64.encodestring(image_files[0].read())}
    return self._single_image_op(data, 'tag')


  def _single_image_op(self, data, op):
    """ DEPRECATED: use _multi_image_op which is more efficient.
    """
    if op not in SUPPORTED_OPS:
      raise Exception('Unsupported op: %s, ops available: %s' % (op, str(SUPPORTED_OPS)))
    data['op'] =  op
    access_token = self.get_access_token()
    url = self._url_for_op(data['op'])
    headers = self._get_json_headers(access_token)
    response = self._get_json_response(url, data, headers)
    return dict([(k, v[0]) for k, v in self._parse_response(response, op).items()])
