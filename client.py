import base64, json, logging, os, time, urllib2, urllib
try:
  from PIL import Image
  CAN_RESIZE = True
except Exception, e:
  CAN_RESIZE = False
  print ('It is recommended to install PIL/Pillow with the desired image format support so that '
         'image resizing to the correct dimesions will be handled for you. '
         'If using pip, try "pip install Pillow"')
from cStringIO import StringIO
from mime_util import post_images_multipart

logger = logging.getLogger(__name__)


class ApiError(Exception):
  """Api error."""
  def __init__(self, msg):
    self.msg = msg
  def __str__(self):
    return repr(self.msg)


class ApiThrottledError(Exception):
  """This is raised when the usage throttle is hit. Client should for wait_seconds before retrying."""
  def __init__(self, msg, wait_seconds):
    self.msg = msg
    self.wait_seconds = wait_seconds
  def __str__(self):
    return repr(self.msg) + '  Wait for %d seconds before retrying.' % self.wait_seconds


SUPPORTED_OPS = ['tag','embed','feedback']

IM_QUALITY = 95
API_VERSION = 'v1'


class ClarifaiApi(object):
  """
  The constructor for API access. You must sign up at developer.clarifai.com first and create an
  application in order to generate your credentials for API access.

  Args:
    app_id: the client_id for an application you've created in your Clarifai account.
    app_secret: the client_secret for the same application.
    base_url: Base URL of the API endpoints.
    model: Name of the recognition model to query. Use the default if None.
    wait_on_throttle: When the API returns a 429 throttled error, sleep for the amount of time
        reported in the X-Throttle-Wait-Seconds HTTP response header.
  """
  def __init__(self, app_id=None, app_secret=None, base_url='https://api.clarifai.com',
               model='default', wait_on_throttle=True):
    if app_id is None:
      self.CLIENT_ID = os.environ.get('CLARIFAI_APP_ID', None)
    else:
      self.CLIENT_ID = app_id
    if app_secret is None:
      self.CLIENT_SECRET = os.environ.get('CLARIFAI_APP_SECRET', None)
    else:
      self.CLIENT_SECRET = app_secret
    self.wait_on_throttle = wait_on_throttle

    self._base_url = base_url
    self.set_model(model)
    self._urls = {
      'tag': os.path.join(self._base_url, '%s/tag/' % API_VERSION),
      'embed': os.path.join(self._base_url, '%s/embed/' % API_VERSION),
      'multiop': os.path.join(self._base_url, '%s/multiop/' % API_VERSION),
      'feedback': os.path.join(self._base_url, '%s/feedback/' % API_VERSION),
      'token': os.path.join(self._base_url, '%s/token/' % API_VERSION),
      'info': os.path.join(self._base_url, '%s/info/' % API_VERSION),
      }
    self.access_token = None
    self.api_info = None

  def set_model(self, model):
    self._model = self._sanitize_param(model)

  def get_access_token(self, renew=False):
    """ Get an access token using your app_id and app_secret.

    You shouldn't need to call this method yourself. If there is no access token yet, this method
    will be called when a request is made. If a token expires, this method will also automatically
    be called to renew the token.

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
    """ Get various information about the current state of the API.

    This provides general information such as the API version number, but also use specific
    information such as the limitations on your account. Some of this information is needed to
    ensure that your API calls will go through within your limits.
    """
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

  def tag_images(self, image_files, model=None, local_ids=None, meta=None):
    """ Autotag a single image from an open file object or multiples images from a list of open file
    objects.

    The only method used on the file object is read() to get the bytes of the compressed
    image representation. Ensure that all file objects are pointing to the  beginning of a valid
    image.

    Args:
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
    open file-like object containing the encoded image bytes.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

    Returns:
      results: an API reponse including the generated tags. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                               open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, ['tag'], model=model, local_ids=local_ids,
                                meta=meta)

  def embed_images(self, image_files, model=None, local_ids=None, meta=None):
    """ Embed a single image from an open file object or multiples images from a list of open file
    objects.

    The only method used on the file object is read() to get the bytes of the compressed
    image representation.

    Args:
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
    open file-like object containing the encoded image bytes.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

    Returns:
      results: an API reponse including the generated embeddings. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_images([open('/path/to/local/image.jpeg'),
                               open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, ['embed'], model=model, local_ids=local_ids,
                                meta=meta)

  def tag_and_embed_images(self, image_files, model=None, local_ids=None, meta=None):
    """ Tag AND embed images in one request. Note: each operation is treated separate for billing
    purposes.

    Args:
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
    open file-like object containing the encoded image bytes.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

     Returns:
      results: an API reponse including the generated tags and embeddings. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_and_embed_images([open('/path/to/local/image.jpeg'),
                                         open('/path/to/local/image2.jpeg')])
    """
    return self._multi_image_op(image_files, ['tag','embed'], model=model, local_ids=local_ids,
                                meta=meta)

  def tag_image_urls(self, image_urls, model=None, local_ids=None, meta=None):
    """ Tag an image from a url or images from a list of urls.

    Args:
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed. Note: all urls must be publically accessible.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

    Returns:
      results: an API reponse including the generated tags. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_image_urls(['http://www.clarifai.com/img/metro-north.jpg',
                                  'http://www.clarifai.com/img/metro-north.jpg'])

    """
    return self._multi_imageurl_op(image_urls, ['tag'], model=model, local_ids=local_ids,
                                   meta=meta)

  def embed_image_urls(self, image_urls, model=None, local_ids=None, meta=None):
    """ Embed an image from a url or images from a list of urls.

    Args:
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed. Note: all urls must be publically accessible.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

    Returns:
      results: an API reponse including the generated embeddings. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.embed_image_url(['http://www.clarifai.com/img/metro-north.jpg',
                                  'http://www.clarifai.com/img/metro-north.jpg'])

    """
    return self._multi_imageurl_op(image_urls, ['embed'], model=model, local_ids=local_ids,
                                   meta=meta)

  def tag_and_embed_image_urls(self, image_urls, model=None, local_ids=None, meta=None):
    """ Tag AND Embed an image from a url or images from a list of urls.

    Args:
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed. Note: all urls must be publically accessible.
      model: specifies the desired model to use for processing of the images.
      local_ids: a single string identifier or list of string identifies that are useful client
    side. These will be returned in the request to match up results (even though results to come
    back in order).
      meta: a string of any extra information to accompany the request. This has to be a string, so
    if passing structured data, pass a json.dumps(meta) string.

    Returns:
      results: an API reponse including the generated tags and embeddings. See the docs at
      https://developer.clarifai.com/docs/ for more detais.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_and_embed_image_url(['http://www.clarifai.com/img/metro-north.jpg',
                                            'http://www.clarifai.com/img/metro-north.jpg'])
    """
    return self._multi_imageurl_op(image_urls, ['tag','embed'], model=model, local_ids=local_ids,
                                   meta=meta)

  def feedback(self, docids=None, image_urls=None, images=None, add_tags=None,
               remove_tags=None, similar_docids=None, dissimilar_docids=None,
               search_click=None):
    """ Tag AND Embed an image from a url or images from a list of urls.

    Args:
      docids: list of docid strings for images already processed by the API.
      image_files: a single (file, name) tuple or a list of (file, name) tuples, where file is an
    open file-like object containing the encoded image bytes.
      image_urls: a single url for the input image to be processed or a list of urls for a set of
    images to be processed. Note: all urls must be publically accessible.
      add_tags: If the user believes additioal tags are relavent to the given image, they can be
    provided in the add_tags argument.
      remove_tags: If the user believes tags were are not relavent to the given image, they can be
    provided in the remove_tags argument.
      similar_docids: If there is a notion of similarity between images, this can be fed back to the
    system by providing an input set of docids and a list of docids that are similar to the input
    docids.
      dissimilar_docids: If there is a notion of similarity between images, this can be fed back to
    the system by providing an input set of docids and a list of docids that are dissimilar to the
    input docids.
      search_click: This is useful when showing search results and a user clicks on images when the
    "search_click" tags were used to generate the search results.

    Returns:
      results: OK if everything went well.

    Example:
      from py.client import ClarifaiApi
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_and_embed_image_url(['http://www.clarifai.com/img/metro-north.jpg',
                                            'http://www.clarifai.com/img/metro-north.jpg'])
    """
    if int(docids is not None) + int(image_urls is not None) + int(images is not None) != 1:
      raise ApiError("Must specify exactly one of docids, image_urls or images")
    if (int(add_tags is not None) + int(remove_tags is not None) +
        int(similar_docids is not None) + int(dissimilar_docids is not None) +
        int(search_click is not None)) == 0:
      raise ApiError(("Must specify one or more of add_tags, remove_tags, similar_docids, "
                      "dissimilar_docids, search_click."))
    payload = {}
    def add_comma_arg(payload, name, value):
      if not isinstance(value, list):
        value = [value]
      payload[name] = ','.join(value)
    if add_tags:
      add_comma_arg(payload, 'add_tags', add_tags)
    if remove_tags:
      add_comma_arg(payload, 'remove_tags', remove_tags)
    if similar_docids:
      add_comma_arg(payload, 'similar_docids', similar_docids)
    if dissimilar_docids:
      add_comma_arg(payload, 'dissimilar_docids', dissimilar_docids)
    if search_click:
      add_comma_arg(payload, 'search_click', search_click)
    if docids is not None:
      add_comma_arg(payload, 'docids', docids)
      return self._multi_imageurl_op(None, ['feedback'], payload=payload)
    elif image_urls is not None:
      return self._multi_imageurl_op(image_urls, ['feedback'], payload=payload)
    else: # must be images
      raise ApiError("Using encoded_images in feed is not supported in Python client yet.")
      # return self._multi_image_op(image_files, ['feedback'], payload=payload)

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
      min_dimension = min(img.size)
      max_dimension = max(img.size)
      min_ratio = float(MIN_SIZE) / min_dimension
      max_ratio = float(MAX_SIZE) / max_dimension
      def get_newsize(img, ratio, SIZE):
        if img.size[0] == min_dimension:
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
      logger.warning('Could not open image file: %s, still sending to server.', image_tup[1])
    return image_tup

  def _process_image_files(self, input_files):
    """ Ensure consistent format for image files from local storage.
    """
    # Handle single file-object as arg.
    if not isinstance(input_files, list):
      input_files = [input_files]
    self._check_batch_size(input_files)
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

  def _check_batch_size(self, data_list):
    """ Ensure the maximum batch size is obeyed on the client side. """
    if self.api_info is None:
      self.get_info()  # sets the image size and other such info from server.
    MAX_BATCH_SIZE = self.api_info['max_batch_size']
    if len(data_list) > MAX_BATCH_SIZE:
      raise ApiError(("Number of images provided in bach %d is greater than maximum allowed per "
                      "request %d") % (len(data_list), MAX_BATCH_SIZE))

  def _multi_image_op(self, image_files, ops, model=None, local_ids=None, meta=None):
    """ Supports both list of tuples (image_file, name) or a list of image_files where a name will
    be created as the index into the list. """
    if len(set(ops).intersection(SUPPORTED_OPS)) != len(ops):
      raise Exception('Unsupported op: %s, ops available: %s' % (str(ops), str(SUPPORTED_OPS)))
    image_data = self._process_image_files(image_files)
    data = {'op': ','.join(ops)}
    if model:
      data['model'] = self._sanitize_param(model)
    elif self._model:
      data['model'] = self._model
    if local_ids:
      self._insert_local_ids(data, local_ids, len(image_data))
      data['local_id'] = ','.join(data['local_id'])
    # if meta:
    #   data['meta'] = self._sanitize_param(meta)
    url = self._url_for_op(ops)
    raw_response = self._get_raw_response(self._get_multipart_headers,
                                          post_images_multipart, image_data, data, url)
    return self._parse_response(raw_response, ops)

  def _sanitize_param(self, param):
    """Convert parameters into a form ready for the wire."""
    if param:
      # Can't send unicode.
      param = str(param)
    return param

  def _insert_local_ids(self, data, local_ids, batch_size):
    if not isinstance(local_ids, list):
      local_ids = [local_ids]
    assert isinstance(local_ids, list)
    assert isinstance(local_ids[0], basestring)
    assert len(local_ids) == batch_size
    data['local_id'] = local_ids

  def _multi_imageurl_op(self, image_urls, ops, model=None, local_ids=None, meta=None,
                         payload=None):
    """ If sending image_url or image_file strings, then we can send as json directly instead of the
    multipart form. """
    if len(set(ops).intersection(SUPPORTED_OPS)) != len(ops):
      raise Exception('Unsupported op: %s, ops available: %s' % (str(ops), str(SUPPORTED_OPS)))
    data =  {'op': ','.join(ops)}
    if image_urls is not None: # for feedback, this might not be required.
      if not isinstance(image_urls, list):
        image_urls = [image_urls]
      self._check_batch_size(image_urls)
      if not isinstance(image_urls[0], basestring):
        raise Exception("image_urls must be strings")
      data['url'] = image_urls
    if model:
      data['model'] = self._sanitize_param(model)
    elif self._model:
      data['model'] = self._model
    if local_ids:
      self._insert_local_ids(data, local_ids, len(image_urls))
      data['local_id'] = ','.join(data['local_id'])
    if payload:
      assert isinstance(payload, dict), "Addition payload must be a dict"
      for k, v in payload.iteritems():
        data[k] = v
    url = self._url_for_op(ops)
    raw_response = self._get_raw_response(self._get_json_headers,
                                          self._get_json_response, url, data)
    return self._parse_response(raw_response, ops)

  def _parse_response(self, response, all_ops):
    """ Get the raw response form the API and convert into nice Python objects. """
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
    headers['Content-Type'] = 'application/json'
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
        if e.code == 429:
          # Throttled.  Wait for the specified number of seconds.
          wait_secs = e.info().get('X-Throttle-Wait-Seconds', 10)
          try:
            wait_secs = int(wait_secs)
          except ValueError as e:
            wait_secs = 10
          if self.wait_on_throttle:
            logger.error('Throttled. Waiting %d seconds.', wait_secs)
            time.sleep(wait_secs)
          raise ApiThrottledError(response, wait_secs)
        try:
          response = json.loads(response)
          if response['status_code'] == 'TOKEN_EXPIRED':
            logger.info('Getting new access token.')
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

  def tag_image_base64(self, image_file):
    """ NOTE: If possible, you should use avoid this method and use tag_images, which is more
    efficient and supports single or multiple images.  This version base64-encodes the images.

    Autotag an image.

    Args:
      image_file: an open file-like object containing the encoded image bytes. The read
      method is called on this object to get the encoded bytes so it can be a file handle or
      StringIO buffer.

    Returns:
      results: A list of (tag, probability) tuples.

    Example:
      clarifai_api = ClarifaiApi()
      clarifai_api.tag_image(open('/path/to/local/image.jpeg'))
    """
    data = {'encoded_image': base64.encodestring(image_file.read())}
    return self._base64_encoded_image_op(data, 'tag')

  def _base64_encoded_image_op(self, data, op):
    """NOTE: _multi_image_op is more efficient, it avoids the overhead of base64 encoding."""
    if op not in SUPPORTED_OPS:
      raise Exception('Unsupported op: %s, ops available: %s' % (op, str(SUPPORTED_OPS)))
    data['op'] =  op
    access_token = self.get_access_token()
    url = self._url_for_op(data['op'])
    headers = self._get_json_headers()
    response = self._get_json_response(url, data, headers)
    return self._parse_response(response, op)
