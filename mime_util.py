from email.encoders import encode_noop
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import urllib
import urllib2
from urlparse import urlparse
from uuid import uuid4

class RequestWithMethod(urllib2.Request):
  """Workaround for using DELETE with urllib2"""
  def __init__(self, url, method, data=None, headers={},
               origin_req_host=None, unverifiable=False):
    self.url = url
    self._method = method
    urllib2.Request.__init__(self, url, data, headers,
                             origin_req_host, unverifiable)
  def get_method(self):
    if self._method:
        return self._method
    else:
        return urllib2.Request.get_method(self)

  def __str__(self):
    return 'url: %s, method %s' % (self.url, self._method)

def post_images_multipart(images, form_data, url, headers={}):
  """POST a multipart MIME request with image data.

  Args:
    images: list of (encoded_image, filename) pairs.
    form_data: dict of API params.
    base_url: (host, port) tuple.
    headers: A dict of extra HTTP headers to send with the request.
  """
  message = multipart_form_message(images, form_data)
  response = post_multipart_request(url, message, headers=headers)
  return response


def parse_url(url):
  """Return a host, port, path tuple from a url."""
  parsed_url = urlparse(url)
  port = parsed_url.port or 80
  if url.startswith('https'):
    port = 443
  return parsed_url.hostname, port, parsed_url.path

def post_multipart_request(url, multipart_message, headers={}):
  data, headers = message_as_post_data(multipart_message, headers)
  req = RequestWithMethod(url, 'POST', data, headers)
  f = urllib2.urlopen(req)
  response = f.read()
  f.close()
  return response

def mime_image(encoded_image, subtype='jpeg', headers={}):
  """From a raw encoded image return a MIME image part."""
  return MIMEImage(encoded_image, subtype, encode_noop, **headers)


# FIXME: Pass real subtype, don't assume jpeg.

def form_data_image(encoded_image, filename, field_name='encoded_image',
                    subtype='jpeg', headers={}):
  """From raw encoded image return a MIME part for POSTing as form data."""
  message = mime_image(encoded_image, subtype, headers)
  disposition_headers = {
    'name': '%s' % field_name,
    'filename': urllib.quote(filename.encode('utf-8')),
  }
  message.add_header('Content-Disposition', 'form-data', **disposition_headers)
  # Django seems fussy and doesn't like the MIME-Version header in multipart POSTs.
  del message['MIME-Version']
  return message


def message_as_post_data(message, headers):
  """Return a string suitable for using as POST data, from a multipart MIME message."""
  # The built-in mail generator outputs broken POST data for several reasons:
  # * It breaks long header lines, and django doesn't like this. Can use Generator.
  # * It uses newlines, not CRLF.  There seems to be no easy fix in 2.7:
  #   http://stackoverflow.com/questions/3086860/how-do-i-generate-a-multipart-mime-message-with-correct-crlf-in-python
  # * It produces the outermost multipart MIME headers, which would need to get stripped off
  #   as form data because the HTTP headers are used instead.
  # So just generate what we need directly.
  assert message.is_multipart()
  # Simple way to get a boundary. urllib3 uses this approach.
  boundary = uuid4().hex
  lines = []
  for part in message.get_payload():
    lines.append('--' + boundary)
    for k, v in part.items():
      lines.append('%s: %s' % (k, v))
    lines.append('')
    lines.append(part.get_payload())
  lines.append('--%s--' % boundary)
  crlf = '\r\n'
  post_data = crlf.join(lines)
  headers['Content-Length'] = str(len(post_data))
  headers['Content-Type'] = 'multipart/form-data; boundary=%s' % boundary
  return post_data, headers


def multipart_form_message(images, form_data={}):
  """Return a MIMEMultipart message to upload images via an HTTP form POST request.

  Args:
    images: a list of (encoded_image, filename) tuples.
    form_data: dict of name, value form fields.
  """
  message = MIMEMultipart('form-data', None)
  for name, val in form_data.iteritems():
    part = Message()
    part.add_header('Content-Disposition', 'form-data', name=name)
    part.set_payload(val)
    message.attach(part)

  for im, filename in images:
    message.attach(form_data_image(im, filename))

  return message
