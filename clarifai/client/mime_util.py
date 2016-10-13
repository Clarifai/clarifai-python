# -*- coding: utf-8 -*-

import sys
import urllib
from email.encoders import encode_noop
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from uuid import uuid4

if sys.version_info >= (3,0):
  import urllib.request as urllib2
  from urllib.parse import urlparse
  from urllib.parse import quote

  def iteritems(d):
    return iter(d.items())
else:
  import urllib2
  from urlparse import urlparse
  from urllib import quote

  def iteritems(d):
    return d.iteritems()

class RequestWithMethod(urllib2.Request):
  """Extend urllib2.Request to support methods beyond GET and POST."""
  def __init__(self, url, method, data=None, headers={},
               origin_req_host=None, unverifiable=False):
    self.url = url
    self._method = method
    urllib2.Request.__init__(self, url, data, headers, origin_req_host, unverifiable)

  def get_method(self):
    if self._method:
        return self._method
    else:
        return urllib2.Request.get_method(self)

  def __str__(self):
    return '%s %s' % (self.get_method(), self.url)


def post_data_multipart(url, media=[], form_data={}, headers={}):
  """POST a multipart MIME request with encoded media.

  Args:
    url: where to send the request.
    media: list of (encoded_data, filename) pairs.
    form_data: dict of API params.
    headers: dict of extra HTTP headers to send with the request.
  """
  message = multipart_form_message(media, form_data)
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

def crlf_mixed_join(lines):
  """ This handles the mix of 'str' and 'unicode' in the data,
  encode 'unicode' lines into 'utf-8' so the lines will be joinable
  otherwise, the non-unicode lines will be auto converted into unicode
  and triggers exception because the MIME data is not unicode convertible

  Also, Python3 makes this even more complicated.
  """
  # set default encoding to 'utf-8'
  encoding = 'utf-8'

  post_data = bytearray()

  idx = 0
  for line in lines:
    if sys.version_info < (3,0):
      if isinstance(line, unicode):
        line = line.encode(encoding)
      # turn to bytearray
      line_bytes = bytearray(line)

    if sys.version_info >= (3,0):
      if isinstance(line, str):
        line_bytes = bytearray(line, encoding)
      else:
        line_bytes = bytearray(line)

    if idx > 0:
      post_data.extend(b'\r\n')

    post_data.extend(line_bytes)
    idx += 1
    
  return post_data

def form_data_media(encoded_data, filename, field_name='encoded_data', headers={}):
  """From raw encoded media return a MIME part for POSTing as form data."""
  message = MIMEApplication(encoded_data, 'application/octet-stream', encode_noop, **headers)

  disposition_headers = {
    'name': '%s' % field_name,
    'filename': quote(filename.encode('utf-8')),
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
    data = part.get_payload(decode=True)
    lines.append(data)
  lines.append('--%s--' % boundary)
  post_data = crlf_mixed_join(lines)
  headers['Content-Length'] = str(len(post_data))
  headers['Content-Type'] = 'multipart/form-data; boundary=%s' % boundary
  return post_data, headers

def multipart_form_message(media, form_data={}):
  """Return a MIMEMultipart message to upload encoded media via an HTTP form POST request.

  Args:
    media: a list of (encoded_data, filename) tuples.
    form_data: dict of name, value form fields.
  """
  message = MIMEMultipart('form-data', None)
  if form_data:
    for (name, val) in iteritems(form_data):
      part = Message()
      part.add_header('Content-Disposition', 'form-data', name=name)
      part.set_payload(val)
      message.attach(part)

  for im, filename in media:
    message.attach(form_data_media(im, filename))

  return message
