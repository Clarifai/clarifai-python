from email.encoders import encode_noop
from email.Generator import Generator
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import httplib
from cStringIO import StringIO
import urllib
from urlparse import urlparse


def post_images_multipart(images, form_data, url):
  """
  :param images: list of (encoded_image, filename) pairs.
  :param form_data: dict of API params.
  :param base_url: (host, port) tuple.
  """
  message = multipart_form_message(images, form_data)
  with open('/tmp/foo', 'w') as f:
    # FIXME
    data = message_as_post_data(message)
    f.write(data)
  response = post_multipart_request(url, message)
  return response


def parse_url(url):
  """Return a host, port, path tuple from a url."""
  parsed_url = urlparse(url)
  port = parsed_url.port or 80
  return parsed_url.hostname, port, parsed_url.path


def post_multipart_request(url, multipart_message):
  host, port, path = parse_url(url)
  h = httplib.HTTP(host, port)
  h.putrequest('POST', path)
  data = message_as_post_data(multipart_message)
  h.putheader('Content-Length', str(len(data)))
  h.putheader('Content-Type', multipart_message.get('Content-Type'))
  h.endheaders()
  h.send(data)
  errcode, errmsg, headers = h.getreply()
  print errcode, errmsg, headers
  return h.file.read()


def mime_image(encoded_image, subtype='jpeg', headers={}):
  """From a raw encoded image return a MIME image part."""
  return MIMEImage(encoded_image, subtype, encode_noop, **headers)


def form_data_image(encoded_image, filename, field_name='encoded_image', subtype='jpeg', headers={}):
  """From raw encoded image return a MIME part for POSTing as form data."""
  message = mime_image(encoded_image, subtype, headers)
  disposition_headers = {
    'name': '%s' % field_name,
    'filename': urllib.quote(filename),
  }
  message.add_header('Content-Disposition', 'form-data', **disposition_headers)
  # Django seems fussy and doesn't like the MIME-Version header in multipart POSTs.
  del message['MIME-Version']
  return message


def message_as_post_data(message):
  """Return a string suitable for using as POST data, from a multipart MIME message."""
  # The built-in mail generator outputs broken POST data for several reasons:
  # * It breaks long header lines, and django doesn't like this. Can use Generator.
  # * It uses newlines, not CRLF.  There seems to be no easy fix in 2.7:
  #   http://stackoverflow.com/questions/3086860/how-do-i-generate-a-multipart-mime-message-with-correct-crlf-in-python
  # * It produces the outermost multipart MIME headers, which would need to get stripped off
  #   as form data because the HTTP headers are used instead.
  # So just generate what we need directly.
  assert message.is_multipart()
  unused = message.as_string()  # Needed to generate the boundary
  boundary = message.get_boundary()
  lines = []
  for part in message.get_payload():
    lines.append('--' + boundary)
    for k, v in part.items():
      lines.append('%s: %s' % (k, v))
    lines.append('')
    lines.append(part.get_payload())
  lines.append('--%s--' % boundary)
  crlf = '\r\n'
  return crlf.join(lines)


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
