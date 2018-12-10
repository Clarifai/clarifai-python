import os


def _input_url(filename):
  return 'https://samples.clarifai.com/' + filename


def _input_file_path(filename):
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', filename)


METRO_IMAGE_URL = _input_url('metro-north.jpg')
WEDDING_IMAGE_URL = _input_url('wedding.jpg')
FACEBOOK_IMAGE_URL = _input_url('facebook.png')
DOG_TIFF_IMAGE_URL = _input_url('dog.tiff')
DOG2_IMAGE_URL = _input_url('dog2.jpeg')
DOG2_NONEXISTENT_IMAGE_URL = _input_url('dog2_bad.jpeg')
PENGUIN_BMP_IMAGE_URL = _input_url('penguin.bmp')
TODDLER_FLOWERS_IMAGE_URL = _input_url('toddler-flowers.jpeg')

BEER_VIDEO_URL = _input_url('beer.mp4')
CONAN_GIF_VIDEO_URL = _input_url('3o6gb3kkXfLvdKEZs4.gif')
SMALL_GIF_VIDEO_URL = _input_url('D7qTae7IQLKSI.gif')

METRO_IMAGE_FILE_PATH = _input_file_path('metro-north.jpg')
TAHOE_IMAGE_FILE_PATH = _input_file_path('tahoe.jpg')
TODDLER_FLOWERS_IMAGE_FILE_PATH = _input_file_path('toddler-flowers.jpeg')
THAI_MARKET_IMAGE_FILE_PATH = _input_file_path('thai-market.jpg')

SMALL_VIDEO_FILE_PATH = _input_file_path('small.mp4')
BEER_VIDEO_FILE_PATH = _input_file_path('beer.mp4')
