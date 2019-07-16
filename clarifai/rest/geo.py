from typing import Dict
"""
Geo support for clarifai api
"""


class GeoPoint(object):
  """ define a Geo point
      which is a (longitude, latitude) tuple
  """

  def __init__(self, longitude, latitude):  # type: (float, float) -> None
    self.longitude = float(longitude)
    self.latitude = float(latitude)

  def dict(self):  # type: () -> dict
    return {'geo_point': {'longitude': self.longitude, 'latitude': self.latitude}}


class GeoBox(object):
  """ define a Geo box
      which is defined with a pair of GeoPoint
      representing the two corners of the Geo box
  """

  def __init__(self, point1, point2):  # type: (GeoPoint, GeoPoint) -> None
    self.point1 = point1
    self.point2 = point2

  def dict(self):  # type: () -> dict
    return {'geo_box': [self.point1.dict(), self.point2.dict()]}


class GeoLimit(object):
  convert_table = {
      'mile': 'withinMiles',
      'kilometer': 'withinKilometers',
      'degree': 'withinDegrees',
      'radian': 'withinRadians'
  }  # type: Dict[str, str]

  def __init__(self, limit_type='mile', limit_range=10):  # type: (str, int) -> None
    if limit_type not in self.convert_table:
      raise ValueError("limit_type could be within %s" % str(self.convert_table.keys()))

    self.limit_type = self.convert_table[limit_type]
    self.limit_range = float(limit_range)

  def dict(self):  # type: () -> dict
    return {'geo_limit': {'type': self.limit_type, 'value': self.limit_range}}


class Geo(object):

  def __init__(self, geo_point=None, geo_limit=None, geo_box=None):
    # type: (GeoPoint, GeoLimit, GeoBox) -> None

    self.geo_point = geo_point
    self.geo_limit = geo_limit
    self.geo_box = geo_box

    # only geo_point for input
    if geo_point is not None and geo_limit is None and geo_box is None:
      pass
    # geo_point and geo_limit for search within range
    elif geo_point is not None and geo_limit is not None and geo_box is None:
      pass
    # only geo_box for search
    elif geo_point is None and geo_limit is None and geo_box is not None:
      pass
    else:
      raise Exception('Invalid Geo object initialization')

  def dict(self):  # type: () -> dict

    # only geo_point for input
    if self.geo_point is not None and self.geo_limit is None and self.geo_box is None:
      data = {'geo': self.geo_point.dict()}
    # geo_point and geo_limit for search within range
    elif self.geo_point is not None and self.geo_limit is not None and self.geo_box is None:
      data = {'geo': self.geo_point.dict()}
      data['geo'].update(self.geo_limit.dict())
    # only geo_box for search
    elif self.geo_point is None and self.geo_limit is None and self.geo_box is not None:
      data = {'geo': self.geo_box.dict()}
    else:
      raise Exception('Invalid Geo object')

    return data
