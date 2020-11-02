import unittest

from clarifai.rest import Geo, GeoBox, GeoLimit, GeoPoint


class TestGeo(unittest.TestCase):
  """
  unit test for the Geo support, on the class level
  not on the client api level
  there will be no API call out in this unittest
  """

  def test_geo_limit(self):
    """ test geo limit """
    GeoLimit("mile", 22)
    GeoLimit("kilometer", 22)
    GeoLimit("degree", 22)
    GeoLimit("radian", 22)
    GeoLimit(limit_range=22)
    GeoLimit()

    with self.assertRaises(ValueError) as e:
      GeoLimit("mile", "not_a_number")

    with self.assertRaises(TypeError) as e:
      GeoLimit("kilometer", None)

    with self.assertRaises(ValueError) as e:
      GeoLimit("out_of_range_type", 30)

  def test_geo(self):
    # make a geo point for input
    Geo(geo_point=GeoPoint(10, 20))

    # make a geo point and geo limit for search
    Geo(geo_point=GeoPoint(10, 20), geo_limit=GeoLimit("mile", 33))

    # make a geo box for search
    p1 = GeoPoint(0, 0)
    p2 = GeoPoint(30, -20.22)
    Geo(geo_box=GeoBox(p1, p2))

    # make a geo point only
    Geo(geo_point=p1)

    # make an invalid Geo
    with self.assertRaises(Exception):
      Geo(geo_point=p1, geo_box=GeoBox(p1, p2), geo_limit=GeoLimit())

    # make an invalid Geo
    with self.assertRaises(Exception):
      Geo()
