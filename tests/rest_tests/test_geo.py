import unittest
from itertools import permutations
from clarifai.rest import Geo, GeoPoint, GeoBox, GeoLimit


class TestGeoSupport(unittest.TestCase):
  """
  unit test for the Geo support, on the class level
  not on the client api level
  there will be no API call out in this unittest
  """

  _multiprocess_can_split_ = True

  def test_geo_points(self):
    """ test get one concept by id """
    geo_p = GeoPoint(0, 0)
    geo_p = GeoPoint(10, 20)
    geo_p = GeoPoint(-30.3, 10.2234)
    geo_p = GeoPoint("-30.3", "10.2234")
    geo_p = GeoPoint("30.3", "10")
    geo_p = GeoPoint("0", "0")

    with self.assertRaises(TypeError) as e:
      gl = GeoPoint()
      self.assertIn(str(e), 'arguments')

    with self.assertRaises(ValueError) as e:
      gl = GeoPoint("b", "c")

    with self.assertRaises(ValueError) as e:
      gl = GeoPoint("", "0.33")

  def test_geo_box(self):
    """ test get one concept by id """
    geo_p1 = GeoPoint(0, 0)
    geo_p2 = GeoPoint(10, 20)
    geo_p3 = GeoPoint(-30.3, 10.2234)

    for p1, p2 in permutations([geo_p1, geo_p2, geo_p3], 2):
      geo_b = GeoBox(p1, p2)

    with self.assertRaises(TypeError) as e:
      gl = GeoBox()
      self.assertIn(str(e), 'arguments')

  def test_geo_limit(self):
    """ test geo limit """
    gl = GeoLimit("mile", 22)
    gl = GeoLimit("kilometer", 22)
    gl = GeoLimit("degree", 22)
    gl = GeoLimit("radian", 22)
    gl = GeoLimit(limit_range=22)
    gl = GeoLimit()

    with self.assertRaises(ValueError) as e:
      gl = GeoLimit("mile", "not_a_number")

    with self.assertRaises(TypeError) as e:
      gl = GeoLimit("kilometer", None)

    with self.assertRaises(ValueError) as e:
      gl = GeoLimit("out_of_range_type", 30)

  def test_geo(self):
    # make a geo point for input
    geo = Geo(geo_point=GeoPoint(10, 20))

    # make a geo point and geo limit for search
    geo = Geo(geo_point=GeoPoint(10, 20), geo_limit=GeoLimit("mile", 33))

    # make a geo box for search
    p1 = GeoPoint(0, 0)
    p2 = GeoPoint(30, -20.22)
    geo = Geo(geo_box=GeoBox(p1, p2))

    # make a geo point only
    geo = Geo(geo_point=p1)

    # make an invalid Geo
    with self.assertRaises(Exception):
      geo = Geo(geo_point=p1, geo_box=GeoBox(p1, p2), geo_limit=GeoLimit())

    # make an invalid Geo
    with self.assertRaises(Exception):
      geo = Geo()


if __name__ == '__main__':
  unittest.main()
