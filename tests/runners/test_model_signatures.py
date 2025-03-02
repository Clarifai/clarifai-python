import unittest

from clarifai.runners.models.model_class import ModelClass, methods


class TestModelSignatures(unittest.TestCase):

  def test_basic(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int) -> int:
        return 2 * x

    #pprint( MyModel._get_method_info('f').signature)
    assert MyModel._get_method_info('f').signature == \
              {'inputs': [{'data_field': 'int_value',
                           'data_type': 'int',
                           'name': 'x',
                           'required': True,
                           'streaming': False}],
               'method_type': 'predict',
               'name': 'f',
               'outputs': [{'data_field': 'int_value',
                            'data_type': 'int',
                            'name': 'return',
                            'streaming': False}]}


if __name__ == '__main__':
  unittest.main()
