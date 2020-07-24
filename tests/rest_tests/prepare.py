import logging
import uuid

from retrying import retry

from clarifai.rest import ClarifaiApp

urls = [
    "https://samples.clarifai.com/metro-north.jpg",
    "https://samples.clarifai.com/wedding.jpg",
    "https://samples.clarifai.com/facebook.png",
    "https://samples.clarifai.com/dog.tiff",
    "https://samples.clarifai.com/penguin.bmp",
]


@retry(stop_max_attempt_number=6, wait_exponential_multiplier=5000, wait_exponential_max=30000)
def delete_all_inputs(app, logger):
  logger.error('Attempting to delete all inputs.')
  app.inputs.delete_all()
  logger.error('All inputs have been deleted.')


def main():
  logger = logging.getLogger('clarifai')
  logger.handlers = []
  logger.addHandler(logging.StreamHandler())
  logger.setLevel(logging.INFO)

  app = ClarifaiApp()

  delete_all_inputs(app, logger)
  app.models.delete_all()

  app.wait_until_inputs_delete_finish()
  app.wait_until_models_delete_finish()

  image1 = app.inputs.create_image_from_url(
      urls[0],
      concepts=['train_custom_prepare', 'railway_custom_prepare'],
      allow_duplicate_url=True)
  image2 = app.inputs.create_image_from_url(
      urls[3], concepts=['dog_custom_prepare', 'animal_custom_prepare'], allow_duplicate_url=True)

  app.wait_for_specific_input_uploads_to_finish(ids=[image1.input_id, image2.input_id])

  model_id = uuid.uuid4().hex

  model1 = app.models.create(
      model_id=model_id,
      concepts=[
          'train_custom_prepare', 'railway_custom_prepare', 'dog_custom_prepare',
          'animal_custom_prepare'
      ])
  model1.train(timeout=120, raise_on_timeout=True)


if __name__ == '__main__':
  main()
