from typing import Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_handler import Output


class MyModel(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text."""

  def load_model(self):
    """Load the model here."""

  def predict(self, text1: str = "", image_url: str = "") -> Output:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output_text = text1 + "Hello World"
    output_image_url = image_url.replace("samples.clarifai.com", "newdomain.com")

    return Output(text=output_text, image_url=output_image_url)

  def generate(self, text1: str = "", image_url: str = "") -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # fake something iterating generating 10 times.
      output_text = text1 + f"Generate Hello World {i}"
      yield Output(text=output_text)

  def stream(self, input_iterator) -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""

    for ri, input in enumerate(input_iterator):
      for i in range(10):  # fake something iterating generating 10 times.
        output_text = input.text + f"Stream Hello World {i}"
        yield Output(text=output_text)
