from clarifai.runners.models.model_class import ModelClass, methods
from clarifai.runners.utils.data_handler import Stream, Text


class MyModel(ModelClass):
  """A custom runner that adds "Hello World" to the end of the text."""

  def load_model(self):
    """Load the model here."""

  @methods.predict
  def predict(self, text1: Text = "") -> Text:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output_text = text1.text + "Hello World"

    return Text(output_text)

  @methods.generate
  def generate(self, text1: Text = Text("")) -> Stream[Text]:
    """Example yielding a whole batch of streamed stuff back."""

    for i in range(10):  # fake something iterating generating 10 times.
      output_text = text1.text + f"Generate Hello World {i}"
      yield Text(output_text)

  @methods.stream
  def stream(self, input_iterator: Stream[Text]) -> Stream[Text]:
    """Example yielding a whole batch of streamed stuff back."""

    for ri, input in enumerate(input_iterator):
      for i in range(10):  # fake something iterating generating 10 times.
        output_text = input.text + f"Stream Hello World {i}"
        yield Text(output_text)
