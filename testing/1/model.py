from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_utils import Param
from typing import Iterator
import random
import string

class MyModel(ModelClass):
  """This is a model that does some string manipulation."""

  def load_model(self):
    """Nothing to load for this model."""

  @ModelClass.method
  def predict(self, prompt: str, number_of_letters: int = Param(default=3, description="number of letters to add")) -> str:
    """Function to append some string information"""
    return new_str(prompt, number_of_letters)

  @ModelClass.method
  def generate(self, prompt: str = "", number_of_letters: int = Param(default=3, description="number of letters to add")) -> Iterator[str]:
    """Example yielding a whole batch of streamed stuff back."""
    for i in range(10):  # fake something iterating generating 10 times.
      yield new_str(str(i) + "-" + prompt, number_of_letters)

  @ModelClass.method
  def s(self, input_iterator: Iterator[str], number_of_letters: int = Param(default=3, description="number of letters to add")) -> Iterator[str]:
    """Example yielding getting an iterator and yielding back results."""
    for i, inp in enumerate(input_iterator):
      yield new_str(inp, number_of_letters)


def new_str(input_str: str, number_of_letters: int = 3) -> str:
    """Append a dash and random letters to the input string."""
    random_letters = ''.join(random.choices(string.ascii_letters, k=number_of_letters))
    return f"{input_str}-{random_letters}\n"


def test_predict() -> None:
    """Test the predict method of MyModel by printing its output."""
    model = MyModel()
    model.load_model()
    print("Testing predict method:")
    output = model.predict("TestPredict", number_of_letters=5)
    print(output, end="\n")

def test_generate() -> None:
    """Test the generate method of MyModel by printing its outputs."""
    model = MyModel()
    model.load_model()
    print("Testing generate method:")
    for output in model.generate("Test", number_of_letters=5):
        print(output, end="\n")

if __name__ == "__main__":
    test_predict()
    test_generate()
