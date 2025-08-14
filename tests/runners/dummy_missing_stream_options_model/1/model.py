from openai import OpenAI

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_types import Text


class MyOpenAIModel(ModelClass):
    """A model that uses OpenAI chat completions without proper stream_options."""

    @ModelClass.method
    def generate(self, text1: Text = "") -> Text:
        """This method does the streaming
        using OpenAI chat completions without proper stream_options."""

        client = OpenAI()

        # This should trigger the validation error
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": text1.text}], stream=True
        )

        output_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                output_text += chunk.choices[0].delta.content

        return Text(output_text)
