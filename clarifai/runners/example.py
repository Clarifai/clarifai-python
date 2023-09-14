from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.runner import Runner


class MyRunner(Runner):
  """A custom runner that adds "Hello World" to the end of the text and replaces the domain of the
  image URL as an example.
  """

  def run_input(self, input: resources_pb2.Input) -> resources_pb2.Output:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output = resources_pb2.Output()

    data = input.data

    if data.text.raw != "":
      output.data.text.raw = data.text.raw + "Hello World"
    if data.image.url != "":
      output.data.text.raw = data.image.url.replace("samples.clarifai.com", "newdomain.com")
    return output


if __name__ == '__main__':
  # Make sure you set these env vars before running the example.
  # CLARIFAI_PAT
  # CLARIFAI_USER_ID

  # You need to first create a runner in the Clarifai API and then use the ID here.
  MyRunner(runner_id="sdk-test-runner").start()
