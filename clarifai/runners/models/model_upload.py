import os
import re
import time
from string import Template

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from rich import print

from clarifai.client import BaseClient
from clarifai.runners.utils.loader import HuggingFaceLoader
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import logger


def _clear_line(n: int = 1) -> None:
  LINE_UP = '\033[1A'  # Move cursor up one line
  LINE_CLEAR = '\x1b[2K'  # Clear the entire line
  for _ in range(n):
    print(LINE_UP, end=LINE_CLEAR, flush=True)


class ModelUploader:
  DEFAULT_PYTHON_VERSION = 3.11
  DEFAULT_TORCH_VERSION = '2.4.0'
  DEFAULT_CUDA_VERSION = '124'
  # List of available torch images for matrix
  '''
        python_version: ['3.8', '3.9', '3.10', '3.11']
        torch_version: ['2.0.0', '2.1.0', '2.2.0', '2.3.0', '2.4.0', '2.4.1', '2.5.0']
        cuda_version: ['124']
  '''
  AVAILABLE_TORCH_IMAGES = [
      '2.0.0-py3.8-cuda124',
      '2.0.0-py3.9-cuda124',
      '2.0.0-py3.10-cuda124',
      '2.0.0-py3.11-cuda124',
      '2.1.0-py3.8-cuda124',
      '2.1.0-py3.9-cuda124',
      '2.1.0-py3.10-cuda124',
      '2.1.0-py3.11-cuda124',
      '2.2.0-py3.8-cuda124',
      '2.2.0-py3.9-cuda124',
      '2.2.0-py3.10-cuda124',
      '2.2.0-py3.11-cuda124',
      '2.3.0-py3.8-cuda124',
      '2.3.0-py3.9-cuda124',
      '2.3.0-py3.10-cuda124',
      '2.3.0-py3.11-cuda124',
      '2.4.0-py3.8-cuda124',
      '2.4.0-py3.9-cuda124',
      '2.4.0-py3.10-cuda124',
      '2.4.0-py3.11-cuda124',
      '2.4.1-py3.8-cuda124',
      '2.4.1-py3.9-cuda124',
      '2.4.1-py3.10-cuda124',
      '2.4.1-py3.11-cuda124',
  ]
  AVAILABLE_PYTHON_IMAGES = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']
  PYTHON_BASE_IMAGE = 'public.ecr.aws/clarifai-models/python-base:{python_version}'
  TORCH_BASE_IMAGE = 'public.ecr.aws/clarifai-models/torch:{torch_version}-py{python_version}-cuda{cuda_version}'

  CONCEPTS_REQUIRED_MODEL_TYPE = [
      'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
  ]

  def __init__(self, folder: str):
    self._client = None
    self.folder = self._validate_folder(folder)
    self.config = self._load_config(os.path.join(self.folder, 'config.yaml'))
    self.model_proto = self._get_model_proto()
    self.model_id = self.model_proto.id
    self.model_version_id = None
    self.inference_compute_info = self._get_inference_compute_info()
    self.is_v3 = True  # Do model build for v3

  @staticmethod
  def _validate_folder(folder):
    if not folder.startswith("/"):
      folder = os.path.join(os.getcwd(), folder)
    logger.info(f"Validating folder: {folder}")
    if not os.path.exists(folder):
      raise FileNotFoundError(f"Folder {folder} not found, please provide a valid folder path")
    files = os.listdir(folder)
    assert "requirements.txt" in files, "requirements.txt not found in the folder"
    assert "config.yaml" in files, "config.yaml not found in the folder"
    assert "1" in files, "Subfolder '1' not found in the folder"
    subfolder_files = os.listdir(os.path.join(folder, '1'))
    assert 'model.py' in subfolder_files, "model.py not found in the folder"
    return folder

  @staticmethod
  def _load_config(config_file: str):
    with open(config_file, 'r') as file:
      config = yaml.safe_load(file)
    return config

  def _validate_config_checkpoints(self):

    assert "type" in self.config.get("checkpoints"), "No loader type specified in the config file"
    loader_type = self.config.get("checkpoints").get("type")
    if not loader_type:
      logger.info("No loader type specified in the config file for checkpoints")
    assert loader_type == "huggingface", "Only huggingface loader supported for now"
    if loader_type == "huggingface":
      assert "repo_id" in self.config.get("checkpoints"), "No repo_id specified in the config file"
      repo_id = self.config.get("checkpoints").get("repo_id")

      # prefer env var for HF_TOKEN but if not provided then use the one from config.yaml if any.
      if 'HF_TOKEN' in os.environ:
        hf_token = os.environ['HF_TOKEN']
      else:
        hf_token = self.config.get("checkpoints").get("hf_token", None)
      return repo_id, hf_token

  @property
  def client(self):
    if self._client is None:
      assert "model" in self.config, "model info not found in the config file"
      model = self.config.get('model')
      assert "user_id" in model, "user_id not found in the config file"
      assert "app_id" in model, "app_id not found in the config file"
      # The owner of the model and the app.
      user_id = model.get('user_id')
      app_id = model.get('app_id')

      base = os.environ.get('CLARIFAI_API_BASE', 'https://api-dev.clarifai.com')

      self._client = BaseClient(user_id=user_id, app_id=app_id, base=base)

    return self._client

  @property
  def model_url(self):
    url_helper = ClarifaiUrlHelper(self._client.auth_helper)
    if self.model_version_id is not None:
      return url_helper.clarifai_url(self.client.user_app_id.user_id,
                                     self.client.user_app_id.app_id, "models", self.model_id)
    else:
      return url_helper.clarifai_url(self.client.user_app_id.user_id,
                                     self.client.user_app_id.app_id, "models", self.model_id,
                                     self.model_version_id)

  def _get_model_proto(self):
    assert "model" in self.config, "model info not found in the config file"
    model = self.config.get('model')

    assert "model_type_id" in model, "model_type_id not found in the config file"
    assert "id" in model, "model_id not found in the config file"
    assert "user_id" in model, "user_id not found in the config file"
    assert "app_id" in model, "app_id not found in the config file"

    model_proto = json_format.ParseDict(model, resources_pb2.Model())

    return model_proto

  def _get_inference_compute_info(self):
    assert ("inference_compute_info" in self.config
           ), "inference_compute_info not found in the config file"
    inference_compute_info = self.config.get('inference_compute_info')
    return json_format.ParseDict(inference_compute_info, resources_pb2.ComputeInfo())

  def check_model_exists(self):
    resp = self.client.STUB.GetModel(
        service_pb2.GetModelRequest(
            user_app_id=self.client.user_app_id, model_id=self.model_proto.id))
    if resp.status.code == status_code_pb2.SUCCESS:
      return True
    return False

  def maybe_create_model(self):
    if self.check_model_exists():
      logger.info(
          f"Model '{self.client.user_app_id.user_id}/{self.client.user_app_id.app_id}/models/{self.model_proto.id}' already exists, "
          f"will create a new version for it.")
      return

    request = service_pb2.PostModelsRequest(
        user_app_id=self.client.user_app_id,
        models=[self.model_proto],
    )
    return self.client.STUB.PostModels(request)

  def _parse_requirements(self):
    # parse the user's requirements.txt to determine the proper base image to build on top of, based on the torch and other large dependencies and it's versions
    # List of dependencies to look for
    dependencies = [
        'torch',
    ]
    # Escape dependency names for regex
    dep_pattern = '|'.join(map(re.escape, dependencies))
    # All possible version specifiers
    version_specifiers = '==|>=|<=|!=|~=|>|<'
    # Compile a regex pattern with verbose mode for readability
    pattern = re.compile(r"""
          ^\s*                                   # Start of line, optional whitespace
          (?P<dependency>""" + dep_pattern + r""")   # Dependency name
          \s*                                   # Optional whitespace
          (?P<specifier>""" + version_specifiers + r""")?  # Optional version specifier
          \s*                                   # Optional whitespace
          (?P<version>[^\s;]+)?                 # Optional version (up to space or semicolon)
          """, re.VERBOSE)

    deendencies_version = {}
    with open(os.path.join(self.folder, 'requirements.txt'), 'r') as file:
      for line in file:
        # Skip empty lines and comments
        line = line.strip()
        if not line or line.startswith('#'):
          continue
        match = pattern.match(line)
        if match:
          dependency = match.group('dependency')
          version = match.group('version')
          deendencies_version[dependency] = version if version else None
    return deendencies_version

  def create_dockerfile(self):
    dockerfile_template = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'dockerfile_template',
        'Dockerfile.template',
    )

    with open(dockerfile_template, 'r') as template_file:
      dockerfile_template = template_file.read()

    dockerfile_template = Template(dockerfile_template)

    # Get the Python version from the config file
    build_info = self.config.get('build_info', {})
    if 'python_version' in build_info:
      python_version = build_info['python_version']
      if python_version not in self.AVAILABLE_PYTHON_IMAGES:
        logger.error(
            f"Python version {python_version} not supported, please use one of the following versions: {self.AVAILABLE_PYTHON_IMAGES}"
        )
        return
      logger.info(
          f"Using Python version {python_version} from the config file to build the Dockerfile")
    else:
      logger.info(
          f"Python version not found in the config file, using default Python version: {self.DEFAULT_PYTHON_VERSION}"
      )
      python_version = self.DEFAULT_PYTHON_VERSION

    base_image = self.PYTHON_BASE_IMAGE.format(python_version=python_version)

    # Parse the requirements.txt file to determine the base image
    dependencies = self._parse_requirements()
    if 'torch' in dependencies and dependencies['torch']:
      torch_version = dependencies['torch']

      for image in self.AVAILABLE_TORCH_IMAGES:
        if torch_version in image and f'py{python_version}' in image:
          base_image = self.TORCH_BASE_IMAGE.format(
              torch_version=torch_version,
              python_version=python_version,
              cuda_version=self.DEFAULT_CUDA_VERSION)
          logger.info(f"Using Torch version {torch_version} base image  to build the Docker image")
          break

    # Replace placeholders with actual values
    dockerfile_content = dockerfile_template.safe_substitute(
        name='main',
        BASE_IMAGE=base_image,
    )

    # Write Dockerfile
    with open(os.path.join(self.folder, 'Dockerfile'), 'w') as dockerfile:
      dockerfile.write(dockerfile_content)

  @property
  def checkpoint_path(self):
    return os.path.join(self.folder, self.checkpoint_suffix)

  @property
  def checkpoint_suffix(self):
    return '1/checkpoints'

  @property
  def tar_file(self):
    return f"{self.folder}.tar.gz"

  def download_checkpoints(self):
    if not self.config.get("checkpoints"):
      logger.info("No checkpoints specified in the config file")
      return True

    repo_id, hf_token = self._validate_config_checkpoints()

    loader = HuggingFaceLoader(repo_id=repo_id, token=hf_token)
    success = loader.download_checkpoints(self.checkpoint_path)

    if not success:
      logger.error(f"Failed to download checkpoints for model {repo_id}")
    else:
      logger.info(f"Downloaded checkpoints for model {repo_id}")
    return success

  def _concepts_protos_from_concepts(self, concepts):
    concept_protos = []
    for concept in concepts:
      concept_protos.append(resources_pb2.Concept(
          id=str(concept[0]),
          name=concept[1],
      ))
    return concept_protos

  def hf_labels_to_config(self, labels, config_file):
    with open(config_file, 'r') as file:
      config = yaml.safe_load(file)
    model = config.get('model')
    model_type_id = model.get('model_type_id')
    assert model_type_id in self.CONCEPTS_REQUIRED_MODEL_TYPE, f"Model type {model_type_id} not supported for concepts"
    concept_protos = self._concepts_protos_from_concepts(labels)

    config['concepts'] = [{'id': concept.id, 'name': concept.name} for concept in concept_protos]

    with open(config_file, 'w') as file:
      yaml.dump(config, file, sort_keys=False)
    concepts = config.get('concepts')
    logger.info(f"Updated config.yaml with {len(concepts)} concepts.")

  def get_model_version_proto(self):

    model_version_proto = resources_pb2.ModelVersion(
        pretrained_model_config=resources_pb2.PretrainedModelConfig(),
        inference_compute_info=self.inference_compute_info,
    )

    model_type_id = self.config.get('model').get('model_type_id')
    if model_type_id in self.CONCEPTS_REQUIRED_MODEL_TYPE:

      if 'concepts' in self.config:
        labels = self.config.get('concepts')
        logger.info(f"Found {len(labels)} concepts in the config file.")
        for concept in labels:
          concept_proto = json_format.ParseDict(concept, resources_pb2.Concept())
          model_version_proto.output_info.data.concepts.append(concept_proto)
      else:
        labels = HuggingFaceLoader.fetch_labels(self.checkpoint_path)
        logger.info(f"Found {len(labels)} concepts from the model checkpoints.")
        # sort the concepts by id and then update the config file
        labels = sorted(labels.items(), key=lambda x: int(x[0]))

        config_file = os.path.join(self.folder, 'config.yaml')
        self.hf_labels_to_config(labels, config_file)

        model_version_proto.output_info.data.concepts.extend(
            self._concepts_protos_from_concepts(labels))
    return model_version_proto

  def upload_model_version(self, download_checkpoints):
    file_path = f"{self.folder}.tar.gz"
    logger.info(f"Will tar it into file: {file_path}")

    model_type_id = self.config.get('model').get('model_type_id')

    if (model_type_id in self.CONCEPTS_REQUIRED_MODEL_TYPE) and 'concepts' not in self.config:
      logger.info(
          f"Model type {model_type_id} requires concepts to be specified in the config.yaml file.."
      )
      if self.config.get("checkpoints"):
        logger.info(
            "Checkpoints specified in the config.yaml file, will download the HF model's config.json file to infer the concepts."
        )

        if not download_checkpoints and not HuggingFaceLoader.validate_config(
            self.checkpoint_path):

          input(
              "Press Enter to download the HuggingFace model's config.json file to infer the concepts and continue..."
          )
          repo_id, hf_token = self._validate_config_checkpoints()
          loader = HuggingFaceLoader(repo_id=repo_id, token=hf_token)
          loader.download_config(self.checkpoint_path)

      else:
        logger.error(
            "No checkpoints specified in the config.yaml file to infer the concepts. Please either specify the concepts directly in the config.yaml file or include a checkpoints section to download the HF model's config.json file to infer the concepts."
        )
        return

    model_version_proto = self.get_model_version_proto()

    if download_checkpoints:
      tar_cmd = f"tar --exclude=*~ -czvf {self.tar_file} -C {self.folder} ."
    else:  # we don't want to send the checkpoints up even if they are in the folder.
      logger.info(f"Skipping {self.checkpoint_path} in the tar file that is uploaded.")
      tar_cmd = f"tar --exclude={self.checkpoint_suffix} --exclude=*~ -czvf {self.tar_file} -C {self.folder} ."
    # Tar the folder
    logger.debug(tar_cmd)
    os.system(tar_cmd)
    logger.info("Tarring complete, about to start upload.")

    file_size = os.path.getsize(self.tar_file)
    logger.info(f"Size of the tar is: {file_size} bytes")

    self.maybe_create_model()

    for response in self.client.STUB.PostModelVersionsUpload(
        self.model_version_stream_upload_iterator(model_version_proto, file_path),):
      percent_completed = 0
      if response.status.code == status_code_pb2.UPLOAD_IN_PROGRESS:
        percent_completed = response.status.percent_completed
      details = response.status.details

      _clear_line()
      print(
          f"Status: {response.status.description}, "
          f"Progress: {percent_completed}% - {details} ",
          f"request_id: {response.status.req_id}",
          end='\r',
          flush=True)
    print()
    if response.status.code != status_code_pb2.MODEL_BUILDING:
      logger.error(f"Failed to upload model version: {response}")
      return
    self.model_version_id = response.model_version_id
    logger.info(f"Created Model Version ID: {self.model_version_id}")
    logger.info(f"Full url to that version is: {self.model_url}")

    success = self.monitor_model_build()
    if success:  # cleanup the tar_file if it exists
      if os.path.exists(self.tar_file):
        logger.info(f"Cleaning up upload file: {self.tar_file}")
        os.remove(self.tar_file)

  def model_version_stream_upload_iterator(self, model_version_proto, file_path):
    yield self.init_upload_model_version(model_version_proto, file_path)
    with open(file_path, "rb") as f:
      file_size = os.path.getsize(file_path)
      chunk_size = int(127 * 1024 * 1024)  # 127MB chunk size
      num_chunks = (file_size // chunk_size) + 1
      logger.info("Uploading file...")
      logger.info(f"File size: {file_size}")
      logger.info(f"Chunk size: {chunk_size}")
      logger.info(f"Number of chunks: {num_chunks}")
      read_so_far = 0
      for part_id in range(num_chunks):
        try:
          chunk_size = min(chunk_size, file_size - read_so_far)
          chunk = f.read(chunk_size)
          if not chunk:
            break
          read_so_far += len(chunk)
          yield service_pb2.PostModelVersionsUploadRequest(
              content_part=resources_pb2.UploadContentPart(
                  data=chunk,
                  part_number=part_id + 1,
                  range_start=read_so_far,
              ))
        except Exception as e:
          logger.exception(f"\nError uploading file: {e}")
          break

    if read_so_far == file_size:
      logger.info("\nUpload complete!, waiting for model build...")

  def init_upload_model_version(self, model_version_proto, file_path):
    file_size = os.path.getsize(file_path)
    logger.info(f"Uploading model version of model {self.model_proto.id}")
    logger.info(f"Using file '{os.path.basename(file_path)}' of size: {file_size} bytes")
    return service_pb2.PostModelVersionsUploadRequest(
        upload_config=service_pb2.PostModelVersionsUploadConfig(
            user_app_id=self.client.user_app_id,
            model_id=self.model_proto.id,
            model_version=model_version_proto,
            total_size=file_size,
            is_v3=self.is_v3,
        ))

  def monitor_model_build(self):
    st = time.time()
    while True:
      resp = self.client.STUB.GetModelVersion(
          service_pb2.GetModelVersionRequest(
              user_app_id=self.client.user_app_id,
              model_id=self.model_proto.id,
              version_id=self.model_version_id,
          ))
      status_code = resp.model_version.status.code
      if status_code == status_code_pb2.MODEL_BUILDING:
        print(f"Model is building... (elapsed {time.time() - st:.1f}s)", end='\r', flush=True)
        time.sleep(1)
      elif status_code == status_code_pb2.MODEL_TRAINED:
        logger.info(f"\nModel build complete! (elapsed {time.time() - st:.1f}s)")
        logger.info(f"Check out the model at {self.model_url}")
        return True
      else:
        logger.info(
            f"\nModel build failed with status: {resp.model_version.status} and response {resp}")
        return False


def main(folder, download_checkpoints, skip_dockerfile):
  uploader = ModelUploader(folder)
  if download_checkpoints:
    uploader.download_checkpoints()
  if not skip_dockerfile:
    uploader.create_dockerfile()
  exists = uploader.check_model_exists()
  if exists:
    logger.info(
        f"Model already exists at {uploader.model_url}, this upload will create a new version for it."
    )
  else:
    logger.info(f"New model will be created at {uploader.model_url} with it's first version.")

  input("Press Enter to continue...")
  uploader.upload_model_version(download_checkpoints)
