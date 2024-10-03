import importlib.util
import json
import os
import subprocess

from clarifai.utils.logging import logger


class HuggingFaceLoarder:

  def __init__(self, repo_id=None, token=None):
    self.repo_id = repo_id
    self.token = token
    if token:
      try:
        if importlib.util.find_spec("huggingface_hub") is None:
          raise ImportError(
              "The 'huggingface_hub' package is not installed. Please install it using 'pip install huggingface_hub'."
          )
        os.environ['HF_TOKEN'] = token
        subprocess.run(f'huggingface-cli login --token={os.environ["HF_TOKEN"]}', shell=True)
      except Exception as e:
        Exception("Error setting up Hugging Face token ", e)

  def download_checkpoints(self, checkpoint_path: str):
    # throw error if huggingface_hub wasn't installed
    try:
      from huggingface_hub import snapshot_download
    except ImportError:
      raise ImportError(
          "The 'huggingface_hub' package is not installed. Please install it using 'pip install huggingface_hub'."
      )
    if os.path.exists(checkpoint_path) and self.validate_download(checkpoint_path):
      logger.info("Checkpoints already exist")
      return True
    else:
      os.makedirs(checkpoint_path, exist_ok=True)
      try:
        is_hf_model_exists = self.validate_hf_model()
        if not is_hf_model_exists:
          logger.error("Model %s not found on Hugging Face" % (self.repo_id))
          return False
        snapshot_download(
            repo_id=self.repo_id, local_dir=checkpoint_path, local_dir_use_symlinks=False)
      except Exception as e:
        logger.exception(f"Error downloading model checkpoints {e}")
        return False
      finally:
        is_downloaded = self.validate_download(checkpoint_path)
        if not is_downloaded:
          logger.error("Error validating downloaded model checkpoints")
          return False
      return True

  def validate_hf_model(self,):
    # check if model exists on HF

    from huggingface_hub import file_exists, repo_exists
    return repo_exists(self.repo_id) and file_exists(self.repo_id, 'config.json')

  def validate_download(self, checkpoint_path: str):
    # check if model exists on HF
    from huggingface_hub import list_repo_files
    checkpoint_dir_files = [
        f for dp, dn, fn in os.walk(os.path.expanduser(checkpoint_path)) for f in fn
    ]
    return (len(checkpoint_dir_files) >= len(list_repo_files(self.repo_id))) and len(
        list_repo_files(self.repo_id)) > 0

  def fetch_labels(self, checkpoint_path: str):
    # Fetch labels for classification, detection and segmentation models
    config_path = os.path.join(checkpoint_path, 'config.json')
    with open(config_path, 'r') as f:
      config = json.load(f)

    labels = config['id2label']
    return labels
