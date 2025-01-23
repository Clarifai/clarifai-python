from clarifai.runners.model_builder import ModelBuilder
from clarifai.utils.logging import logger


def main(folder, download_checkpoints, skip_dockerfile):
  buidler = ModelBuilder(folder)
  if download_checkpoints:
    buidler.download_checkpoints()
  if not skip_dockerfile:
    buidler.create_dockerfile()
  exists = buidler.check_model_exists()
  if exists:
    logger.info(
        f"Model already exists at {buidler.model_url}, this upload will create a new version for it."
    )
  else:
    logger.info(f"New model will be created at {be.model_url} with it's first version.")

  input("Press Enter to continue...")
  buidler.upload_model_version(download_checkpoints)
