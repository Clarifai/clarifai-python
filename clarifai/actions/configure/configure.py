import json
import logging
import os
from json import JSONDecodeError

from clarifai.actions._interfaces.action_interface import ActionInterface
from clarifai.constants.utils import CLARIFAI_CONFIG_FOLDER, CLARIFAI_CONFIG_FILE, DEFAULT_PROFILE_NAME


class Configure(ActionInterface):
    def execute(self, args):
        if vars(args).get('help', None):
            self.print_help()
            return
        else:
            config_path = os.path.join(
                os.path.expanduser("~"),
                CLARIFAI_CONFIG_FOLDER
            )
            config_file = CLARIFAI_CONFIG_FILE
            if not os.path.exists(config_path):
                os.mkdir(config_path)
            if not os.path.isdir(config_path):
                raise AssertionError(f"path for config: {config_path} has to be a directory.")
            full_config_file_path = os.path.join(
                config_path,
                config_file
            )
            config = {}
            if os.path.exists(full_config_file_path):
                f = open(full_config_file_path, 'r')
                try:
                    config = json.loads(f.read())
                except JSONDecodeError as _:
                    logging.getLogger().debug(f"the config file may not be in appropriate JSON format! Check file: {full_config_file_path}")
                    config = {}
                except TypeError as _:
                    logging.getLogger().debug(f"Some other problem reading the config json from file: {full_config_file_path}")
                    config = {}
                f.close()
            f = open(full_config_file_path, "w")
            if args.profile:
                previous_pat = ""
                if args.profile in config:
                    previous_pat = config[args.profile].get("personal_access_token", "")
                masked_pat = f"{''.join(['X']*(len(previous_pat)-2))}{previous_pat[-2:]}" if len(previous_pat) > 2 else ""
                pat_token = input(f"Please enter a PAT i.e personal access token associated with this profile: {args.profile} [{masked_pat}] == ")
                config[args.profile] = {
                    "personal_access_token": str(pat_token).strip()
                }
            else:
                logging.getLogger().debug("presuming that the profile is default")
                pat_token = input(
                    f"Please enter a PAT i.e personal access token associated with this profile: {args.profile} == "
                )
                config[f"{DEFAULT_PROFILE_NAME}"] = {
                        "personal_access_token": str(pat_token).strip()
                }
            f.write(json.dumps(config))
            f.close()
            logging.getLogger().debug(f"The config for clarifai has been (re)created at path: {full_config_file_path}")

    def print_help(self):
        logging.getLogger().info(f"action class '{self.__class__.__name__}' doesn't have a helper function defined.")
        print(f"action class '{self.__class__.__name__}' doesn't have a helper function defined.")
