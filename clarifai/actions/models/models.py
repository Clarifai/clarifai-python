from clarifai.actions._interfaces.action_interface import ActionInterface
from clarifai.utils import logging


class Models(ActionInterface):
    def execute(self, args):
        pass

    def print_help(self):
        logging.get_logger().info(f"action class '{self.__class__.__name__}' doesn't have a helper function defined.")
