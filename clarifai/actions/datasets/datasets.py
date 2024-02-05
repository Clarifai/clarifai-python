import logging

from clarifai.actions._interfaces.action_interface import ActionInterface


class Datasets(ActionInterface):
    def execute(self, args):
        pass

    def print_help(self):
        logging.getLogger().info(f"action class '{self.__class__.__name__}' doesn't have a helper function defined.")
