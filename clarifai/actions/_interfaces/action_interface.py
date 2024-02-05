from abc import ABC, abstractmethod


class ActionInterface(ABC):
    @abstractmethod
    def execute(self, args):
        pass

    def print_help(self):
        pass
