"""
Warning: This part of the client is in beta and its public interface may still change.
"""

from clarifai.rest.solutions.moderation import ModerationSolution


class Solutions(object):

  def __init__(self, api_key=None):
    self.moderation = ModerationSolution(api_key)
