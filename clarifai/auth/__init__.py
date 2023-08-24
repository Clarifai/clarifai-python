# Note: clarifai.auth is just added for backwards compatibility. It will be removed in the future.

from clarifai.client.auth.register import V2Stub
from clarifai.client.auth.stub import create_stub

__all__ = ('V2Stub', 'create_stub')
