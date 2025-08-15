__version__ = "11.6.8"

# Note(zeiler): this is to fix old protobuf issues which are fixed in clarifi_grpc/__init__.py
# this is a best effort and users should really just upgrade their protobuf package.
import clarifai_grpc

_ = clarifai_grpc
