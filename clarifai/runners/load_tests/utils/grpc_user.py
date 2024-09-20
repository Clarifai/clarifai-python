#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subclass GrpcUser when load testing via Clarifai gRPC client.
Set your desired host and key as class attributes of your user!

Either define the host attribute when subclassing GrpcUser or set it in the UI.
Leaving host as None will use the Clarifai production environment.
"""
import logging
import sys
import time
from datetime import datetime
from typing import Callable

import grpc.experimental.gevent as grpc_gevent
from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from locust import User, events

from clarifai.client import BaseClient

grpc_gevent.init_gevent()

log = logging.getLogger("console")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh = logging.FileHandler(f'locust_{datetime.now().strftime("%S")}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


class GrpcUser(User):
  abstract = True
  # host = None
  stub_class = service_pb2_grpc.V2Stub
  client = BaseClient

  # key = None

  def __init__(self, environment):
    super().__init__(environment)
    self.client = BaseClient.from_env()
    self.stub = self.client.STUB

  def predict(self, name: str, func: Callable, *args, **kwargs):
    """call allows locust to record statistics for Clarifai's gRPC client.

    Args:
            func (callable): the function to call
            name (string): identifier for the call to record on locust stats
            *args (any): args to pass to callable
            **kwargs (any): kwargs to pass to callable

    Returns:
        response from calling Callable
    """
    start_time = time.time()
    response_length = 0
    resp = None
    response_time = 0
    exception = None

    try:
      resp = func(*args, **kwargs)
      response_length = sys.getsizeof(resp)
    except Exception as e:
      exception = e
      log.error(f"Request raised exception '{e}'", stack_info=True)
    else:
      if resp.status.code != status_code_pb2.SUCCESS:
        msg = f"Request returned but had response status code '{resp.status.code}' with body '{resp}'"
        log.warning(msg)
        exception = Exception(msg)

    response_time = int((time.time() - start_time) * 1000)
    events.request.fire(
        request_type="client",
        name=name,
        response_time=response_time,
        response_length=response_length,
        exception=exception,
        response=resp)
    if resp is not None:
      return resp

  def _validate_host(self, host_url: str):
    if host_url in [None, ""]:
      log.warning("No host URL provided, gRPC stub will be created for the production environment")
      self.host = None
    elif not host_url.startswith("https://"):
      log.warning("Provided host URL does not have https:// prefix, adding...")
      self.host = "https://" + host_url

  def generate(self, name: str, func: Callable, *args, **kwargs):
    """call allows locust to record statistics for Clarifai's gRPC client.

    Args:
            func (callable): the function to call which should yield responses
            name (string): identifier for the call to record on locust stats
            *args (any): args to pass to callable
            **kwargs (any): kwargs to pass to callable

    Returns:
        response from calling Callable
    """
    start_time = time.time()
    response_length = 0
    resp = None
    response_time = 0
    exception = None

    responses = []
    try:
      for resp in func(*args, **kwargs):
        responses.append(resp)
      response_length = sys.getsizeof(responses)
    except Exception as e:
      exception = e
      log.error(f"Request raised exception '{e}'", stack_info=True)
    else:
      if resp.status.code != status_code_pb2.SUCCESS:
        msg = f"Request returned but had response status code '{resp.status.code}' with body '{resp}'"
        log.warning(msg)
        exception = Exception(msg)

    response_time = int((time.time() - start_time) * 1000)
    events.request.fire(
        request_type="client",
        name=name,
        response_time=response_time,
        response_length=response_length,
        exception=exception,
        response=resp)
    return responses
