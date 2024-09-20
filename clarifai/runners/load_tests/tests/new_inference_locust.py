#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import grpc.experimental.gevent as grpc_gevent
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from deploy.locust.locust_v2.utils.grpc_user import GrpcUser
from locust import task

grpc_gevent.init_gevent()


class ApiUser(GrpcUser):
  # wait_time = constant_throughput(10)

  def _setup_req(self):
    if 'CLARIFAI_DEPLOYMENT_ID' in os.environ:
      deployment = resources_pb2.Deployment(
          id=os.environ['CLARIFAI_DEPLOYMENT_ID'], user_id=self.client.user_app_id.user_id)
      runner_selector = resources_pb2.RunnerSelector(deployment=deployment)
    else:
      compute_cluster = resources_pb2.ComputeCluster(
          id=os.environ['CLARIFAI_COMPUTE_CLUSTER_ID'], user_id=self.client.user_app_id.user_id)
      nodepool = resources_pb2.Nodepool(
          id=os.environ['CLARIFAI_NODEPOOL_ID'], compute_cluster=compute_cluster)
      runner_selector = resources_pb2.RunnerSelector(nodepool=nodepool)

    # userAppID = self.client.client.user_app_id
    os.environ["CLARIFAI_MODEL_ID"]
    TEXT = """lowGPT is a website that allows anyone to quickly share and explore useful ChatGPT prompts that can improve their daily workflow.
Generate a highly converting and appealing blog outline based on the following prompt:
How to use FlowGPT to search for the best prompt that makes cold emailing more efficient?"""

    inputs = [resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=TEXT)))]

    req = service_pb2.PostModelOutputsRequest(
        model_id=os.environ["CLARIFAI_MODEL_ID"],
        user_app_id=self.client.user_app_id,
        inputs=inputs,
        runner_selector=runner_selector,
    )
    return req

  @task
  def call_predict(self):

    req = self._setup_req()
    # print(req)
    resp = self.predict("predict on %s" % os.environ['CLARIFAI_MODEL_ID'],
                        self.stub.PostModelOutputs, req)

    if resp.status.code != status_code_pb2.SUCCESS:
      raise Exception("Failed to predict: %s" % resp)


@task
def call_generate(self):

  req = self._setup_req()

  responses = self.generate("generate on %s" % os.environ['CLARIFAI_MODEL_ID'],
                            self.stub.GenerateModelOutputs, req)
  if len(responses) == 0:
    raise Exception("Failed to get responses, length 0")
  if responses[-1].status.code != status_code_pb2.SUCCESS:
    raise Exception("Failed to predict: %s" % responses[-1].status.description)
