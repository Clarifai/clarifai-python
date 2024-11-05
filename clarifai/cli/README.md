# Clarifai CLI

## Overview

Clarifai offers a user-friendly interface for deploying your local model into production with Clarifai, featuring:

* A convenient command-line interface (CLI)
* Easy implementation and testing in Python
* No need for MLops expertise.

## Compute Orchestration

Quick example for deploying a `visual-classifier` model

### Login

First, login to cli using clarifai account details in a config file as shown below:

```bash
$ clarifai login --config <config-filepath>
```

### Setup

To prepare for deployment step, we have to setup a Compute Cluster with Nodepool of required server config to deploy the model.

So, First, create a new Compute Cluster
```bash
$ clarifai computecluster create --config <compute-cluster-config-filepath>
```

Then, create a new Nodepool in the created Compute Cluster
```bash
$ clarifai nodepool create --config <nodepool-config-filepath>
```

### Deployment

After setup, we can deploy the `visual-classifier` model using a deployment config file as shown below:

```bash
$ clarifai deployment create --config <deployment-config-filepath>
```

### List Resources

List out existing Compute Clusters:

```bash
$ clarifai computecluster list
```

List out existing Nodepools:

```bash
$ clarifai nodepool list --compute_cluster_id <compute-cluster-id>
```

List out existing Deployments:

```bash
$ clarifai deployment list --nodepool_id <nodepool-id>
```

### Delete Resources

Delete existing Deployment:

```bash
$ clarifai deployment delete --nodepool_id <nodepool-id> --deployment_id <deployment-id>
```

Delete existing Nodepool:

```bash
$ clarifai nodepool delete --compute_cluster_id <compute-cluster-id> --nodepool_id <nodepool-id>
```

Delete existing Compute Clusters:

```bash
$ clarifai computecluster delete --compute_cluster_id <compute-cluster-id>
```

## Learn More

* [Example Configs](https://github.com/Clarifai/examples/tree/main/ComputeOrchestration/configs)
