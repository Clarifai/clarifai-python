# Clarifai CLI

## Overview

Clarifai offers a user-friendly interface for deploying your local model into production with Clarifai, featuring:

* A convenient command-line interface (CLI)
* Easy implementation and testing in Python
* No need for MLops expertise.

## Context Management

Manage CLI contexts for authentication and environment configuration:
### List all contexts
```bash
clarifai config get-contexts
```

### Switch context
```bash
clarifai config use-context production
```
### Show current context
```bash
clarifai config current-context
```

### Create new context
```bash
clarifai config create-context staging --user-id myuser --pat 678***
```
### View entire configuration
```bash
clarifai config view
```
### Delete a context
```bash
clarifai config delete-context old-context
```
### Edit configuration file
```bash
clarifai config edit
```

### Print environment variables for the active context
```bash
clarifai context env
```

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

## Pipelines

### List Pipelines

List all pipelines for the user across all apps:

```bash
$ clarifai pipeline list
```

List pipelines within a specific app:

```bash
$ clarifai pipeline list --app_id <app-id>
```

List with pagination:

```bash
$ clarifai pipeline list --page_no 1 --per_page 10
```

### List Pipeline Steps

List all pipeline steps for the user across all apps:

```bash
$ clarifai pipelinestep list
```

List pipeline steps within a specific app:

```bash
$ clarifai pipelinestep list --app_id <app-id>
```

List pipeline steps for a specific pipeline:

```bash
$ clarifai pipelinestep list --app_id <app-id> --pipeline_id <pipeline-id>
```

### Aliases

Both commands support the `ls` alias for convenience:

```bash
$ clarifai pipeline ls
$ clarifai pipelinestep ls
```

## Learn More

* [Example Configs](https://github.com/Clarifai/examples/tree/main/ComputeOrchestration/configs)
