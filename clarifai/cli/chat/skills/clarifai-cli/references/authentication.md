# Clarifai CLI Authentication

## Overview

The Clarifai CLI uses Personal Access Tokens (PAT) for authentication. Contexts store your credentials and settings.

## Login

```bash
clarifai login
```

This interactive command:
1. Prompts for your PAT
2. Prompts for user ID and app ID
3. Creates a CLI context

## Environment Variables

You can also set credentials via environment variables:

```bash
export CLARIFAI_PAT="your_personal_access_token"
export CLARIFAI_USER_ID="your_user_id"
export CLARIFAI_APP_ID="your_app_id"
```

## Context Management

### List Contexts

```bash
clarifai config list
```

### Switch Context

```bash
clarifai config use my-context
```

### Delete Context

```bash
clarifai config delete old-context
```

### Show Current Context

```bash
clarifai config show
```

## Context File Location

Contexts are stored in `~/.clarifai/config.yaml`

## Required Credentials

| Operation | Required |
|-----------|----------|
| Model upload | PAT, user_id, app_id |
| Model predict | PAT |
| Pipeline run | PAT, compute_cluster_id, nodepool_id |
| Local runner | PAT, user_id, app_id, compute_cluster_id, nodepool_id |

## Getting Your PAT

1. Go to https://clarifai.com/settings/security
2. Click "Create Personal Access Token"
3. Copy the token
4. Run `clarifai login` and paste it