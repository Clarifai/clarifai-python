# CLI Operations

## Available Commands
```bash
clarifai --help                    # Show all available commands
clarifai model --help             # Model operations (list, predict, upload, etc.)
clarifai pipeline --help          # Pipeline operations
clarifai computecluster --help    # Compute cluster management
clarifai nodepool --help          # Nodepool management
clarifai deployment --help        # Deployment management
clarifai login --help             # Authentication setup
```

## Authentication
- Requires `CLARIFAI_PAT` environment variable or config file
- Use `clarifai login` to configure authentication
- Test commands may require valid API credentials

## CLI Help Output
```
$ clarifai --help
Usage: clarifai [OPTIONS] COMMAND [ARGS]...

  Clarifai CLI

Commands:
  computecluster (cc)     Manage Compute Clusters
  config                  Manage configuration profiles
  deployment (dp)         Manage Deployments
  login                   Login command to set PAT
  model                   Manage & Develop Models
  nodepool (np)           Manage Nodepools
  pipeline (pl)           Manage pipelines
  pipeline-step (ps)      Manage pipeline steps
  run                     Execute script with context
  shell-completion        Shell completion script
```

## Package Information
```
$ python -c "import clarifai; print(f'Version: {clarifai.__version__}')"
Clarifai imported successfully
Version: 11.6.8
```
