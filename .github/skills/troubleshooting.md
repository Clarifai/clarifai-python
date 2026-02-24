# Troubleshooting

## Installation Issues
- **Import errors**: Ensure `pip install -e .` completed successfully or use global clarifai installation
- **CLI not found**: Ensure virtual environment is activated and package installed
- **Network timeouts**: Common issue; document as environment limitation
- **"ModuleNotFoundError: google"**: Missing protobuf dependencies; requires successful `pip install -e .`

## Development Issues
- **Test failures**: May require `CLARIFAI_PAT` environment variable for API tests
- **Lint failures**: Use `ruff check . --fix` to auto-fix most style issues
- **CLI authentication**: Set up with `clarifai login` or `CLARIFAI_PAT` environment variable
- **Import issues after changes**: Reinstall with `pip install -e .` to pick up changes

## Common Error Messages
- `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org')`: Network/firewall limitation
- `error: subprocess-exited-with-error` during pip install: Network connectivity issue
- `ModuleNotFoundError`: Missing dependencies due to incomplete installation
- `TimeoutError: The read operation timed out`: PyPI connectivity problem
