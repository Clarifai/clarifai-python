"""
This is simply the main file for the server that imports ModelRunner implementation
and starts the server.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import pickle
from concurrent import futures
from typing import Optional

from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_protocol.utils.grpc_server import GRPCServer

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.utils.logging import logger
from clarifai.utils.secrets import get_secrets_path, load_secrets, start_secrets_watcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help="The port to host the gRPC server at.",
        choices=range(1024, 65535),
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        default=os.environ.get('CLARIFAI_NUM_THREADS', 32),
        help="The number of threads to use for the gRPC server. Runner (ie. grpc=False) threads are read from the config file and ModelBuilder defaults.",
        choices=range(1, 129),
    )  # pylint: disable=range-builtin-not-iterating
    parser.add_argument(
        '--max_queue_size',
        type=int,
        default=10,
        help='Max queue size of requests before we begin to reject requests (default: 10).',
        choices=range(1, 21),
    )  # pylint: disable=range-builtin-not-iterating
    parser.add_argument(
        '--max_msg_length',
        type=int,
        default=1024 * 1024 * 1024,
        help='Max message length of grpc requests (default: 1 GB).',
    )
    parser.add_argument(
        '--enable_tls',
        action='store_true',
        default=False,
        help='Set to true to enable TLS (default: False) since this server is meant for local development only.',
    )
    parser.add_argument(
        '--grpc',
        action='store_true',
        default=False,
        help='Set to true to start the gRPC server (default: False). If set to false, the server will not start and only the runner loop will start to fetch work from the API.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='The path to the model directory that contains implemention of the model.',
    )
    parser.add_argument(
        '--disable_secrets_reload',
        action='store_true',
        default=os.environ.get('CLARIFAI_DISABLE_SECRETS_RELOAD', '').lower()
        in ('true', '1', 'yes'),
        help='Disable automatic model reloading when secrets change (default: False, can be set via CLARIFAI_DISABLE_SECRETS_RELOAD env var). Use this if experiencing GPU memory issues or subprocess problems.',
    )

    parsed_args = parser.parse_args()

    server = ModelServer(
        parsed_args.model_path, disable_secrets_reload=parsed_args.disable_secrets_reload
    )
    server.serve(
        port=parsed_args.port,
        pool_size=parsed_args.pool_size,
        max_queue_size=parsed_args.max_queue_size,
        max_msg_length=parsed_args.max_msg_length,
        enable_tls=parsed_args.enable_tls,
        grpc=parsed_args.grpc,
    )


class ModelServer:
    def __init__(self, model_path, disable_secrets_reload=None):
        self.model_path = model_path

        # Check environment variable if not explicitly set
        if disable_secrets_reload is None:
            disable_secrets_reload = os.environ.get(
                'CLARIFAI_DISABLE_SECRETS_RELOAD', ''
            ).lower() in ('true', '1', 'yes')

        self.disable_secrets_reload = disable_secrets_reload
        self._servicer = None
        self._runner = None
        self._secrets_path = get_secrets_path()
        self._watcher_thread = None

        # Initialize secrets system with enhanced validation
        self._initialize_secrets_system()

        # Build model after secrets are loaded - use subprocess for clean GPU memory
        self._builder = ModelBuilder(model_path, download_validation_only=True)
        logger.info("Creating initial model instance in subprocess...")
        self._current_model = self._create_model_in_subprocess()
        if self._current_model is None:
            logger.error("Failed to create initial model in subprocess")
            raise RuntimeError("Failed to create initial model")
        logger.info("ModelServer initialized successfully")

    def _initialize_secrets_system(self):
        """Initialize the secrets management system with comprehensive validation."""
        if not self._secrets_path:
            logger.info("No secrets path configured, running without secrets")
            return

        logger.info(f"Initializing secrets system with path: {self._secrets_path}")

        # Load existing secrets if directory exists
        if self._secrets_path.exists():
            try:
                loaded_secrets = load_secrets(self._secrets_path)
                if loaded_secrets:
                    logger.info(f"Loaded {len(loaded_secrets)} initial secrets")
                else:
                    logger.info("Secrets directory exists but contains no valid secrets")
            except Exception as e:
                logger.error(f"Error loading initial secrets: {e}")
        else:
            logger.info(f"Secrets directory does not exist yet: {self._secrets_path}")

        # Start the watcher unless disabled
        if self.disable_secrets_reload:
            logger.info("Secrets watcher disabled - secrets will not be automatically reloaded")
            return

        # Always start the watcher regardless of current directory state
        # This handles the case where secrets are mounted after server startup
        try:
            self._watcher_thread = start_secrets_watcher(
                self._secrets_path, self.reload_model_on_secrets_change, interval=10.0
            )
            logger.info("Secrets watcher started successfully")
        except Exception as e:
            logger.error(f"Failed to start secrets watcher: {e}")
            # Don't fail server startup if watcher fails
            self._watcher_thread = None

    def _create_model_in_subprocess(self) -> Optional[object]:
        """Create model instance in a subprocess to ensure clean GPU memory allocation.
        
        This avoids GPU memory duplication issues by creating the model in an isolated process
        that starts with clean GPU state. The model instance is returned via IPC.
        
        Returns:
            The model instance created in the subprocess, or None if creation failed.
        """
        logger.info("Creating model instance in isolated subprocess...")
        
        try:
            # Create a temporary file for IPC
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                temp_file = f.name
            
            # Subprocess code to create model and serialize it
            subprocess_code = f'''
import sys
import pickle
import tempfile
sys.path.insert(0, '{os.path.dirname(os.path.dirname(__file__))}')

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.logging import logger

try:
    logger.info("Subprocess: Starting model creation...")
    builder = ModelBuilder('{self.model_path}', download_validation_only=True)
    model = builder.create_model_instance()
    logger.info("Subprocess: Model created successfully")
    
    # Serialize the model instance
    with open('{temp_file}', 'wb') as f:
        pickle.dump(model, f)
    logger.info("Subprocess: Model serialized successfully")
    
except Exception as e:
    logger.error(f"Subprocess: Error creating model: {{e}}")
    import traceback
    logger.error(f"Subprocess: Traceback: {{traceback.format_exc()}}")
    sys.exit(1)
'''
            
            # Run model creation in subprocess
            process = subprocess.Popen(
                [sys.executable, "-c", subprocess_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for subprocess to complete
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            
            if process.returncode == 0:
                # Load the model from the temporary file
                try:
                    with open(temp_file, 'rb') as f:
                        model = pickle.load(f)
                    logger.info("Successfully loaded model from subprocess")
                    return model
                except Exception as e:
                    logger.error(f"Failed to deserialize model from subprocess: {e}")
                    return None
            else:
                logger.error(f"Subprocess model creation failed with return code {process.returncode}")
                logger.error(f"Subprocess stdout: {stdout}")
                logger.error(f"Subprocess stderr: {stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Subprocess model creation timed out")
            process.kill()
            return None
        except Exception as e:
            logger.error(f"Error in subprocess model creation: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    def reload_model_on_secrets_change(self) -> None:
        """Reload model and environment secrets when the secrets directory changes.

        This method uses subprocess-based model creation to avoid GPU memory duplication
        and automatically reclaim GPU memory without framework-specific cleanup.
        """
        logger.info("Detected secrets change, initiating subprocess-based model reload...")

        # Step 1: Reload secrets from filesystem
        if self._secrets_path is not None:
            try:
                loaded_secrets = load_secrets(self._secrets_path)
                if loaded_secrets:
                    logger.info(f"Reloaded {len(loaded_secrets)} secrets")
                else:
                    logger.warning("No secrets loaded during reload")
            except Exception as e:
                logger.error(f"Failed to reload secrets: {e}")
                return

        # Step 2: Create new model instance in subprocess (no cleanup needed)
        # The subprocess starts with clean GPU memory, avoiding duplication issues
        try:
            logger.info("Creating new model instance in isolated subprocess...")
            new_model = self._create_model_in_subprocess()
            
            if new_model is None:
                logger.error("Failed to create model in subprocess - keeping current model")
                return
                
            # Only update the current model reference after successful creation
            self._current_model = new_model
            logger.info("Model instance created successfully in subprocess")
        except Exception as e:
            logger.error(f"Failed to create model in subprocess: {e}")
            # Keep the previous model instance if subprocess creation fails
            return

        # Step 3: Update servicer with new model
        if self._servicer and self._current_model:
            try:
                self._servicer.set_model(self._current_model)
                logger.info("Updated servicer with new model instance")
            except Exception as e:
                logger.error(f"Failed to update servicer with new model: {e}")

        # Step 4: Update runner with new model  
        if self._runner and self._current_model:
            try:
                self._runner.set_model(self._current_model)
                logger.info("Updated runner with new model instance")
            except Exception as e:
                logger.error(f"Failed to update runner with new model: {e}")

        logger.info("Subprocess-based model reload sequence completed successfully")

    def shutdown(self):
        """Gracefully shutdown the server and cleanup resources."""
        logger.info("Shutting down ModelServer...")

        # Stop the watcher thread
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Stopping secrets watcher...")
            # Note: Since it's a daemon thread, it will stop when main process exits
            
        # Note: No need for model.cleanup() since we use subprocess-based model creation
        # The subprocess approach automatically handles GPU memory cleanup
        logger.info("ModelServer shutdown completed")

    def serve(
        self,
        port=8000,
        pool_size=32,
        num_threads=0,
        max_queue_size=10,
        max_msg_length=1024 * 1024 * 1024,
        enable_tls=False,
        grpc=False,
        user_id: Optional[str] = os.environ.get("CLARIFAI_USER_ID", None),
        compute_cluster_id: Optional[str] = os.environ.get("CLARIFAI_COMPUTE_CLUSTER_ID", None),
        nodepool_id: Optional[str] = os.environ.get("CLARIFAI_NODEPOOL_ID", None),
        runner_id: Optional[str] = os.environ.get("CLARIFAI_RUNNER_ID", None),
        base_url: Optional[str] = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com"),
        pat: Optional[str] = os.environ.get("CLARIFAI_PAT", None),
        context=None,  # This is the current context object that contains user_id, app_id, model_id, etc.
    ):
        # `num_threads` can be set in config.yaml or via the environment variable CLARIFAI_NUM_THREADS="<integer>".
        # Note: The value in config.yaml takes precedence over the environment variable.
        if num_threads == 0:
            num_threads = self._builder.config.get("num_threads")
        # Setup the grpc server for local development.
        if grpc:
            self.start_servicer(
                port,
                pool_size,
                max_queue_size,
                max_msg_length,
                enable_tls,
            )
        else:
            # start the runner with the proper env variables and as a runner protocol.
            self.start_runner(
                context,
                compute_cluster_id,
                user_id,
                nodepool_id,
                runner_id,
                base_url,
                pat,
                num_threads,
            )

    def start_servicer(self, port, pool_size, max_queue_size, max_msg_length, enable_tls):
        # initialize the servicer with the runner so that it gets the predict(), generate(), stream() classes.
        self._servicer = ModelServicer(self._current_model)

        server = GRPCServer(
            futures.ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="ServeCalls",
            ),
            max_msg_length,
            max_queue_size,
        )
        server.add_port_to_server('[::]:%s' % port, enable_tls)

        service_pb2_grpc.add_V2Servicer_to_server(self._servicer, server)
        server.start()
        logger.info("Started server on port %s", port)
        logger.info(f"Access the model at http://localhost:{port}")
        server.wait_for_termination()

    def start_runner(
        self,
        context,
        compute_cluster_id,
        user_id,
        nodepool_id,
        runner_id,
        base_url,
        pat,
        num_threads,
    ):
        # initialize the Runner class. This is what the user implements.
        assert compute_cluster_id is not None, "compute_cluster_id must be set for the runner."
        assert nodepool_id is not None, "nodepool_id must be set for the runner"
        assert runner_id is not None, "runner_id must be set for the runner."
        assert base_url is not None, "base_url must be set for the runner."
        self._runner = ModelRunner(
            model=self._current_model,
            user_id=user_id,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            runner_id=runner_id,
            base_url=base_url,
            pat=pat,
            num_parallel_polls=num_threads,
        )

        if context is None:
            logger.debug("Context is None. Skipping code snippet generation.")
        else:
            method_signatures = self._builder.get_method_signatures(mocking=False)
            from clarifai.runners.utils import code_script

            snippet = code_script.generate_client_script(
                method_signatures,
                user_id=context.user_id,
                app_id=context.app_id,
                model_id=context.model_id,
                deployment_id=context.deployment_id,
                base_url=context.api_base,
            )
            logger.info(
                "✅ Your model is running locally and is ready for requests from the API...\n"
            )
            logger.info(
                f"> Code Snippet: To call your model via the API, use this code snippet:\n{snippet}"
            )
            logger.info(
                f"> Playground:   To chat with your model, visit: {context.ui}/playground?model={context.model_id}__{context.model_version_id}&user_id={context.user_id}&app_id={context.app_id}\n"
            )
            logger.info(
                f"> API URL:      To call your model via the API, use this model URL: {context.ui}/{context.user_id}/{context.app_id}/models/{context.model_id}\n"
            )
            logger.info("Press CTRL+C to stop the runner.\n")
        self._runner.start()  # start the runner to fetch work from the API.


if __name__ == '__main__':
    main()
