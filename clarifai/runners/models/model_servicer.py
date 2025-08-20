# This is an enhanced version of ModelServicer that integrates with secrets management
# Add these methods to your existing ModelServicer class

from clarifai.utils.logging import logger
from clarifai.utils.secrets import populate_params_from_secrets, set_request_context


class ModelServicer:
    def __init__(self, model):
        self.model = model

    def update_model(self, new_model):
        """Update the model instance (called when secrets change)."""
        self.model = new_model
        logger.info("ModelServicer updated with new model instance")

    def _prepare_request_with_secrets(self, request):
        """Prepare request by extracting params and populating from secrets if needed."""
        # Extract request parameters (this will depend on your request structure)
        request_params = self._extract_request_params(request)

        # Populate missing parameters from secrets
        updated_params = populate_params_from_secrets(request_params)

        # Set the context for the secrets helper
        set_request_context(updated_params)

        # Update the request object with populated parameters
        self._update_request_params(request, updated_params)

        return request

    def _extract_request_params(self, request):
        """Extract parameters from the request object.

        This method should be customized based on your specific request structure.
        Common patterns include:
        - request.params (if params is a dict-like object)
        - request.inputs[0].data.params (for some protobuf structures)
        - getattr(request, 'parameters', {})
        """
        params = {}

        # Example implementations - adapt based on your request structure:
        if hasattr(request, 'params') and request.params:
            params = dict(request.params)
        elif hasattr(request, 'inputs') and request.inputs:
            for input_obj in request.inputs:
                if hasattr(input_obj, 'data') and hasattr(input_obj.data, 'params'):
                    params.update(dict(input_obj.data.params))
        elif hasattr(request, 'model_parameters'):
            params = dict(request.model_parameters)

        return params

    def _update_request_params(self, request, updated_params):
        """Update the request object with the populated parameters.

        This method should be customized based on your specific request structure.
        """
        # Example implementations - adapt based on your request structure:
        if hasattr(request, 'params'):
            for key, value in updated_params.items():
                if key not in request.params or not request.params[key]:
                    request.params[key] = value
        elif hasattr(request, 'inputs') and request.inputs:
            for input_obj in request.inputs:
                if hasattr(input_obj, 'data') and hasattr(input_obj.data, 'params'):
                    for key, value in updated_params.items():
                        if key not in input_obj.data.params or not input_obj.data.params[key]:
                            input_obj.data.params[key] = value
        elif hasattr(request, 'model_parameters'):
            for key, value in updated_params.items():
                if key not in request.model_parameters or not request.model_parameters[key]:
                    request.model_parameters[key] = value

    def PostModelOutputs(self, request, context):
        """Handle predict requests with secrets integration."""
        try:
            # Prepare request with secrets
            prepared_request = self._prepare_request_with_secrets(request)

            # Call the original model method
            return self.model.predict(prepared_request, context)
        except Exception as e:
            logger.error(f"Error in PostModelOutputs: {e}")
            raise
        finally:
            # Clear request context
            set_request_context({})

    def PostModelOutputsStream(self, request, context):
        """Handle streaming predict requests with secrets integration."""
        try:
            # Prepare request with secrets
            prepared_request = self._prepare_request_with_secrets(request)

            # Call the original model method
            return self.model.stream(prepared_request, context)
        except Exception as e:
            logger.error(f"Error in PostModelOutputsStream: {e}")
            raise
        finally:
            # Clear request context
            set_request_context({})

    def GenerateText(self, request, context):
        """Handle generate requests with secrets integration."""
        try:
            # Prepare request with secrets
            prepared_request = self._prepare_request_with_secrets(request)

            # Call the original model method
            return self.model.generate(prepared_request, context)
        except Exception as e:
            logger.error(f"Error in GenerateText: {e}")
            raise
        finally:
            # Clear request context
            set_request_context({})
