=================
Predict Tutorial
=================


Predict with Models
==========================

For more information on any of the public models, visit https://clarifai.com/models

.. code-block:: python
   :linenos:

   # Note: CLARIFAI_PAT must be set as env variable.
   from clarifai.client.model import Model

   # Model Predict
   model_prediction = Model("https://clarifai.com/anthropic/completion/models/claude-v2").predict_by_bytes(b"Write a tweet on future of AI", "text")

   model = Model(user_id="user_id", app_id="app_id", model_id="model_id")
   model_prediction = model.predict_by_url(url="url", input_type="image") # Supports image, text, audio, video

   # Customizing Model Inference Output
   model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                     output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
   model_prediction = model.predict_by_filepath(filepath="local_filepath", input_type="text") # Supports image, text, audio, video

   model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                     output_config={"sample_ms": 2000}) # Return predictions for specified interval
   model_prediction = model.predict_by_url(url="VIDEO_URL", input_type="video")

Predict with Workflow
==========================

For more information on any of the public workflows, visit https://clarifai.com/workflows

.. code-block:: python
   :linenos:

   # Note: CLARIFAI_PAT must be set as env variable.
   from clarifai.client.workflow import Workflow

   # Workflow Predict
   workflow = Workflow("workflow_url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
   workflow_prediction = workflow.predict_by_url(url="url", input_type="image") # Supports image, text, audio, video

   # Customizing Workflow Inference Output
   workflow = Workflow(user_id="user_id", app_id="app_id", workflow_id="workflow_id",
                     output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
   workflow_prediction = workflow.predict_by_filepath(filepath="local_filepath", input_type="text") # Supports image, text, audio, video
