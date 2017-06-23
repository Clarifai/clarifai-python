=================
Predict Tutorial
=================


Predict with Public Models
==========================

For more information on any of the public models, visit https://developer.clarifai.com/models

.. code-block:: python

   from clarifai.rest import ClarifaiApp

   app = ClarifaiApp()

   #General model
   model = app.models.get('general-v1.3')

   response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')


   #Travel model
   model = app.models.get('travel-v1.0')

   response = model.predict_by_url(url='https://samples.clarifai.com/travel.jpg')


   #Food model
   model = app.models.get('food-items-v1.0')

   response = model.predict_by_url(url='https://samples.clarifai.com/food.jpg')


   #NSFW model
   model = app.models.get('nsfw-v1.0')

   response = model.predict_by_url(url='https://samples.clarifai.com/nsfw.jpg')


   #Apparel model
   model = app.models.get('apparel')

   response = model.predict_by_url(url='https://samples.clarifai.com/apparel.jpg')


   #Celebrity model
   model = app.models.get('celeb-v1.3')

   response = model.predict_by_url(url='https://samples.clarifai.com/celebrity.jpg')


   #Demographics model
   model = app.models.get('demographics')

   response = model.predict_by_url(url='https://samples.clarifai.com/demographics.jpg')


   #Face Detection model
   model = app.models.get('face-v1.3')

   response = model.predict_by_url(url='https://developer.clarifai.com/static/images/model-samples/face-001.jpg')


   #Focus Detection model
   model = app.models.get('focus')

   response = model.predict_by_url(url='https://samples.clarifai.com/focus.jpg')


   #General Embedding model
   model = app.models.get('general-v1.3', model_type='embed')

   response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')


   #Logo model
   model = app.models.get('logo')

   response = model.predict_by_url(url='https://samples.clarifai.com/logo.jpg')


   #Color model
   model = app.models.get('color', model_type='color')

   response = model.predict_by_url(url='https://samples.clarifai.com/wedding.jpg')



