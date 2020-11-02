=================
Predict Tutorial
=================


Predict with Public Models
==========================

For more information on any of the public models, visit https://clarifai.com/models

.. code-block:: python

   from clarifai.rest import ClarifaiApp

   app = ClarifaiApp()

   #General model
   model = app.models.get(model_id="aaa03c23b3724a16a56b629203edc62c")

   response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')


   #Travel model
   model = app.models.get(model_id='eee28c313d69466f836ab83287a54ed9')

   response = model.predict_by_url(url='https://samples.clarifai.com/travel.jpg')


   #Food model
   model = app.models.get(model_id="bd367be194cf45149e75f01d59f77ba7")

   response = model.predict_by_url(url='https://samples.clarifai.com/food.jpg')


   #NSFW model
   model = app.models.get(model_id="e9576d86d2004ed1a38ba0cf39ecb4b1")

   response = model.predict_by_url(url='https://samples.clarifai.com/nsfw.jpg')


   #Apparel model
   model = app.models.get(model_id="e0be3b9d6a454f0493ac3a30784001ff")

   response = model.predict_by_url(url='https://samples.clarifai.com/apparel.jpg')


   #Celebrity model
   model = app.models.get(model_id="e466caa0619f444ab97497640cefc4dc")

   response = model.predict_by_url(url='https://samples.clarifai.com/celebrity.jpg')


   #Demographics model
   model = app.models.get(model_id="c0c0ac362b03416da06ab3fa36fb58e3")

   response = model.predict_by_url(url='https://samples.clarifai.com/demographics.jpg')


   #Face Detection model
   model = app.models.get(model_id="a403429f2ddf4b49b307e318f00e528b")

   response = model.predict_by_url(url='https://portal.clarifai.com/developer/static/images/model-samples/celeb-001.jpg')


   #General Embedding model
   model = app.models.get(model_id="bbb5f41425b8468d9b7a554ff10f8581")

   response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')


   #Logo model
   model = app.models.get(model_id="c443119bf2ed4da98487520d01a0b1e3")

   response = model.predict_by_url(url='https://samples.clarifai.com/logo.jpg')


   #Color model
   model = app.models.get(model_id="eeed0b6733a644cea07cf4c60f87ebb7")

   response = model.predict_by_url(url='https://samples.clarifai.com/wedding.jpg')
