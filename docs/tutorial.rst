.. _intro-tutorial:

==============
Tutorial
==============

Each of the examples below is a small independent code snippet within 10 lines that could work by copy and paste to a python source code file. By playing with them, you should be getting started with Clarifai API. For more information about the API, check the API Reference.


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



Upload Images
=============

.. code-block:: python
   :linenos:

   from clarifai.rest import ClarifaiApp

   app = ClarifaiApp()

   app.inputs.create_image_from_url(url='https://samples.clarifai.com/puppy.jpeg', concepts=['my puppy'])
   app.inputs.create_image_from_url(url='https://samples.clarifai.com/wedding.jpg', not_concepts=['my puppy'])

Create a Model
==============

.. note:: This assumes you follow through the tutorial and finished the "Upload Images"
          Otherwise you may not be able to create the model.

.. code-block:: python
   :linenos:

   model = app.models.create(model_id="puppy", concepts=["my puppy"])

Train the Model
===============

.. note:: This assumes you follow through the tutorial and finished the "Upload Images"
          and "Create a Model" to create a model.
          Otherwise you may not be able to train the model.

.. code-block:: python
   :linenos:

   model.train()

Predict with Model
==================

.. note:: This assumes you follow through the tutorial and finished the "Upload Images",
          "Create a Model", and "Train the Model".
          Otherwise you may not be able to make predictions with the model.

.. code-block:: python
   :linenos:

   from clarifai.rest import ClarifaiApp

   app = ClarifaiApp()

   model = app.models.get('puppy')
   model.predict_by_url('https://samples.clarifai.com/metro-north.jpg')

Instantiate an Image
====================

.. code-block:: python
   :linenos:

   from clarifai.rest import Image as ClImage

   # make an image with an url
   img = ClImage(url='https://samples.clarifai.com/dog1.jpeg')

   # make an image with a filename
   img = ClImage(filename='/tmp/user/dog.jpg')

   # allow duplicate url
   img = ClImage(url='https://samples.clarifai.com/dog1.jpeg', allow_dup_url=True)

   # make an image with concepts
   img = ClImage(url='https://samples.clarifai.com/dog1.jpeg', \
                 concepts=['cat', 'animal'])

   # make an image with metadata
   img = ClImage(url='https://samples.clarifai.com/dog1.jpeg', \
                 concepts=['cat', 'animal'], \
                 metadata={'id':123,
                           'city':'New York'
                          })

Bulk Import Images
==================

If you have a large amount of images, you may not want to upload them one by one by calling
`app.inputs.create_image_from_url('https://samples.clarifai.com/dog1.jpeg')`

Instead you may want to use the bulk import API.

.. note:: The max number images per batch is 128. If you have more than 128 images to upload,
          you may want to chunk them into 128 or less, and bulk import them batch by batch.

In order to use this, you have to instantiate Image() objects from various sources.

.. code-block:: python
   :linenos:

   from clarifai.rest import ClarifaiApp
   from clarifai.rest import Image as ClImage

   # assume there are 100 urls in the list
   images = []
   for url in urls:
     img = ClImage(url=url)
     images.append(img)

   app.inputs.bulk_create_images(images)


Search the Image
================

.. note:: This assumes you follow through the tutorial and finished the "Upload Images"
          Otherwise you may not be able to search

.. code-block:: python
   :linenos:

   from clarifai.rest import ClarifaiApp

   app = ClarifaiApp()

   app.inputs.search_by_annotated_concepts(concept='my puppy')

   app.inputs.search_by_predicted_concepts(concept='dog')

   app.inputs.search_by_image(url='https://samples.clarifai.com/dog1.jpeg')

   app.inputs.search_by_metadata(metadata={'key':'value'})

