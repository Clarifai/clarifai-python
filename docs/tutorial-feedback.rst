=================
Feedback Tutorial
=================

Concept model prediction
===============================

.. code-block:: python

   from clarifai.rest import ClarifaiApp
   from clarifai.rest import FeedbackInfo

   app = ClarifaiApp()

   # positive feedback: this is a dog
   general_model_id = "aaa03c23b3724a16a56b629203edc62c"
   m = app.models.get(model_id=general_model_id)

   m.send_concept_feedback(input_id='id1', url='https://samples.clarifai.com/dog2.jpeg',
                           concepts=['dog', 'animal'],
                           feedback_info=FeedbackInfo(output_id='OID',
                                                      session_id='SID',
                                                      end_user_id='UID',
                                                      event_type='annotation'))

   # negative feedback: this is not a cat
   m = app.models.get(model_id=general_model_id)

   m.send_concept_feedback(input_id='id1', url='https://samples.clarifai.com/dog2.jpeg',
                           not_concepts=['cat', 'kitty'],
                           feedback_info=FeedbackInfo(output_id='OID',
                                                      session_id='SID',
                                                      end_user_id='UID',
                                                      event_type='annotation'))

   # all together: this is a dog but not a cat
   m = app.models.get(model_id=general_model_id)

   m.send_concept_feedback(input_id='id1', url='https://samples.clarifai.com/dog2.jpeg',
                           concepts=['dog'], not_concepts=['cat', 'kitty'],
                           feedback_info=FeedbackInfo(output_id='OID',
                                                      session_id='SID',
                                                      end_user_id='UID',
                                                      event_type='annotation'))


Detection model prediction
===============================

.. code-block:: python

   from clarifai.rest import ClarifaiApp
   from clarifai.rest import FeedbackInfo
   from clarifai.rest import Region, RegionInfo, BoundingBox, Concept

   app = ClarifaiApp()

   m.send_region_feedback(input_id='id2', url='https://portal.clarifai.com/developer/static/images/model-samples/celeb-001.jpg',
                          regions=[Region(region_info=RegionInfo(bbox=BoundingBox(top_row=0.1,
                                                                                  left_col=0.2,
                                                                                  bottom_row=0.5,
                                                                                  right_col=0.5)),
                                          concepts=[Concept(concept_id='people', value=True),
                                                    Concept(concept_id='portrait', value=True)])],
                          feedback_info=FeedbackInfo(output_id='OID',
                                                     session_id='SID',
                                                     end_user_id='UID',
                                                     event_type='annotation'))


Face detection model prediction
================================


#
# send feedback for celebrity model
#

.. code-block:: python

   from clarifai.rest import ClarifaiApp
   from clarifai.rest import FeedbackInfo
   from clarifai.rest import Region, RegionInfo, BoundingBox, Concept
   from clarifai.rest import Face, FaceIdentity
   from clarifai.rest import FaceAgeAppearance, FaceGenderAppearance, FaceMulticulturalAppearance

   app = ClarifaiApp()

   #
   # send feedback for celebrity model
   #
   m.send_region_feedback(input_id='id2', url='https://developer.clarifai.com/static/images/model-samples/celeb-001.jpg',
                          regions=[Region(region_info=RegionInfo(bbox=BoundingBox(top_row=0.1,
                                                                                  left_col=0.2,
                                                                                  bottom_row=0.5,
                                                                                  right_col=0.5)),
                                          face=Face(identity=FaceIdentity([Concept(concept_id='celeb1', value=True)]))
                                          )
                                   ],
                          feedback_info=FeedbackInfo(output_id='OID',
                                                     session_id='SID',
                                                     end_user_id='UID',
                                                     event_type='annotation'))

   #
   # send feedback for age, gender, multicultural appearance
   #

   m.send_region_feedback(input_id='id2', url='https://developer.clarifai.com/static/images/model-samples/celeb-001.jpg',
                          regions=[Region(region_info=RegionInfo(bbox=BoundingBox(top_row=0.1,
                                                                                  left_col=0.2,
                                                                                  bottom_row=0.5,
                                                                                  right_col=0.5)),
                                          face=Face(age_appearance=FaceAgeAppearance([Concept(concept_id='20', value=True),
                                                                                      Concept(concept_id='30', value=False)
                                                                                      ]),
                                                    gender_appearance=FaceGenderAppearance([Concept(concept_id='male', value=True)]),
                                                    multicultural_appearance=FaceMulticulturalAppearance([Concept(concept_id='asian', value=True)])
                                                   )
                                          )
                                   ],
                          feedback_info=FeedbackInfo(output_id='OID',
                                                     session_id='SID',
                                                     end_user_id='UID',
                                                     event_type='annotation'))
