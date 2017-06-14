==================
Installation Guide
==================

.. note:: Generally, the python client works with Python 2.x and Python 3.x. But it is only tested against 2.7 and 3.5. Feel free to report BUGs if you encounter on other versions.

Install the package
===================

You can install the latest stable clarifai using pip (which is the canonical way to install Python
packages). To install using ``pip`` run::

   pip install clarifai==2.0.27

You can also install from git using latest source::

   pip install git+git://github.com/Clarifai/clarifai-python.git

Configuration
=============

The client uses CLARIFAI_APP_ID and CLARIFAI_APP_SECRET for authentication and token generation.
Each application you create uses its own unique ID and secret to authenticate requests.
The client will use the authentication information passed to it by three methods with the following precedence order::
+ Passed in to the constructor through the `app_id` and `app_secret` parameters.
+ Set as the CLARIFAI_APP_ID and CLARIFAI_APP_SECRET environment variables
+ Placed in the .clarifai/config file using the command below.
You can get these values from https://developer.clarifai.com/account/applications and then run::

   $ clarifai config
   CLARIFAI_APP_ID: []: ************************************YQEd
   CLARIFAI_APP_SECRET: []: ************************************gCqT

If you do not see any error message after this, you are all set and can proceed with using the client.

Windows Users
=============

For Windows users, you may fail running the ``clarifai config`` when you try to configure the runtime environment.
This is because Windows uses file extension to determine executables and by default file ``clarifai`` without file
extension is nonexecutables.
In order to run the command, you may want to launch it with the python interpreter.

.. code-block:: bash

    C:\Python27>python.exe Scripts\clarifai config
    CLARIFAI_APP_ID: []: ************************************YQEd
    CLARIFAI_APP_SECRET: []: ************************************gCqT

AWS Lambda Users
================

For AWS Lambda users, in order to use the library correctly, you are recommended to set two
environmental variables `CLARIFAI_APP_ID` and `CLARIFAI_APP_SECRET` in the lambda function
configuration, or hardcode the APP_ID and APP_SECRET in the API instantiation.

