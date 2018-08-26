==================
Installation Guide
==================

.. note:: Generally, the python client works with Python 2.x and Python 3.x. But it is only tested against
          2.7, 3.5, 3.6 and 3.7. Feel free to report any bugs you encounter in any other version.

Install the package
===================

You can install the latest stable clarifai using pip (which is the canonical way to install Python
packages). To install using ``pip``, run::

   pip install clarifai --upgrade

You can also install directly from Github (though we recommend the previous approach)::

   pip install git+git://github.com/Clarifai/clarifai-python.git

Configuration
=============
Generate your Clarifai API key [on the API keys page](https://clarifai.com/developer/account/keys). The client uses
it for authentication.

You can use three methods to pass the API key to your client (if you use more than one, the first in the following
precedence order will be used):

1. Pass it to the ``ClarifaiApp`` constructor through the ``api_key`` parameter.
2. Set it as the ``CLARIFAI_API_KEY`` environment variable.
3. Place it in the ``.clarifai/config`` file using the following command (see exceptions for Windows below)::

       $ ./scripts/clarifai config
       CLARIFAI_API_KEY: []: ************************************YQEd

Windows Users
-------------

For Windows users, running ``./scripts/clarifai config`` may fail when you try to configure the runtime environment.
This is because Windows uses the file extension to determine executables and by default, file ``clarifai`` without file
extension is non-executable.
In order to run the command, you may want to launch it with the python interpreter.

.. code-block:: bash

    C:\Python27\python.exe Scripts\clarifai config
    CLARIFAI_API_KEY: []: ************************************YQEd

AWS Lambda Users
================

For AWS Lambda users, in order to use the library correctly, you are recommended to set an
environmental variable `CLARIFAI_API_KEY` in the lambda function
configuration, or "hardcode" the ``api_key`` parameter to the ``ClarifaiApp`` constructor.
